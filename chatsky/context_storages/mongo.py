"""
Mongo
-----
The Mongo module provides a MongoDB-based version of the :py:class:`.DBContextStorage` class.
This class is used to store and retrieve context data in a MongoDB.
It allows Chatsky to easily store and retrieve context data in a format that is highly scalable
and easy to work with.

MongoDB is a widely-used, open-source NoSQL database that is known for its scalability and performance.
It stores data in a format similar to JSON, making it easy to work with the data in a variety of programming languages
and environments. Additionally, MongoDB is highly scalable and can handle large amounts of data
and high levels of read and write traffic.
"""

from asyncio import gather
from typing import Any, Dict, Set, Tuple, Optional, List

try:
    from pymongo import UpdateOne
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorClientSession

    mongo_available = True
except ImportError:
    AsyncIOMotorClientSession = Any

    mongo_available = False

from .database import DBContextStorage, _SUBSCRIPT_DICT, NameConfig
from .protocol import get_protocol_install_suggestion


class MongoContextStorage(DBContextStorage):
    """
    Implements :py:class:`.DBContextStorage` with `mongodb` as the database backend.

    CONTEXTS table is stored as `COLLECTION_PREFIX_contexts` collection.
    LOGS table is stored as `COLLECTION_PREFIX_logs` collection.

    :param path: Database URI. Example: `mongodb://user:password@host:port/dbname`.
    :param rewrite_existing: Whether `TURNS` modified locally should be updated in database or not.
    :param partial_read_config: Dictionary of subscripts for all possible turn items.
    :param collection_prefix: "namespace" prefix for the two collections created for context storing.
    """

    _UNIQUE_KEYS = "unique_keys"
    _ID_FIELD = "_id"

    is_concurrent: bool = True

    def __init__(
        self,
        path: str,
        rewrite_existing: bool = False,
        partial_read_config: Optional[_SUBSCRIPT_DICT] = None,
        collection_prefix: str = "chatsky_collection",
        transactions_enabled: bool = False,
    ):
        DBContextStorage.__init__(self, path, rewrite_existing, partial_read_config)

        if not mongo_available:
            install_suggestion = get_protocol_install_suggestion("mongodb")
            raise ImportError("`mongodb` package is missing.\n" + install_suggestion)
        self._transactions_enabled = transactions_enabled
        self._mongo = AsyncIOMotorClient(self.full_path, uuidRepresentation="standard")
        db = self._mongo.get_default_database()

        self.main_table = db[f"{collection_prefix}_{NameConfig._main_table}"]
        self.turns_table = db[f"{collection_prefix}_{NameConfig._turns_table}"]

    async def _connect(self):
        await gather(
            self.main_table.create_index(NameConfig._id_column, background=True, unique=True),
            self.turns_table.create_index(
                [NameConfig._id_column, NameConfig._key_column], background=True, unique=True
            ),
        )

    async def _load_main_info(self, ctx_id: str) -> Optional[Dict[str, Any]]:
        result = await self.main_table.find_one(
            {NameConfig._id_column: ctx_id},
            NameConfig.get_context_main_fields,
        )
        return {f: result[f] for f in NameConfig.get_context_main_fields} if result is not None else None

    async def _inner_update_context(
        self,
        ctx_id: str,
        ctx_info_dump: Optional[Dict[str, Any]],
        field_info: List[Tuple[str, List[Tuple[int, Optional[bytes]]]]],
        session: Optional[AsyncIOMotorClientSession],
    ) -> None:
        if ctx_info_dump is not None:
            await self.main_table.update_one(
                {NameConfig._id_column: ctx_id},
                {
                    "$set": {
                        NameConfig._id_column: ctx_id,
                    }
                    | {f: ctx_info_dump[f] for f in NameConfig.get_context_main_fields}
                },
                upsert=True,
                session=session,
            )
        if len(field_info) > 0:
            await self.turns_table.bulk_write(
                [
                    UpdateOne(
                        {NameConfig._id_column: ctx_id, NameConfig._key_column: k},
                        {"$set": {field_name: v}},
                        upsert=True,
                    )
                    for field_name, items in field_info
                    for k, v in items
                ],
                session=session,
            )

    async def _update_context(
        self,
        ctx_id: str,
        ctx_info: Optional[Dict[str, Any]],
        field_info: List[Tuple[str, List[Tuple[int, Optional[bytes]]]]],
    ) -> None:
        if self._transactions_enabled:
            async with await self._mongo.start_session() as session:
                async with session.start_transaction():
                    await self._inner_update_context(ctx_id, ctx_info, field_info, session)
        else:
            await self._inner_update_context(ctx_id, ctx_info, field_info, None)

    async def _delete_context(self, ctx_id: str) -> None:
        await gather(
            self.main_table.delete_one({NameConfig._id_column: ctx_id}),
            self.turns_table.delete_one({NameConfig._id_column: ctx_id}),
        )

    async def _load_field_latest(self, ctx_id: str, field_name: str) -> List[Tuple[int, bytes]]:
        limit, key = 0, dict()
        if isinstance(self._subscripts[field_name], int):
            limit = self._subscripts[field_name]
        elif isinstance(self._subscripts[field_name], Set):
            key = {NameConfig._key_column: {"$in": list(self._subscripts[field_name])}}
        result = (
            await self.turns_table.find(
                {NameConfig._id_column: ctx_id, field_name: {"$exists": True, "$ne": None}, **key},
                [NameConfig._key_column, field_name],
                sort=[(NameConfig._key_column, -1)],
            )
            .limit(limit)
            .to_list(None)
        )
        return [(item[NameConfig._key_column], item[field_name]) for item in result]

    async def _load_field_keys(self, ctx_id: str, field_name: str) -> List[int]:
        result = await self.turns_table.aggregate(
            [
                {"$match": {NameConfig._id_column: ctx_id, field_name: {"$ne": None}}},
                {"$group": {"_id": None, self._UNIQUE_KEYS: {"$addToSet": f"${NameConfig._key_column}"}}},
            ]
        ).to_list(None)
        return result[0][self._UNIQUE_KEYS] if len(result) == 1 else list()

    async def _load_field_items(self, ctx_id: str, field_name: str, keys: Set[int]) -> List[Tuple[int, bytes]]:
        result = await self.turns_table.find(
            {
                NameConfig._id_column: ctx_id,
                NameConfig._key_column: {"$in": list(keys)},
                field_name: {"$exists": True, "$ne": None},
            },
            [NameConfig._key_column, field_name],
        ).to_list(None)
        return [(item[NameConfig._key_column], item[field_name]) for item in result]

    async def _clear_all(self) -> None:
        await gather(self.main_table.delete_many({}), self.turns_table.delete_many({}))
