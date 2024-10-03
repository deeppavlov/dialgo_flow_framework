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

import asyncio
from typing import Dict, Hashable, Set, Tuple, Optional, List

try:
    from pymongo import UpdateOne
    from pymongo.collection import Collection
    from motor.motor_asyncio import AsyncIOMotorClient

    mongo_available = True
except ImportError:
    mongo_available = False

from .database import DBContextStorage, FieldConfig
from .protocol import get_protocol_install_suggestion


class MongoContextStorage(DBContextStorage):
    """
    Implements :py:class:`.DBContextStorage` with `mongodb` as the database backend.

    CONTEXTS table is stored as `COLLECTION_PREFIX_contexts` collection.
    LOGS table is stored as `COLLECTION_PREFIX_logs` collection.

    :param path: Database URI. Example: `mongodb://user:password@host:port/dbname`.
    :param context_schema: Context schema for this storage.
    :param serializer: Serializer that will be used for serializing contexts.
    :param collection_prefix: "namespace" prefix for the two collections created for context storing.
    """

    _UNIQUE_KEYS = "unique_keys"
    _ID_FIELD = "_id"

    is_asynchronous = True

    def __init__(
        self,
        path: str,
        rewrite_existing: bool = False,
        configuration: Optional[Dict[str, FieldConfig]] = None,
        collection_prefix: str = "chatsky_collection",
    ):
        DBContextStorage.__init__(self, path, rewrite_existing, configuration)

        if not mongo_available:
            install_suggestion = get_protocol_install_suggestion("mongodb")
            raise ImportError("`mongodb` package is missing.\n" + install_suggestion)
        self._mongo = AsyncIOMotorClient(self.full_path, uuidRepresentation="standard")
        db = self._mongo.get_default_database()

        self._main_table = db[f"{collection_prefix}_{self._main_table_name}"]
        self._turns_table = db[f"{collection_prefix}_{self._turns_table_name}"]
        self._misc_table = db[f"{collection_prefix}_{self._misc_table_name}"]

        asyncio.run(
            asyncio.gather(
                self._main_table.create_index(
                    self._id_column_name, background=True, unique=True
                ),
                self._turns_table.create_index(
                    [self._id_column_name, self._key_column_name], background=True, unique=True
                ),
                self._misc_table.create_index(
                    [self._id_column_name, self._key_column_name], background=True, unique=True
                )
            )
        )

    def _get_config_for_field(self, field_name: str) -> Tuple[Collection, str, FieldConfig]:
        if field_name == self.labels_config.name:
            return self._turns_table, field_name, self.labels_config
        elif field_name == self.requests_config.name:
            return self._turns_table, field_name, self.requests_config
        elif field_name == self.responses_config.name:
            return self._turns_table, field_name, self.responses_config
        elif field_name == self.misc_config.name:
            return self._misc_table, self._value_column_name, self.misc_config
        else:
            raise ValueError(f"Unknown field name: {field_name}!")

    async def load_main_info(self, ctx_id: str) -> Optional[Tuple[int, int, int, bytes]]:
        result = await self._main_table.find_one(
            {self._id_column_name: ctx_id},
            [self._current_turn_id_column_name, self._created_at_column_name, self._updated_at_column_name, self._framework_data_column_name]
        )
        return (result[self._current_turn_id_column_name], result[self._created_at_column_name], result[self._updated_at_column_name], result[self._framework_data_column_name]) if result is not None else None

    async def update_main_info(self, ctx_id: str, turn_id: int, crt_at: int, upd_at: int, fw_data: bytes) -> None:
        await self._main_table.update_one(
            {self._id_column_name: ctx_id},
            {
                "$set": {
                    self._id_column_name: ctx_id,
                    self._current_turn_id_column_name: turn_id,
                    self._created_at_column_name: crt_at,
                    self._updated_at_column_name: upd_at,
                    self._framework_data_column_name: fw_data,
                }
            },
            upsert=True,
        )

    async def delete_context(self, ctx_id: str) -> None:
        await asyncio.gather(
            self._main_table.delete_one({self._id_column_name: ctx_id}),
            self._turns_table.delete_one({self._id_column_name: ctx_id}),
            self._misc_table.delete_one({self._id_column_name: ctx_id})
        )

    async def load_field_latest(self, ctx_id: str, field_name: str) -> List[Tuple[Hashable, bytes]]:
        field_table, field_name, field_config = self._get_config_for_field(field_name)
        sort, limit, key = None, 0, dict()
        if field_table == self._turns_table:
            sort = [(self._key_column_name, -1)]
        if isinstance(field_config.subscript, int):
            limit = field_config.subscript
        if isinstance(field_config.subscript, Set):
            key = {self._key_column_name: {"$in": list(field_config.subscript)}}
        result = await field_table.find(
            {self._id_column_name: ctx_id, field_name: {"$exists": True, "$ne": None}, **key},
            [self._key_column_name, field_name],
            sort=sort
        ).limit(limit).to_list(None)
        return [(item[self._key_column_name], item[field_name]) for item in result]

    async def load_field_keys(self, ctx_id: str, field_name: str) -> List[Hashable]:
        field_table, field_name, _ = self._get_config_for_field(field_name)
        result = await field_table.aggregate(
            [
                {"$match": {self._id_column_name: ctx_id, field_name: {"$ne": None}}},
                {"$group": {"_id": None, self._UNIQUE_KEYS: {"$addToSet": f"${self._key_column_name}"}}},
            ]
        ).to_list(None)
        return result[0][self._UNIQUE_KEYS] if len(result) == 1 else list()

    async def load_field_items(self, ctx_id: str, field_name: str, keys: Set[Hashable]) -> List[bytes]:
        field_table, field_name, _ = self._get_config_for_field(field_name)
        result = await field_table.find(
            {self._id_column_name: ctx_id, self._key_column_name: {"$in": list(keys)}, field_name: {"$exists": True, "$ne": None}},
            [self._key_column_name, field_name]
        ).to_list(None)
        return [(item[self._key_column_name], item[field_name]) for item in result]

    async def update_field_items(self, ctx_id: str, field_name: str, items: List[Tuple[Hashable, bytes]]) -> None:
        field_table, field_name, _ = self._get_config_for_field(field_name)
        if len(items) == 0:
            return
        await field_table.bulk_write(
            [
                UpdateOne(
                    {self._id_column_name: ctx_id, self._key_column_name: k},
                    {"$set": {field_name: v}},
                    upsert=True,
                ) for k, v in items
            ]
        )

    async def clear_all(self) -> None:
        await asyncio.gather(
            self._main_table.delete_many({}),
            self._turns_table.delete_many({}),
            self._misc_table.delete_many({})
        )
