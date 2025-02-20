"""
Redis
-----
The Redis module provides a Redis-based version of the :py:class:`.DBContextStorage` class.
This class is used to store and retrieve context data in a Redis.
It allows Chatsky to easily store and retrieve context data in a format that is highly scalable
and easy to work with.

Redis is an open-source, in-memory data structure store that is known for its
high performance and scalability. It stores data in key-value pairs and supports a variety of data
structures such as strings, hashes, lists, sets, and more.
Additionally, Redis can be used as a cache, message broker, and database, making it a versatile
and powerful choice for data storage and management.
"""

from asyncio import gather
from typing import Any, Dict, List, Set, Tuple, Optional

try:
    from redis.asyncio import Redis

    redis_available = True
except ImportError:
    redis_available = False

from .database import DBContextStorage, _SUBSCRIPT_DICT, NameConfig
from .protocol import get_protocol_install_suggestion


class RedisContextStorage(DBContextStorage):
    """
    Implements :py:class:`.DBContextStorage` with `redis` as the database backend.

    The main context info is stored in redis hashes, one for each context.
    The `TURNS` table values are stored in redis hashes, one for each field.

    That's how MAIN table fields are stored:
    `"KEY_PREFIX:main:ctx_id": "DATA"`
    That's how TURNS table fields are stored:
    `"KEY_PREFIX:turns:ctx_id:FIELD_NAME": "DATA"`

    :param path: Database URI string. Example: `redis://user:password@host:port`.
    :param rewrite_existing: Whether `TURNS` modified locally should be updated in database or not.
    :param partial_read_config: Dictionary of subscripts for all possible turn items.
    :param key_prefix: "namespace" prefix for all keys, should be set for efficient clearing of all data.
    """

    is_concurrent: bool = True

    def __init__(
        self,
        path: str,
        rewrite_existing: bool = False,
        partial_read_config: Optional[_SUBSCRIPT_DICT] = None,
        key_prefix: str = "chatsky_keys",
    ):
        DBContextStorage.__init__(self, path, rewrite_existing, partial_read_config)

        if not redis_available:
            install_suggestion = get_protocol_install_suggestion("redis")
            raise ImportError("`redis` package is missing.\n" + install_suggestion)
        if not bool(key_prefix):
            raise ValueError("`key_prefix` parameter shouldn't be empty")
        self.database = Redis.from_url(self.full_path)

        self._prefix = key_prefix
        self._main_key = f"{key_prefix}:{NameConfig._main_table}"
        self._turns_key = f"{key_prefix}:{NameConfig._turns_table}"

    async def _connect(self):
        pass

    @staticmethod
    def _keys_to_bytes(keys: List[int]) -> List[bytes]:
        return [str(f).encode("utf-8") for f in keys]

    @staticmethod
    def _bytes_to_keys(keys: List[bytes]) -> List[int]:
        return [int(f.decode("utf-8")) for f in keys]

    async def _load_main_info(self, ctx_id: str) -> Optional[Dict[str, Any]]:
        if await self.database.exists(f"{self._main_key}:{ctx_id}"):
            retrieved_fields = await gather(
                *[self.database.hget(f"{self._main_key}:{ctx_id}", f) for f in NameConfig.get_context_main_fields]
            )
            return {f: v for f, v in zip(NameConfig.get_context_main_fields, retrieved_fields)}
        else:
            return None

    async def _update_context(
        self,
        ctx_id: str,
        ctx_info: Optional[Dict[str, Any]],
        field_info: List[Tuple[str, List[Tuple[int, Optional[bytes]]]]],
    ) -> None:
        update_main, update_values, delete_keys = list(), list(), list()
        if ctx_info is not None:
            update_main = [
                (f, ctx_info[f] if isinstance(ctx_info[f], bytes) else str(ctx_info[f]))
                for f in NameConfig.get_context_main_fields
            ]
        for field_name, items in field_info:
            new_delete_keys = list()
            for k, v in items:
                if v is None:
                    new_delete_keys += [k]
                else:
                    update_values += [(field_name, k, v)]
            if len(new_delete_keys) > 0:
                field_key = f"{self._turns_key}:{ctx_id}:{field_name}"
                valid_keys = [
                    k for k in await self.database.hkeys(field_key) if k in self._keys_to_bytes(new_delete_keys)
                ]
                delete_keys += [(field_key, valid_keys)]
        await gather(
            *[self.database.hset(f"{self._main_key}:{ctx_id}", f, u) for f, u in update_main],
            *[self.database.hset(f"{self._turns_key}:{ctx_id}:{f}", str(k), v) for f, k, v in update_values],
            *[self.database.hdel(f, *k) for f, k in delete_keys],
        )

    async def _delete_context(self, ctx_id: str) -> None:
        keys = await self.database.keys(f"{self._prefix}:*:{ctx_id}*")
        if len(keys) > 0:
            await self.database.delete(*keys)

    async def _load_field_latest(self, ctx_id: str, field_name: str) -> List[Tuple[int, bytes]]:
        field_key = f"{self._turns_key}:{ctx_id}:{field_name}"
        keys = sorted(await self.database.hkeys(field_key), key=lambda k: int(k), reverse=True)
        if isinstance(self._subscripts[field_name], int):
            keys = keys[: self._subscripts[field_name]]
        elif isinstance(self._subscripts[field_name], Set):
            keys = [k for k in keys if k in self._keys_to_bytes(self._subscripts[field_name])]
        values = await gather(*[self.database.hget(field_key, k) for k in keys])
        return [(k, v) for k, v in zip(self._bytes_to_keys(keys), values)]

    async def _load_field_keys(self, ctx_id: str, field_name: str) -> List[int]:
        return self._bytes_to_keys(await self.database.hkeys(f"{self._turns_key}:{ctx_id}:{field_name}"))

    async def _load_field_items(self, ctx_id: str, field_name: str, keys: List[int]) -> List[Tuple[int, bytes]]:
        field_key = f"{self._turns_key}:{ctx_id}:{field_name}"
        load = [k for k in await self.database.hkeys(field_key) if k in self._keys_to_bytes(keys)]
        values = await gather(*[self.database.hget(field_key, k) for k in load])
        return [(k, v) for k, v in zip(self._bytes_to_keys(load), values)]

    async def _clear_all(self) -> None:
        keys = await self.database.keys(f"{self._prefix}:*")
        if len(keys) > 0:
            await self.database.delete(*keys)
