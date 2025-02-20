"""
Yandex DB
---------
The Yandex DB module provides a version of the :py:class:`.DBContextStorage` class that designed to work with
Yandex and other databases. Yandex DataBase is a fully-managed cloud-native SQL service that makes it easy to set up,
operate, and scale high-performance and high-availability databases for your applications.

The Yandex DB module uses the Yandex Cloud SDK, which is a python library that allows you to work
with Yandex Cloud services using python. This allows Chatsky to easily integrate with the Yandex DataBase and
take advantage of the scalability and high-availability features provided by the service.
"""

from asyncio import gather
from os.path import join
from typing import Any, Awaitable, Callable, Dict, Set, Tuple, List, Optional
from urllib.parse import urlsplit

try:
    from ydb import (
        SerializableReadWrite,
        SchemeError,
        TableDescription,
        Column,
        OptionalType,
        PrimitiveType,
    )
    from ydb.aio import Driver, SessionPool
    from ydb.table import Session

    ydb_available = True
except ImportError:
    ydb_available = False

from .database import DBContextStorage, _SUBSCRIPT_DICT, NameConfig
from .protocol import get_protocol_install_suggestion


class YDBContextStorage(DBContextStorage):
    """
    Version of the :py:class:`.DBContextStorage` for YDB.

    `CONTEXT` table is represented by `contexts` table.
    Columns of the table are: `id`, `current_turn_id`, `created_at` `updated_at`, `misc` and `framework_data`.

    `TURNS` table is represented by `turns` table.
    olumns of the table are: `id`, `key`, `label`, `request` and `response`.

    :param path: Standard sqlalchemy URI string. One of `grpc` or `grpcs` can be chosen as a protocol.
        Example: `grpc://localhost:2134/local`.
        NB! Do not forget to provide credentials in environmental variables
        or set `YDB_ANONYMOUS_CREDENTIALS` variable to `1`!
    :param rewrite_existing: Whether `TURNS` modified locally should be updated in database or not.
    :param partial_read_config: Dictionary of subscripts for all possible turn items.
    :param table_name_prefix: "namespace" prefix for the two tables created for context storing.
    :param timeout: Waiting timeout for the database driver.
    """

    _LIMIT_VAR = "limit"
    _KEY_VAR = "key"

    is_concurrent: bool = True

    def __init__(
        self,
        path: str,
        rewrite_existing: bool = False,
        partial_read_config: Optional[_SUBSCRIPT_DICT] = None,
        table_name_prefix: str = "chatsky_table",
        timeout: int = 5,
    ):
        DBContextStorage.__init__(self, path, rewrite_existing, partial_read_config)

        protocol, netloc, self.database, _, _ = urlsplit(path)
        if not ydb_available:
            install_suggestion = get_protocol_install_suggestion("grpc")
            raise ImportError("`ydb` package is missing.\n" + install_suggestion)

        self.table_prefix = table_name_prefix
        self._timeout = timeout
        self._endpoint = f"{protocol}://{netloc}"

    async def _connect(self) -> None:
        self._driver = Driver(endpoint=self._endpoint, database=self.database)
        client_settings = self._driver.table_client._table_client_settings.with_allow_truncated_result(True)
        self._driver.table_client._table_client_settings = client_settings
        await self._driver.wait(fail_fast=True, timeout=self._timeout)

        self.pool = SessionPool(self._driver, size=10)
        self.main_table = f"{self.table_prefix}_{NameConfig._main_table}"
        self.turns_table = f"{self.table_prefix}_{NameConfig._turns_table}"

        if not await self._does_table_exist(self.main_table):
            await self._create_main_table(self.main_table)
        if not await self._does_table_exist(self.turns_table):
            await self._create_turns_table(self.turns_table)

    async def _does_table_exist(self, table_name: str) -> bool:
        async def callee(session: Session) -> None:
            await session.describe_table(join(self.database, table_name))

        try:
            await self.pool.retry_operation(callee)
            return True
        except SchemeError:
            return False

    async def _create_main_table(self, table_name: str) -> None:
        async def callee(session: Session) -> None:
            await session.create_table(
                "/".join([self.database, table_name]),
                TableDescription()
                .with_column(Column(NameConfig._id_column, PrimitiveType.Utf8))
                .with_column(Column(NameConfig._current_turn_id_column, PrimitiveType.Uint64))
                .with_column(Column(NameConfig._created_at_column, PrimitiveType.Uint64))
                .with_column(Column(NameConfig._updated_at_column, PrimitiveType.Uint64))
                .with_column(Column(NameConfig._misc_column, PrimitiveType.String))
                .with_column(Column(NameConfig._framework_data_column, PrimitiveType.String))
                .with_primary_key(NameConfig._id_column),
            )

        await self.pool.retry_operation(callee)

    async def _create_turns_table(self, table_name: str) -> None:
        async def callee(session: Session) -> None:
            await session.create_table(
                "/".join([self.database, table_name]),
                TableDescription()
                .with_column(Column(NameConfig._id_column, PrimitiveType.Utf8))
                .with_column(Column(NameConfig._key_column, PrimitiveType.Uint32))
                .with_column(Column(NameConfig._labels_field, OptionalType(PrimitiveType.String)))
                .with_column(Column(NameConfig._requests_field, OptionalType(PrimitiveType.String)))
                .with_column(Column(NameConfig._responses_field, OptionalType(PrimitiveType.String)))
                .with_primary_keys(NameConfig._id_column, NameConfig._key_column),
            )

        await self.pool.retry_operation(callee)

    async def _load_main_info(self, ctx_id: str) -> Optional[Dict[str, Any]]:
        async def callee(session: Session) -> Optional[Dict[str, Any]]:
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${NameConfig._id_column} AS Utf8;
                SELECT {NameConfig._current_turn_id_column}, {NameConfig._created_at_column}, {NameConfig._updated_at_column}, {NameConfig._misc_column}, {NameConfig._framework_data_column}
                FROM {self.main_table}
                WHERE {NameConfig._id_column} = ${NameConfig._id_column};
                """  # noqa: E501
            result_sets = await session.transaction().execute(
                await session.prepare(query),
                {
                    f"${NameConfig._id_column}": ctx_id,
                },
                commit_tx=True,
            )
            return (
                {f: result_sets[0].rows[0][f] for f in NameConfig.get_context_main_fields}
                if len(result_sets[0].rows) > 0
                else None
            )

        return await self.pool.retry_operation(callee)

    async def _update_context(
        self,
        ctx_id: str,
        ctx_info: Optional[Dict[str, Any]],
        field_info: List[Tuple[str, List[Tuple[int, Optional[bytes]]]]],
    ) -> None:
        async def callee(session: Session) -> None:
            transaction = await session.transaction(SerializableReadWrite()).begin()
            if ctx_info is not None:
                query = f"""
                    PRAGMA TablePathPrefix("{self.database}");
                    DECLARE ${NameConfig._id_column} AS Utf8;
                    DECLARE ${NameConfig._current_turn_id_column} AS Uint64;
                    DECLARE ${NameConfig._created_at_column} AS Uint64;
                    DECLARE ${NameConfig._updated_at_column} AS Uint64;
                    DECLARE ${NameConfig._misc_column} AS String;
                    DECLARE ${NameConfig._framework_data_column} AS String;
                    UPSERT INTO {self.main_table} ({NameConfig._id_column}, {NameConfig._current_turn_id_column}, {NameConfig._created_at_column}, {NameConfig._updated_at_column}, {NameConfig._misc_column}, {NameConfig._framework_data_column})
                    VALUES (${NameConfig._id_column}, ${NameConfig._current_turn_id_column}, ${NameConfig._created_at_column}, ${NameConfig._updated_at_column}, ${NameConfig._misc_column}, ${NameConfig._framework_data_column});
                    """  # noqa: E501
                await transaction.execute(
                    await session.prepare(query),
                    {
                        f"${NameConfig._id_column}": ctx_id,
                    }
                    | {f"${f}": ctx_info[f] for f in NameConfig.get_context_main_fields},
                )
            for field_name, items in field_info:
                declare, prepare, values = list(), dict(), list()
                for i, (k, v) in enumerate(items):
                    declare += [f"DECLARE ${self._KEY_VAR}_{i} AS Uint32;"]
                    prepare.update({f"${self._KEY_VAR}_{i}": k})
                    if v is not None:
                        declare += [f"DECLARE ${field_name}_{i} AS String;"]
                        prepare.update({f"${field_name}_{i}": v})
                        value_param = f"${field_name}_{i}"
                    else:
                        value_param = "NULL"
                    values += [f"(${NameConfig._id_column}, ${self._KEY_VAR}_{i}, {value_param})"]
                query = f"""
                    PRAGMA TablePathPrefix("{self.database}");
                    DECLARE ${NameConfig._id_column} AS Utf8;
                    {" ".join(declare)}
                    UPSERT INTO {self.turns_table} ({NameConfig._id_column}, {NameConfig._key_column}, {field_name})
                    VALUES {", ".join(values)};
                    """  # noqa: E501
                await transaction.execute(
                    await session.prepare(query),
                    {
                        f"${NameConfig._id_column}": ctx_id,
                        **prepare,
                    },
                )
            await transaction.commit()

        await self.pool.retry_operation(callee)

    async def _delete_context(self, ctx_id: str) -> None:
        def construct_callee(table_name: str) -> Callable[[Session], Awaitable[None]]:
            async def callee(session: Session) -> None:
                query = f"""
                    PRAGMA TablePathPrefix("{self.database}");
                    DECLARE ${NameConfig._id_column} AS Utf8;
                    DELETE FROM {table_name}
                    WHERE {NameConfig._id_column} = ${NameConfig._id_column};
                    """  # noqa: E501
                await session.transaction().execute(
                    await session.prepare(query),
                    {
                        f"${NameConfig._id_column}": ctx_id,
                    },
                    commit_tx=True,
                )

            return callee

        await gather(
            self.pool.retry_operation(construct_callee(self.main_table)),
            self.pool.retry_operation(construct_callee(self.turns_table)),
        )

    async def _load_field_latest(self, ctx_id: str, field_name: str) -> List[Tuple[int, bytes]]:
        async def callee(session: Session) -> List[Tuple[int, bytes]]:
            declare, prepare, limit, key = list(), dict(), "", ""
            if isinstance(self._subscripts[field_name], int):
                declare += [f"DECLARE ${self._LIMIT_VAR} AS Uint64;"]
                prepare.update({f"${self._LIMIT_VAR}": self._subscripts[field_name]})
                limit = f"LIMIT ${self._LIMIT_VAR}"
            elif isinstance(self._subscripts[field_name], Set):
                values = list()
                for i, k in enumerate(self._subscripts[field_name]):
                    declare += [f"DECLARE ${self._KEY_VAR}_{i} AS Uint32;"]
                    prepare.update({f"${self._KEY_VAR}_{i}": k})
                    values += [f"${self._KEY_VAR}_{i}"]
                key = f"AND {NameConfig._key_column} IN ({', '.join(values)})"
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${NameConfig._id_column} AS Utf8;
                {" ".join(declare)}
                SELECT {NameConfig._key_column}, {field_name}
                FROM {self.turns_table}
                WHERE {NameConfig._id_column} = ${NameConfig._id_column} AND {field_name} IS NOT NULL {key}
                ORDER BY {NameConfig._key_column} DESC {limit};
                """  # noqa: E501
            result_sets = await session.transaction().execute(
                await session.prepare(query),
                {
                    f"${NameConfig._id_column}": ctx_id,
                    **prepare,
                },
                commit_tx=True,
            )
            return (
                [(e[NameConfig._key_column], e[field_name]) for e in result_sets[0].rows]
                if len(result_sets[0].rows) > 0
                else list()
            )

        return await self.pool.retry_operation(callee)

    async def _load_field_keys(self, ctx_id: str, field_name: str) -> List[int]:
        async def callee(session: Session) -> List[int]:
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${NameConfig._id_column} AS Utf8;
                SELECT {NameConfig._key_column}
                FROM {self.turns_table}
                WHERE {NameConfig._id_column} = ${NameConfig._id_column} AND {field_name} IS NOT NULL;
                """  # noqa: E501
            result_sets = await session.transaction().execute(
                await session.prepare(query),
                {
                    f"${NameConfig._id_column}": ctx_id,
                },
                commit_tx=True,
            )
            return [e[NameConfig._key_column] for e in result_sets[0].rows] if len(result_sets[0].rows) > 0 else list()

        return await self.pool.retry_operation(callee)

    async def _load_field_items(self, ctx_id: str, field_name: str, keys: List[int]) -> List[Tuple[int, bytes]]:
        async def callee(session: Session) -> List[Tuple[int, bytes]]:
            declare, prepare = list(), dict()
            for i, k in enumerate(keys):
                declare += [f"DECLARE ${self._KEY_VAR}_{i} AS Uint32;"]
                prepare.update({f"${self._KEY_VAR}_{i}": k})
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${NameConfig._id_column} AS Utf8;
                {" ".join(declare)}
                SELECT {NameConfig._key_column}, {field_name}
                FROM {self.turns_table}
                WHERE {NameConfig._id_column} = ${NameConfig._id_column} AND {field_name} IS NOT NULL
                AND {NameConfig._key_column} IN ({", ".join(prepare.keys())});
                """  # noqa: E501
            result_sets = await session.transaction().execute(
                await session.prepare(query),
                {
                    f"${NameConfig._id_column}": ctx_id,
                    **prepare,
                },
                commit_tx=True,
            )
            return (
                [(e[NameConfig._key_column], e[field_name]) for e in result_sets[0].rows]
                if len(result_sets[0].rows) > 0
                else list()
            )

        return await self.pool.retry_operation(callee)

    async def _clear_all(self) -> None:
        def construct_callee(table_name: str) -> Callable[[Session], Awaitable[None]]:
            async def callee(session: Session) -> None:
                query = f"""
                    PRAGMA TablePathPrefix("{self.database}");
                    DELETE FROM {table_name};
                    """  # noqa: E501
                await session.transaction().execute(await session.prepare(query), dict(), commit_tx=True)

            return callee

        await gather(
            self.pool.retry_operation(construct_callee(self.main_table)),
            self.pool.retry_operation(construct_callee(self.turns_table)),
        )
