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
from typing import Awaitable, Callable, Set, Tuple, List, Optional
from urllib.parse import urlsplit

from .database import DBContextStorage, _SUBSCRIPT_DICT
from .protocol import get_protocol_install_suggestion

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


class YDBContextStorage(DBContextStorage):
    """
    Version of the :py:class:`.DBContextStorage` for YDB.

    CONTEXT table is represented by `contexts` table.
    Columns of the table are: active_ctx, id, storage_key, data, created_at and updated_at.

    LOGS table is represented by `logs` table.
    Columns of the table are: id, field, key, value and updated_at.

    :param path: Standard sqlalchemy URI string. One of `grpc` or `grpcs` can be chosen as a protocol.
        Example: `grpc://localhost:2134/local`.
        NB! Do not forget to provide credentials in environmental variables
        or set `YDB_ANONYMOUS_CREDENTIALS` variable to `1`!
    :param context_schema: Context schema for this storage.
    :param serializer: Serializer that will be used for serializing contexts.
    :param table_name_prefix: "namespace" prefix for the two tables created for context storing.
    :param table_name: The name of the table to use.
    """

    _LIMIT_VAR = "limit"
    _KEY_VAR = "key"

    is_concurrent: bool = True

    def __init__(
        self,
        path: str,
        rewrite_existing: bool = False,
        configuration: Optional[_SUBSCRIPT_DICT] = None,
        table_name_prefix: str = "chatsky_table",
        timeout: int = 5,
    ):
        DBContextStorage.__init__(self, path, rewrite_existing, configuration)

        protocol, netloc, self.database, _, _ = urlsplit(path)
        if not ydb_available:
            install_suggestion = get_protocol_install_suggestion("grpc")
            raise ImportError("`ydb` package is missing.\n" + install_suggestion)

        self.table_prefix = table_name_prefix
        self._timeout = timeout
        self._endpoint = f"{protocol}://{netloc}"

    async def connect(self):
        await super().connect()
        await self._init_drive(self._timeout, self._endpoint)

    async def _init_drive(self, timeout: int, endpoint: str) -> None:
        self._driver = Driver(endpoint=endpoint, database=self.database)
        client_settings = self._driver.table_client._table_client_settings.with_allow_truncated_result(True)
        self._driver.table_client._table_client_settings = client_settings
        await self._driver.wait(fail_fast=True, timeout=timeout)

        self.pool = SessionPool(self._driver, size=10)

        self.main_table = f"{self.table_prefix}_{self._main_table_name}"
        if not await self._does_table_exist(self.main_table):
            await self._create_main_table(self.main_table)

        self.turns_table = f"{self.table_prefix}_{self._turns_table_name}"
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
                .with_column(Column(self._id_column_name, PrimitiveType.Utf8))
                .with_column(Column(self._current_turn_id_column_name, PrimitiveType.Uint64))
                .with_column(Column(self._created_at_column_name, PrimitiveType.Uint64))
                .with_column(Column(self._updated_at_column_name, PrimitiveType.Uint64))
                .with_column(Column(self._misc_column_name, PrimitiveType.String))
                .with_column(Column(self._framework_data_column_name, PrimitiveType.String))
                .with_primary_key(self._id_column_name),
            )

        await self.pool.retry_operation(callee)

    async def _create_turns_table(self, table_name: str) -> None:
        async def callee(session: Session) -> None:
            await session.create_table(
                "/".join([self.database, table_name]),
                TableDescription()
                .with_column(Column(self._id_column_name, PrimitiveType.Utf8))
                .with_column(Column(self._key_column_name, PrimitiveType.Uint32))
                .with_column(Column(self._labels_field_name, OptionalType(PrimitiveType.String)))
                .with_column(Column(self._requests_field_name, OptionalType(PrimitiveType.String)))
                .with_column(Column(self._responses_field_name, OptionalType(PrimitiveType.String)))
                .with_primary_keys(self._id_column_name, self._key_column_name),
            )

        await self.pool.retry_operation(callee)

    async def _load_main_info(self, ctx_id: str) -> Optional[Tuple[int, int, int, bytes, bytes]]:
        async def callee(session: Session) -> Optional[Tuple[int, int, int, bytes, bytes]]:
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${self._id_column_name} AS Utf8;
                SELECT {self._current_turn_id_column_name}, {self._created_at_column_name}, {self._updated_at_column_name}, {self._misc_column_name}, {self._framework_data_column_name}
                FROM {self.main_table}
                WHERE {self._id_column_name} = ${self._id_column_name};
                """  # noqa: E501
            result_sets = await session.transaction(SerializableReadWrite()).execute(
                await session.prepare(query),
                {
                    f"${self._id_column_name}": ctx_id,
                },
                commit_tx=True,
            )
            return (
                (
                    result_sets[0].rows[0][self._current_turn_id_column_name],
                    result_sets[0].rows[0][self._created_at_column_name],
                    result_sets[0].rows[0][self._updated_at_column_name],
                    result_sets[0].rows[0][self._misc_column_name],
                    result_sets[0].rows[0][self._framework_data_column_name],
                )
                if len(result_sets[0].rows) > 0
                else None
            )

        return await self.pool.retry_operation(callee)

    async def _update_main_info(
        self, ctx_id: str, turn_id: int, crt_at: int, upd_at: int, misc: bytes, fw_data: bytes
    ) -> None:
        async def callee(session: Session) -> None:
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${self._id_column_name} AS Utf8;
                DECLARE ${self._current_turn_id_column_name} AS Uint64;
                DECLARE ${self._created_at_column_name} AS Uint64;
                DECLARE ${self._updated_at_column_name} AS Uint64;
                DECLARE ${self._misc_column_name} AS String;
                DECLARE ${self._framework_data_column_name} AS String;
                UPSERT INTO {self.main_table} ({self._id_column_name}, {self._current_turn_id_column_name}, {self._created_at_column_name}, {self._updated_at_column_name}, {self._misc_column_name}, {self._framework_data_column_name})
                VALUES (${self._id_column_name}, ${self._current_turn_id_column_name}, ${self._created_at_column_name}, ${self._updated_at_column_name}, ${self._misc_column_name}, ${self._framework_data_column_name});
                """  # noqa: E501
            await session.transaction(SerializableReadWrite()).execute(
                await session.prepare(query),
                {
                    f"${self._id_column_name}": ctx_id,
                    f"${self._current_turn_id_column_name}": turn_id,
                    f"${self._created_at_column_name}": crt_at,
                    f"${self._updated_at_column_name}": upd_at,
                    f"${self._misc_column_name}": misc,
                    f"${self._framework_data_column_name}": fw_data,
                },
                commit_tx=True,
            )

        await self.pool.retry_operation(callee)

    async def _delete_context(self, ctx_id: str) -> None:
        def construct_callee(table_name: str) -> Callable[[Session], Awaitable[None]]:
            async def callee(session: Session) -> None:
                query = f"""
                    PRAGMA TablePathPrefix("{self.database}");
                    DECLARE ${self._id_column_name} AS Utf8;
                    DELETE FROM {table_name}
                    WHERE {self._id_column_name} = ${self._id_column_name};
                    """  # noqa: E501
                await session.transaction(SerializableReadWrite()).execute(
                    await session.prepare(query),
                    {
                        f"${self._id_column_name}": ctx_id,
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
                    declare += [f"DECLARE ${self._KEY_VAR}_{i} AS Utf8;"]
                    prepare.update({f"${self._KEY_VAR}_{i}": k})
                    values += [f"${self._KEY_VAR}_{i}"]
                key = f"AND {self._KEY_VAR} IN ({', '.join(values)})"
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${self._id_column_name} AS Utf8;
                {" ".join(declare)}
                SELECT {self._key_column_name}, {field_name}
                FROM {self.turns_table}
                WHERE {self._id_column_name} = ${self._id_column_name} AND {field_name} IS NOT NULL {key}
                ORDER BY {self._key_column_name} DESC {limit};
                """  # noqa: E501
            result_sets = await session.transaction(SerializableReadWrite()).execute(
                await session.prepare(query),
                {
                    f"${self._id_column_name}": ctx_id,
                    **prepare,
                },
                commit_tx=True,
            )
            return (
                [(e[self._key_column_name], e[field_name]) for e in result_sets[0].rows]
                if len(result_sets[0].rows) > 0
                else list()
            )

        return await self.pool.retry_operation(callee)

    async def _load_field_keys(self, ctx_id: str, field_name: str) -> List[int]:
        async def callee(session: Session) -> List[int]:
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${self._id_column_name} AS Utf8;
                SELECT {self._key_column_name}
                FROM {self.turns_table}
                WHERE {self._id_column_name} = ${self._id_column_name} AND {field_name} IS NOT NULL;
                """  # noqa: E501
            result_sets = await session.transaction(SerializableReadWrite()).execute(
                await session.prepare(query),
                {
                    f"${self._id_column_name}": ctx_id,
                },
                commit_tx=True,
            )
            return [e[self._key_column_name] for e in result_sets[0].rows] if len(result_sets[0].rows) > 0 else list()

        return await self.pool.retry_operation(callee)

    async def _load_field_items(self, ctx_id: str, field_name: str, keys: List[int]) -> List[Tuple[int, bytes]]:
        async def callee(session: Session) -> List[Tuple[int, bytes]]:
            declare, prepare = list(), dict()
            for i, k in enumerate(keys):
                declare += [f"DECLARE ${self._KEY_VAR}_{i} AS Uint32;"]
                prepare.update({f"${self._KEY_VAR}_{i}": k})
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${self._id_column_name} AS Utf8;
                {" ".join(declare)}
                SELECT {self._key_column_name}, {field_name}
                FROM {self.turns_table}
                WHERE {self._id_column_name} = ${self._id_column_name} AND {field_name} IS NOT NULL
                AND {self._key_column_name} IN ({", ".join(prepare.keys())});
                """  # noqa: E501
            result_sets = await session.transaction(SerializableReadWrite()).execute(
                await session.prepare(query),
                {
                    f"${self._id_column_name}": ctx_id,
                    **prepare,
                },
                commit_tx=True,
            )
            return (
                [(e[self._key_column_name], e[field_name]) for e in result_sets[0].rows]
                if len(result_sets[0].rows) > 0
                else list()
            )

        return await self.pool.retry_operation(callee)

    async def _update_field_items(self, ctx_id: str, field_name: str, items: List[Tuple[int, Optional[bytes]]]) -> None:
        async def callee(session: Session) -> None:
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
                values += [f"(${self._id_column_name}, ${self._KEY_VAR}_{i}, {value_param})"]
            query = f"""
                PRAGMA TablePathPrefix("{self.database}");
                DECLARE ${self._id_column_name} AS Utf8;
                {" ".join(declare)}
                UPSERT INTO {self.turns_table} ({self._id_column_name}, {self._key_column_name}, {field_name})
                VALUES {", ".join(values)};
                """  # noqa: E501

            await session.transaction(SerializableReadWrite()).execute(
                await session.prepare(query),
                {
                    f"${self._id_column_name}": ctx_id,
                    **prepare,
                },
                commit_tx=True,
            )

        await self.pool.retry_operation(callee)

    async def _clear_all(self) -> None:
        def construct_callee(table_name: str) -> Callable[[Session], Awaitable[None]]:
            async def callee(session: Session) -> None:
                query = f"""
                    PRAGMA TablePathPrefix("{self.database}");
                    DELETE FROM {table_name};
                    """  # noqa: E501
                await session.transaction(SerializableReadWrite()).execute(
                    await session.prepare(query), dict(), commit_tx=True
                )

            return callee

        await gather(
            self.pool.retry_operation(construct_callee(self.main_table)),
            self.pool.retry_operation(construct_callee(self.turns_table)),
        )
