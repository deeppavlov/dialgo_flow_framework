"""
SQL
---
The SQL module provides a SQL-based version of the :py:class:`.DBContextStorage` class.
This class is used to store and retrieve context data from SQL databases.
It allows Chatsky to easily store and retrieve context data in a format that is highly scalable
and easy to work with.

The SQL module provides the ability to choose the backend of your choice from
MySQL, PostgreSQL, or SQLite. You can choose the one that is most suitable for your use case and environment.
MySQL and PostgreSQL are widely used open-source relational databases that are known for their
reliability and scalability. SQLite is a self-contained, high-reliability, embedded, full-featured,
public-domain, SQL database engine.
"""

from __future__ import annotations
from asyncio import gather
from importlib import import_module
from typing import Any, Callable, Collection, Dict, List, Optional, Set, Tuple
from logging import getLogger

try:
    from sqlalchemy import (
        Table,
        MetaData,
        Column,
        LargeBinary,
        String,
        BigInteger,
        ForeignKey,
        Integer,
        Index,
        Insert,
        inspect,
        select,
        delete,
        event,
    )
    from sqlalchemy.ext.asyncio import create_async_engine

    sqlalchemy_available = True
except (ImportError, ModuleNotFoundError):
    sqlalchemy_available = False

try:
    import asyncpg

    _ = asyncpg

    postgres_available = True
except (ImportError, ModuleNotFoundError):
    postgres_available = False

try:
    import asyncmy

    _ = asyncmy

    mysql_available = True
except (ImportError, ModuleNotFoundError):
    mysql_available = True

try:
    import aiosqlite

    _ = aiosqlite

    sqlite_available = True
except (ImportError, ModuleNotFoundError):
    sqlite_available = False

from .database import DBContextStorage, _SUBSCRIPT_DICT, NameConfig
from .protocol import get_protocol_install_suggestion

if not sqlalchemy_available:
    postgres_available = sqlite_available = mysql_available = False

logger = getLogger(__name__)


def _sqlite_enable_foreign_key(dbapi_con, con_record):
    dbapi_con.execute("pragma foreign_keys=ON")


def _import_insert_for_dialect(dialect: str) -> Callable[[Table], "Insert"]:
    return getattr(import_module(f"sqlalchemy.dialects.{dialect}"), "insert")


def _get_upsert_stmt(dialect: str, insert_stmt, columns: Collection[str], unique: Collection[str]):
    if dialect == "postgresql" or dialect == "sqlite":
        if len(columns) > 0:
            update_stmt = insert_stmt.on_conflict_do_update(
                index_elements=unique, set_={column: insert_stmt.excluded[column] for column in columns}
            )
        else:
            update_stmt = insert_stmt.on_conflict_do_nothing()
    elif dialect == "mysql":
        if len(columns) > 0:
            update_stmt = insert_stmt.on_duplicate_key_update(
                **{column: insert_stmt.inserted[column] for column in columns}
            )
        else:
            update_stmt = insert_stmt.prefix_with("IGNORE")
    else:
        update_stmt = insert_stmt
    return update_stmt


class SQLContextStorage(DBContextStorage):
    """
    SQL-based version of the :py:class:`.DBContextStorage`.
    Compatible with MySQL, Postgresql, Sqlite.
    When using Sqlite on a Windows system, keep in mind that you have to use double backslashes '\\'
    instead of forward slashes '/' in the file path.

    `MAIN` table is represented by `main` table.
    Columns of the table are: `id`, `current_turn_id`, `created_at` `updated_at`, `misc` and `framework_data`.

    `TURNS` table is represented by `turns` table.
    Columns of the table are: `id`, `key`, `label`, `request` and `response`.

    :param path: Standard sqlalchemy URI string.
        Examples: `sqlite+aiosqlite://path_to_the_file/file_name`,
        `mysql+asyncmy://root:pass@localhost:3306/test`,
        `postgresql+asyncpg://postgres:pass@localhost:5430/test`.
    :param rewrite_existing: Whether `TURNS` modified locally should be updated in database or not.
    :param partial_read_config: Dictionary of subscripts for all possible turn items.
    :param table_name_prefix: "namespace" prefix for the two tables created for context storing.
    :param database_id_length: Length of context ID column in the database.
    """

    def __init__(
        self,
        path: str,
        rewrite_existing: bool = False,
        partial_read_config: Optional[_SUBSCRIPT_DICT] = None,
        table_name_prefix: str = "chatsky_table",
        database_id_length: int = 255,
    ):
        DBContextStorage.__init__(self, path, rewrite_existing, partial_read_config)

        self._check_availability()
        self.engine = create_async_engine(self.full_path, pool_pre_ping=True)
        self.dialect: str = self.engine.dialect.name
        self._INSERT_CALLABLE = _import_insert_for_dialect(self.dialect)

        if self.dialect == "sqlite":
            event.listen(self.engine.sync_engine, "connect", _sqlite_enable_foreign_key)

        metadata = MetaData()
        self.main_table = Table(
            f"{table_name_prefix}_{NameConfig._main_table}",
            metadata,
            Column(NameConfig._id_column, String(database_id_length), index=True, unique=True, nullable=False),
            Column(NameConfig._current_turn_id_column, BigInteger(), nullable=False),
            Column(NameConfig._created_at_column, BigInteger(), nullable=False),
            Column(NameConfig._updated_at_column, BigInteger(), nullable=False),
            Column(NameConfig._misc_column, LargeBinary(), nullable=False),
            Column(NameConfig._framework_data_column, LargeBinary(), nullable=False),
        )
        self.turns_table = Table(
            f"{table_name_prefix}_{NameConfig._turns_table}",
            metadata,
            Column(
                NameConfig._id_column,
                String(database_id_length),
                ForeignKey(self.main_table.name, NameConfig._id_column),
                nullable=False,
            ),
            Column(NameConfig._key_column, Integer(), nullable=False),
            Column(NameConfig._labels_field, LargeBinary(), nullable=True, default=None),
            Column(NameConfig._requests_field, LargeBinary(), nullable=True, default=None),
            Column(NameConfig._responses_field, LargeBinary(), nullable=True, default=None),
            Index(f"{NameConfig._turns_table}_index", NameConfig._id_column, NameConfig._key_column, unique=True),
        )

    @property
    def is_concurrent(self) -> bool:
        return self.dialect != "sqlite"

    async def _connect(self):
        async with self.engine.begin() as conn:
            for table in [self.main_table, self.turns_table]:
                if not await conn.run_sync(lambda sync_conn: inspect(sync_conn).has_table(table.name)):
                    logger.debug(f"SQL table created: {table.name}")
                    await conn.run_sync(table.create, self.engine)
                else:
                    logger.debug(f"SQL table already exists: {table.name}")

    def _check_availability(self):
        """
        Chech availability of the specified backend, raise error if not available.

        :param custom_driver: custom driver is requested - no checks will be performed.
        """
        if self.full_path.startswith("postgresql") and not postgres_available:
            install_suggestion = get_protocol_install_suggestion("postgresql")
            raise ImportError("Packages `sqlalchemy` and/or `asyncpg` are missing.\n" + install_suggestion)
        elif self.full_path.startswith("mysql") and not mysql_available:
            install_suggestion = get_protocol_install_suggestion("mysql")
            raise ImportError("Packages `sqlalchemy` and/or `asyncmy` are missing.\n" + install_suggestion)
        elif self.full_path.startswith("sqlite") and not sqlite_available:
            install_suggestion = get_protocol_install_suggestion("sqlite")
            raise ImportError("Package `sqlalchemy` and/or `aiosqlite` is missing.\n" + install_suggestion)

    async def _load_main_info(self, ctx_id: str) -> Optional[Dict[str, Any]]:
        stmt = select(self.main_table).where(self.main_table.c[NameConfig._id_column] == ctx_id)
        async with self.engine.begin() as conn:
            result = (await conn.execute(stmt)).fetchone()
            return (
                None if result is None else {f: result[i + 1] for i, f in enumerate(NameConfig.get_context_main_fields)}
            )

    async def _update_context(
        self,
        ctx_id: str,
        ctx_info: Optional[Dict[str, Any]],
        field_info: List[Tuple[str, List[Tuple[int, Optional[bytes]]]]],
    ) -> None:
        main_update_stmt, turns_update_stmts = None, list()
        if ctx_info is not None:
            main_insert_stmt = self._INSERT_CALLABLE(self.main_table).values(
                {
                    NameConfig._id_column: ctx_id,
                }
                | {f: ctx_info[f] for f in NameConfig.get_context_main_fields}
            )
            main_update_stmt = _get_upsert_stmt(
                self.dialect,
                main_insert_stmt,
                [
                    NameConfig._updated_at_column,
                    NameConfig._current_turn_id_column,
                    NameConfig._misc_column,
                    NameConfig._framework_data_column,
                ],
                [NameConfig._id_column],
            )
        for field_name, items in field_info:
            turns_insert_values = [
                {
                    NameConfig._id_column: ctx_id,
                    NameConfig._key_column: k,
                    field_name: v,
                }
                for k, v in items
                if len(items) > 0
            ]
            if len(turns_insert_values) > 0:
                turns_insert_stmt = self._INSERT_CALLABLE(self.turns_table).values(turns_insert_values)
                turns_update_stmts += [
                    _get_upsert_stmt(
                        self.dialect,
                        turns_insert_stmt,
                        [field_name],
                        [NameConfig._id_column, NameConfig._key_column],
                    )
                ]
        async with self.engine.begin() as conn:
            if main_update_stmt is not None:
                await conn.execute(main_update_stmt)
            for turns_update_stmt in turns_update_stmts:
                await conn.execute(turns_update_stmt)

    # TODO: use foreign keys instead maybe?
    async def _delete_context(self, ctx_id: str) -> None:
        async with self.engine.begin() as conn:
            await gather(
                conn.execute(delete(self.main_table).where(self.main_table.c[NameConfig._id_column] == ctx_id)),
                conn.execute(delete(self.turns_table).where(self.turns_table.c[NameConfig._id_column] == ctx_id)),
            )

    async def _load_field_latest(self, ctx_id: str, field_name: str) -> List[Tuple[int, bytes]]:
        stmt = select(self.turns_table.c[NameConfig._key_column], self.turns_table.c[field_name])
        stmt = stmt.where(self.turns_table.c[NameConfig._id_column] == ctx_id)
        stmt = stmt.where(self.turns_table.c[field_name] != None)  # noqa: E711
        stmt = stmt.order_by(self.turns_table.c[NameConfig._key_column].desc())
        if isinstance(self._subscripts[field_name], int):
            stmt = stmt.limit(self._subscripts[field_name])
        elif isinstance(self._subscripts[field_name], Set):
            stmt = stmt.where(self.turns_table.c[NameConfig._key_column].in_(self._subscripts[field_name]))
        async with self.engine.begin() as conn:
            return list((await conn.execute(stmt)).fetchall())

    async def _load_field_keys(self, ctx_id: str, field_name: str) -> List[int]:
        stmt = select(self.turns_table.c[NameConfig._key_column])
        stmt = stmt.where(self.turns_table.c[NameConfig._id_column] == ctx_id)
        stmt = stmt.where(self.turns_table.c[field_name] != None)  # noqa: E711
        async with self.engine.begin() as conn:
            return [k[0] for k in (await conn.execute(stmt)).fetchall()]

    async def _load_field_items(self, ctx_id: str, field_name: str, keys: List[int]) -> List[Tuple[int, bytes]]:
        stmt = select(self.turns_table.c[NameConfig._key_column], self.turns_table.c[field_name])
        stmt = stmt.where(self.turns_table.c[NameConfig._id_column] == ctx_id)
        stmt = stmt.where(self.turns_table.c[NameConfig._key_column].in_(tuple(keys)))
        stmt = stmt.where(self.turns_table.c[field_name] != None)  # noqa: E711
        async with self.engine.begin() as conn:
            return list((await conn.execute(stmt)).fetchall())

    async def _clear_all(self) -> None:
        async with self.engine.begin() as conn:
            await gather(conn.execute(delete(self.main_table)), conn.execute(delete(self.turns_table)))
