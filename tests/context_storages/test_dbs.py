import os
from platform import system
from socket import AF_INET, SOCK_STREAM, socket
from typing import Awaitable, Callable, Optional
import asyncio
import random

import pytest

from chatsky.context_storages import (
    get_protocol_install_suggestion,
    context_storage_factory,
    json_available,
    pickle_available,
    postgres_available,
    mysql_available,
    sqlite_available,
    redis_available,
    mongo_available,
    ydb_available,
    DBContextStorage,
)
from chatsky.utils.testing.cleanup_db import (
    delete_file,
    delete_mongo,
    delete_redis,
    delete_sql,
    delete_ydb,
)
from chatsky import Pipeline
from chatsky.core.ctx_utils import ContextMainInfo
from chatsky.context_storages.database import _SUBSCRIPT_TYPE
from chatsky.utils.testing import TOY_SCRIPT_KWARGS, HAPPY_PATH, check_happy_path

from tests.test_utils import get_path_from_tests_to_current_dir

AddContextType = Callable[[DBContextStorage, str, ContextMainInfo], Awaitable[None]]

dot_path_to_addon = get_path_from_tests_to_current_dir(__file__, separator=".")


def ping_localhost(port: int, timeout: int = 60) -> bool:
    try:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(("localhost", port))
    except OSError:
        return False
    else:
        sock.close()
        return True


MONGO_ACTIVE = ping_localhost(27017)

REDIS_ACTIVE = ping_localhost(6379)

POSTGRES_ACTIVE = ping_localhost(5432)

MYSQL_ACTIVE = ping_localhost(3307)

YDB_ACTIVE = ping_localhost(2136)


@pytest.mark.parametrize(
    ["protocol", "expected"],
    [
        ("pickle", "Try to run `pip install chatsky[pickle]`"),
        ("postgresql", "Try to run `pip install chatsky[postgresql]`"),
        ("false", ""),
    ],
)
def test_protocol_suggestion(protocol: str, expected: str) -> None:
    result = get_protocol_install_suggestion(protocol)
    assert result == expected


@pytest.mark.parametrize(
    "db_kwargs,db_teardown",
    [
        pytest.param({"path": ""}, None, id="memory"),
        pytest.param({"path": "shelve://{__testing_file__}"}, delete_file, id="shelve"),
        pytest.param(
            {"path": "json://{__testing_file__}"},
            delete_file,
            id="json",
            marks=[pytest.mark.skipif(not json_available, reason="Asynchronous file (JSON) dependencies missing")],
        ),
        pytest.param(
            {"path": "pickle://{__testing_file__}"},
            delete_file,
            id="pickle",
            marks=[pytest.mark.skipif(not pickle_available, reason="Asynchronous file (pickle) dependencies missing")],
        ),
        pytest.param(
            {
                "path": "mongodb://{MONGO_INITDB_ROOT_USERNAME}:{MONGO_INITDB_ROOT_PASSWORD}@"
                "localhost:27017/{MONGO_INITDB_ROOT_USERNAME}"
            },
            delete_mongo,
            id="mongo",
            marks=[
                pytest.mark.docker,
                pytest.mark.skipif(not MONGO_ACTIVE, reason="Mongodb server is not running"),
                pytest.mark.skipif(not mongo_available, reason="Mongodb dependencies missing"),
            ],
        ),
        pytest.param(
            {"path": "redis://:{REDIS_PASSWORD}@localhost:6379/0"},
            delete_redis,
            id="redis",
            marks=[
                pytest.mark.docker,
                pytest.mark.skipif(not REDIS_ACTIVE, reason="Redis server is not running"),
                pytest.mark.skipif(not redis_available, reason="Redis dependencies missing"),
            ],
        ),
        pytest.param(
            {"path": "postgresql+asyncpg://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}"},
            delete_sql,
            id="postgres",
            marks=[
                pytest.mark.docker,
                pytest.mark.skipif(not POSTGRES_ACTIVE, reason="Postgres server is not running"),
                pytest.mark.skipif(not postgres_available, reason="Postgres dependencies missing"),
            ],
        ),
        pytest.param(
            {"path": "sqlite+aiosqlite:{__separator__}{__testing_file__}"},
            delete_sql,
            id="sqlite",
            marks=[pytest.mark.skipif(not sqlite_available, reason="Sqlite dependencies missing")],
        ),
        pytest.param(
            {"path": "mysql+asyncmy://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@localhost:3307/{MYSQL_DATABASE}"},
            delete_sql,
            id="mysql",
            marks=[
                pytest.mark.docker,
                pytest.mark.skipif(not MYSQL_ACTIVE, reason="Mysql server is not running"),
                pytest.mark.skipif(not mysql_available, reason="Mysql dependencies missing"),
            ],
        ),
        pytest.param(
            {"path": "{YDB_ENDPOINT}{YDB_DATABASE}"},
            delete_ydb,
            id="ydb",
            marks=[
                pytest.mark.docker,
                pytest.mark.skipif(not YDB_ACTIVE, reason="YQL server not running"),
                pytest.mark.skipif(not ydb_available, reason="YDB dependencies missing"),
            ],
        ),
    ],
)
class TestContextStorages:
    @pytest.fixture
    async def db(self, db_kwargs, db_teardown, tmpdir_factory):
        kwargs = {"__separator__": "///" if system() == "Windows" else "////", **os.environ}
        if "{__testing_file__}" in db_kwargs["path"]:
            kwargs["__testing_file__"] = str(tmpdir_factory.mktemp("data").join("file.db"))
        db_kwargs["path"] = db_kwargs["path"].format(**kwargs)
        context_storage = context_storage_factory(**db_kwargs)
        await context_storage.connect()

        yield context_storage

        if db_teardown is not None:
            await db_teardown(context_storage)

    @pytest.fixture
    async def ctx_info(self):
        yield ContextMainInfo(current_turn_id=1)

    @pytest.fixture
    async def add_context(self):
        async def add_context_internal(db: DBContextStorage, ctx_id: str, ctx_info: ContextMainInfo) -> None:
            await db.update_context(ctx_id, ctx_info, [("labels", [(0, b"0")], list())])

        yield add_context_internal

    @staticmethod
    def configure_context_storage(
        context_storage: DBContextStorage,
        rewrite_existing: Optional[bool] = None,
        labels_subscript: Optional[_SUBSCRIPT_TYPE] = None,
        requests_subscript: Optional[_SUBSCRIPT_TYPE] = None,
        responses_subscript: Optional[_SUBSCRIPT_TYPE] = None,
        all_subscript: Optional[_SUBSCRIPT_TYPE] = None,
    ) -> None:
        if rewrite_existing is not None:
            context_storage.rewrite_existing = rewrite_existing
        if all_subscript is not None:
            labels_subscript = requests_subscript = responses_subscript = all_subscript
        if labels_subscript is not None:
            context_storage._subscripts["labels"] = labels_subscript
        if requests_subscript is not None:
            context_storage._subscripts["requests"] = requests_subscript
        if responses_subscript is not None:
            context_storage._subscripts["responses"] = responses_subscript

    async def test_add_context(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        # test the fixture
        await add_context(db, "1", ctx_info)

    async def test_get_main_info(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        await add_context(db, "1", ctx_info)
        assert await db.load_main_info("1") == ctx_info
        assert await db.load_main_info("2") is None

    async def test_update_main_info(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        await add_context(db, "1", ctx_info)
        await add_context(db, "2", ctx_info)
        assert await db.load_main_info("1") == ctx_info
        assert await db.load_main_info("2") == ctx_info

        new_info = ContextMainInfo(current_turn_id=2)
        await db.update_context("1", new_info)
        assert await db.load_main_info("1") == new_info
        assert await db.load_main_info("2") == ctx_info

    async def test_wrong_field_name(self, db: DBContextStorage):
        with pytest.raises(ValueError, match="Invalid value 'non-existent' for argument 'field_name'."):
            await db.load_field_latest("1", "non-existent")
        with pytest.raises(ValueError, match="Invalid value 'non-existent' for argument 'field_name'."):
            await db.load_field_keys("1", "non-existent")
        with pytest.raises(ValueError, match="Invalid value 'non-existent' for argument 'field_name'."):
            await db.load_field_items("1", "non-existent", [1, 2])
        with pytest.raises(ValueError, match="Invalid value 'non-existent' for argument 'field_name'."):
            await db.update_context("1", field_info=[("non-existent", [(1, b"2")], list())])

    async def test_field_get(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        await add_context(db, "1", ctx_info)

        assert await db.load_field_latest("1", "labels") == [(0, b"0")]
        assert set(await db.load_field_keys("1", "labels")) == {0}

        assert await db.load_field_latest("1", "requests") == []
        assert set(await db.load_field_keys("1", "requests")) == set()

    async def test_field_load(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        await add_context(db, "1", ctx_info)

        await db.update_context("1", field_info=[("requests", [(1, b"1"), (3, b"3"), (2, b"2"), (4, b"4")], list())])

        assert await db.load_field_items("1", "requests", [1, 2]) == [(1, b"1"), (2, b"2")]
        assert await db.load_field_items("1", "requests", [4, 3]) == [(3, b"3"), (4, b"4")]

    async def test_field_update(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        await add_context(db, "1", ctx_info)
        assert await db.load_field_latest("1", "labels") == [(0, b"0")]
        assert await db.load_field_latest("1", "requests") == []

        await db.update_context("1", field_info=[("labels", [(0, b"1")], list())])
        await db.update_context("1", field_info=[("requests", [(4, b"4")], list())])
        await db.update_context("1", field_info=[("labels", [(2, b"2")], list())])

        assert await db.load_field_latest("1", "labels") == [(2, b"2"), (0, b"1")]
        assert set(await db.load_field_keys("1", "labels")) == {0, 2}
        assert await db.load_field_latest("1", "requests") == [(4, b"4")]
        assert set(await db.load_field_keys("1", "requests")) == {4}

    async def test_int_key_field_subscript(
        self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType
    ):
        await add_context(db, "1", ctx_info)
        await db.update_context("1", field_info=[("requests", [(2, b"2")], list())])
        await db.update_context("1", field_info=[("requests", [(1, b"1")], list())])
        await db.update_context("1", field_info=[("requests", [(0, b"0")], list())])

        self.configure_context_storage(db, requests_subscript=2)
        assert await db.load_field_latest("1", "requests") == [(2, b"2"), (1, b"1")]

        self.configure_context_storage(db, requests_subscript="__all__")
        assert await db.load_field_latest("1", "requests") == [(2, b"2"), (1, b"1"), (0, b"0")]

        await db.update_context("1", field_info=[("requests", [(5, b"5")], list())])

        self.configure_context_storage(db, requests_subscript=2)
        assert await db.load_field_latest("1", "requests") == [(5, b"5"), (2, b"2")]

        self.configure_context_storage(db, requests_subscript={5, 1})
        assert await db.load_field_latest("1", "requests") == [(5, b"5"), (1, b"1")]

    async def test_delete_field_key(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        await add_context(db, "1", ctx_info)

        await db.update_context("1", field_info=[("labels", list(), [0])])

        assert await db.load_field_latest("1", "labels") == []

    async def test_raises_on_missing_field_keys(
        self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType
    ):
        await add_context(db, "1", ctx_info)

        assert set(await db.load_field_items("1", "labels", [0, 1])) == {(0, b"0")}
        assert set(await db.load_field_items("1", "requests", [0])) == set()

    async def test_delete_context(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        await add_context(db, "1", ctx_info)
        await add_context(db, "2", ctx_info)

        await db.delete_context("1")

        assert await db.load_main_info("1") is None
        assert await db.load_main_info("2") == ContextMainInfo(current_turn_id=1)

        assert set(await db.load_field_keys("1", "labels")) == set()
        assert set(await db.load_field_keys("2", "labels")) == {0}

    async def test_clear_all(self, db: DBContextStorage, ctx_info: ContextMainInfo, add_context: AddContextType):
        await add_context(db, "1", ctx_info)
        await add_context(db, "2", ctx_info)

        await db.clear_all()

        assert await db.load_main_info("1") is None
        assert await db.load_main_info("2") is None
        assert set(await db.load_field_keys("1", "labels")) == set()
        assert set(await db.load_field_keys("2", "labels")) == set()

    @pytest.mark.slow
    async def test_concurrent_operations(self, db: DBContextStorage):
        async def db_operations(key: int):
            str_key = str(key)
            key_misc = {f"{key}": key + 2}
            await asyncio.sleep(random.random() / 100)
            ctx_info = ContextMainInfo(current_turn_id=key, misc=key_misc)
            ctx_info._created_at = key + 1
            ctx_info._updated_at = key
            await db.update_context(str_key, ctx_info)
            await asyncio.sleep(random.random() / 100)
            assert await db.load_main_info(str_key) == ctx_info

            for idx in range(1, 20):
                requests_update = [(0, bytes(2 * key + idx)), (idx, bytes(key + idx))]
                await db.update_context(str_key, field_info=[("requests", requests_update, list())])
                await asyncio.sleep(random.random() / 100)
                keys = list(range(idx + 1))
                assert set(await db.load_field_keys(str_key, "requests")) == set(keys)
                assert set(await db.load_field_items(str_key, "requests", keys)) == {
                    (0, bytes(2 * key + idx)),
                    *[(k, bytes(key + k)) for k in range(1, idx + 1)],
                }

        operations = [db_operations(key * 2) for key in range(3)]
        await asyncio.gather(*operations)

    async def test_pipeline(self, db: DBContextStorage) -> None:
        # Test Pipeline workload on DB
        pipeline = Pipeline(**TOY_SCRIPT_KWARGS, context_storage=db)
        check_happy_path(pipeline, happy_path=HAPPY_PATH)
