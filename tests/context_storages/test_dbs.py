import asyncio

import pytest
import socket
import os
from platform import system

from dff.context_storages import (
    get_protocol_install_suggestion,
    json_available,
    pickle_available,
    ShelveContextStorage,
    postgres_available,
    mysql_available,
    sqlite_available,
    redis_available,
    mongo_available,
    ydb_available,
    context_storage_factory,
)

from dff.utils.testing.cleanup_db import (
    delete_shelve,
    delete_json,
    delete_pickle,
    delete_mongo,
    delete_redis,
    delete_sql,
    delete_ydb,
)
from tests.context_storages.test_functions import run_all_functions

from tests.test_utils import get_path_from_tests_to_current_dir

dot_path_to_addon = get_path_from_tests_to_current_dir(__file__, separator=".")


def ping_localhost(port: int, timeout=60):
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", port))
    except OSError:
        return False
    else:
        s.close()
        return True


MONGO_ACTIVE = ping_localhost(27017)

REDIS_ACTIVE = ping_localhost(6379)

POSTGRES_ACTIVE = ping_localhost(5432)

MYSQL_ACTIVE = ping_localhost(3307)

YDB_ACTIVE = ping_localhost(2136)


@pytest.mark.parametrize(
    ["protocol", "expected"],
    [
        ("pickle", "Try to run `pip install dff[pickle]`"),
        ("postgresql", "Try to run `pip install dff[postgresql]`"),
        ("false", ""),
    ],
)
def test_protocol_suggestion(protocol, expected):
    result = get_protocol_install_suggestion(protocol)
    assert result == expected


def test_shelve(testing_file, testing_context, context_id):
    db = ShelveContextStorage(f"shelve://{testing_file}")
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_shelve(db))


def test_dict(testing_context, context_id):
    db = dict()
    run_all_functions(db, testing_context, context_id)


@pytest.mark.skipif(not json_available, reason="JSON dependencies missing")
def test_json(testing_file, testing_context, context_id):
    db = context_storage_factory(f"json://{testing_file}")
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_json(db))


@pytest.mark.skipif(not pickle_available, reason="Pickle dependencies missing")
def test_pickle(testing_file, testing_context, context_id):
    db = context_storage_factory(f"pickle://{testing_file}")
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_pickle(db))


@pytest.mark.skipif(not MONGO_ACTIVE, reason="Mongodb server is not running")
@pytest.mark.skipif(not mongo_available, reason="Mongodb dependencies missing")
def test_mongo(testing_context, context_id):
    if system() == "Windows":
        pytest.skip()

    db = context_storage_factory(
        "mongodb://{}:{}@localhost:27017/{}".format(
            os.getenv("MONGO_INITDB_ROOT_USERNAME"),
            os.getenv("MONGO_INITDB_ROOT_PASSWORD"),
            os.getenv("MONGO_INITDB_ROOT_USERNAME"),
        )
    )
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_mongo(db))


@pytest.mark.skipif(not REDIS_ACTIVE, reason="Redis server is not running")
@pytest.mark.skipif(not redis_available, reason="Redis dependencies missing")
def test_redis(testing_context, context_id):
    db = context_storage_factory("redis://{}:{}@localhost:6379/{}".format("", os.getenv("REDIS_PASSWORD"), "0"))
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_redis(db))


@pytest.mark.skipif(not POSTGRES_ACTIVE, reason="Postgres server is not running")
@pytest.mark.skipif(not postgres_available, reason="Postgres dependencies missing")
def test_postgres(testing_context, context_id):
    db = context_storage_factory(
        "postgresql+asyncpg://{}:{}@localhost:5432/{}".format(
            os.getenv("POSTGRES_USERNAME"),
            os.getenv("POSTGRES_PASSWORD"),
            os.getenv("POSTGRES_DB"),
        )
    )
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_sql(db))


@pytest.mark.skipif(not sqlite_available, reason="Sqlite dependencies missing")
def test_sqlite(testing_file, testing_context, context_id):
    separator = "///" if system() == "Windows" else "////"
    db = context_storage_factory(f"sqlite+aiosqlite:{separator}{testing_file}")
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_sql(db))


@pytest.mark.skipif(not MYSQL_ACTIVE, reason="Mysql server is not running")
@pytest.mark.skipif(not mysql_available, reason="Mysql dependencies missing")
def test_mysql(testing_context, context_id):
    db = context_storage_factory(
        "mysql+asyncmy://{}:{}@localhost:3307/{}".format(
            os.getenv("MYSQL_USERNAME"),
            os.getenv("MYSQL_PASSWORD"),
            os.getenv("MYSQL_DATABASE"),
        )
    )
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_sql(db))


@pytest.mark.skipif(not YDB_ACTIVE, reason="YQL server not running")
@pytest.mark.skipif(not ydb_available, reason="YDB dependencies missing")
def test_ydb(testing_context, context_id):
    db = context_storage_factory(
        "{}{}".format(
            os.getenv("YDB_ENDPOINT"),
            os.getenv("YDB_DATABASE"),
        ),
        table_name_prefix="test_dff_table",
    )
    run_all_functions(db, testing_context, context_id)
    asyncio.run(delete_ydb(db))
