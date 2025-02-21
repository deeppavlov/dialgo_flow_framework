# %% [markdown]
"""
# 6. SQLite

This is a tutorial on using SQLite.

See %mddoclink(api,context_storages.sql,SQLContextStorage) class
for storing you users' contexts in SQL databases.

Chatsky uses [sqlalchemy](https://docs.sqlalchemy.org/en/20/)
and [aiosqlite](https://readthedocs.org/projects/aiosqlite/)
libraries for asynchronous access to SQLite DB.

Note that protocol separator for windows differs from one for linux.
"""

# %pip install chatsky[sqlite]=={chatsky}

# %%
import pathlib
from platform import system

from chatsky.context_storages import context_storage_factory

from chatsky import Pipeline
from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)
from chatsky.utils.testing.toy_script import TOY_SCRIPT_KWARGS, HAPPY_PATH


# %%
pathlib.Path("dbs").mkdir(exist_ok=True)
db_file = pathlib.Path("dbs/sqlite.db")
db_file.touch(exist_ok=True)

separator = "///" if system() == "Windows" else "////"
db_uri = f"sqlite+aiosqlite:{separator}{db_file.absolute()}"
db = context_storage_factory(db_uri)


pipeline = Pipeline(**TOY_SCRIPT_KWARGS, context_storage=db)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    if is_interactive_mode():
        pipeline.run()
