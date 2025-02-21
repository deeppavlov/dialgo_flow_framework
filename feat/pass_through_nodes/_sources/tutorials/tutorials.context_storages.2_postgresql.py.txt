# %% [markdown]
"""
# 2. PostgreSQL

This is a tutorial on using PostgreSQL.

See %mddoclink(api,context_storages.sql,SQLContextStorage) class
for storing your users' contexts in SQL databases.

Chatsky uses [sqlalchemy](https://docs.sqlalchemy.org/en/20/)
and [asyncpg](https://magicstack.github.io/asyncpg/current/)
libraries for asynchronous access to PostgreSQL DB.
"""

# %pip install chatsky[postgresql]=={chatsky}

# %%
import os

from chatsky.context_storages import context_storage_factory

from chatsky import Pipeline
from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)
from chatsky.utils.testing.toy_script import TOY_SCRIPT_KWARGS, HAPPY_PATH


# %%
db_uri = "postgresql+asyncpg://{}:{}@localhost:5432/{}".format(
    os.environ["POSTGRES_USERNAME"],
    os.environ["POSTGRES_PASSWORD"],
    os.environ["POSTGRES_DB"],
)
db = context_storage_factory(db_uri)


pipeline = Pipeline(**TOY_SCRIPT_KWARGS, context_storage=db)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    if is_interactive_mode():
        pipeline.run()
