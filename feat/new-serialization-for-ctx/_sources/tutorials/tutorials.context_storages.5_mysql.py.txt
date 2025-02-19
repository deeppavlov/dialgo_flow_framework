# %% [markdown]
"""
# 5. MySQL

This is a tutorial on using MySQL.

See %mddoclink(api,context_storages.sql,SQLContextStorage) class
for storing you users' contexts in SQL databases.

Chatsky uses [sqlalchemy](https://docs.sqlalchemy.org/en/20/)
and [asyncmy](https://github.com/long2ice/asyncmy)
libraries for asynchronous access to MySQL DB.
"""

# %pip install chatsky[mysql]=={chatsky}

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
db_uri = "mysql+asyncmy://{}:{}@localhost:3307/{}".format(
    os.environ["MYSQL_USERNAME"],
    os.environ["MYSQL_PASSWORD"],
    os.environ["MYSQL_DATABASE"],
)
db = context_storage_factory(db_uri)


pipeline = Pipeline(**TOY_SCRIPT_KWARGS, context_storage=db)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    if is_interactive_mode():
        pipeline.run()
