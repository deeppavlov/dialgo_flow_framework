# %% [markdown]
"""
# 3. MongoDB

This is a tutorial on using MongoDB.

See %mddoclink(api,context_storages.mongo,MongoContextStorage) class
for storing you users' contexts in Mongo database.

Chatsky uses [motor](https://motor.readthedocs.io/en/stable/)
library for asynchronous access to MongoDB.
"""

# %pip install chatsky[mongodb]=={chatsky}

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
db_uri = "mongodb://{}:{}@localhost:27017/{}".format(
    os.environ["MONGO_INITDB_ROOT_USERNAME"],
    os.environ["MONGO_INITDB_ROOT_PASSWORD"],
    os.environ["MONGO_INITDB_ROOT_USERNAME"],
)
db = context_storage_factory(db_uri)

pipeline = Pipeline(**TOY_SCRIPT_KWARGS, context_storage=db)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    if is_interactive_mode():
        pipeline.run()
