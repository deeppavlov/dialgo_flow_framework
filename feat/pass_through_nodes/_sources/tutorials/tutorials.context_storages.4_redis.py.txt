# %% [markdown]
"""
# 4. Redis

This is a tutorial on using Redis.

See %mddoclink(api,context_storages.redis,RedisContextStorage) class
for storing you users' contexts in Redis database.

Chatsky uses [redis.asyncio](https://redis.readthedocs.io/en/latest/)
library for asynchronous access to Redis DB.
"""

# %pip install chatsky[redis]=={chatsky}

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
db_uri = "redis://{}:{}@localhost:6379/{}".format(
    "", os.environ["REDIS_PASSWORD"], "0"
)
db = context_storage_factory(db_uri)


pipeline = Pipeline(**TOY_SCRIPT_KWARGS, context_storage=db)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    if is_interactive_mode():
        pipeline.run()
