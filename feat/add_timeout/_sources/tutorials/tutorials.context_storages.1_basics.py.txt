# %% [markdown]
"""
# 1. Basics

The following tutorial shows the basic use of the database connection.

See %mddoclink(api,context_storages.database,context_storage_factory) function
for creating a context storage by path.

In this example JSON file is used as a storage.
"""

# %pip install chatsky[json,pickle]=={chatsky}

# %%
import pathlib

from chatsky.context_storages import context_storage_factory

from chatsky import Pipeline
from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)
from chatsky.utils.testing.toy_script import TOY_SCRIPT_KWARGS, HAPPY_PATH

pathlib.Path("dbs").mkdir(exist_ok=True)
db = context_storage_factory("json://dbs/file.json")
# db = context_storage_factory("pickle://dbs/file.pkl")
# db = context_storage_factory("shelve://dbs/file.shlv")

pipeline = Pipeline(**TOY_SCRIPT_KWARGS, context_storage=db)

if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    # a function for automatic tutorial running (testing) with HAPPY_PATH

    if is_interactive_mode():
        pipeline.run()
