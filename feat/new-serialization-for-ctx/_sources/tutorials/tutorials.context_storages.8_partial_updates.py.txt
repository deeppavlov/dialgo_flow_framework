# %% [markdown]
"""
# 8. Partial context updates

The following tutorial shows the advanced usage
of context storage and context storage schema.
"""

# %pip install chatsky=={chatsky}

# %%
from pathlib import Path

from chatsky import Pipeline
from chatsky.context_storages import context_storage_factory
from chatsky.context_storages.database import NameConfig
from chatsky.utils.testing.common import check_happy_path, is_interactive_mode
from chatsky.utils.testing.toy_script import TOY_SCRIPT_KWARGS, HAPPY_PATH

# %%
Path("dbs").mkdir(exist_ok=True)
db = context_storage_factory("shelve://dbs/partly.shlv")

pipeline = Pipeline(**TOY_SCRIPT_KWARGS, context_storage=db)

# %% [markdown]
"""
Most of the `Context` fields, that might grow in size uncontrollably,
are stored in a special structure, `ContextDict`.
This structure can be used for fine-grained access to the underlying
database, partial and asynchronous element loading.
In particular, this is relevant for `labels`, `requests` and `responses`
fields, while `misc` and `framework_data` are always loaded fully.

How does that partial field writing work?

In most cases, every context storage
operates two "tables", "dictionaries", "files", etc.

One of them is called `MAIN` and contains all the "primitive" `Context`
data (and also the data that will be read and written completely every time) -
that includes context `id`, `current_turn_id`, `_created_at`,
`_updated_at`, `misc` and `framework_data` fields.

The other one is called `TURNS` and contains triplets of the data generated on
each conversation step: `label`, `request` and `response`.

Whenever a context is loaded, all of its information from `MAIN` table
and one to few items from `TURNS` table are loaded.
More items from `TURNS` table can be loaded later on demand
(via the `get` or `__getitem__` methods of corresponding fields).
"""

# %% [markdown]
"""
Database table layout and default behavior are controlled by
some special fields of the `DBContextStorage` class.

All the table and field names are stored in a special `NameConfig`
static class.
"""

# %% [markdown]
"""
One of the important configuration options is `_subscripts`:
this property controls the number of *last* dictionary items
that will be read and written
(the items are ordered by keys, ascending) - default value is 3.
In order to read *all* items at once, the property
can also be set to "__all__" literal.
In order to read only a specific subset of keys, the property
can be set to a set of desired integers.
"""

# %%
# All items will be loaded on every turn.
db._subscripts[NameConfig._requests_field] = "__all__"

# %%
# 5 last items will be loaded on every turn.
db._subscripts[NameConfig._requests_field] = 5

# %%
# Items 1, 3, 5 and 7 will be loaded on every turn.
db._subscripts[NameConfig._requests_field] = {1, 3, 5, 7}

# %% [markdown]
"""
Last but not least, comes `rewrite_existing` boolean flag.

Without it any "silent" modifications to the values of `ContextDict` will be
discarded at the end of each turn.

I.e. explicit modification of values via methods such as `__setitem__`,
`__delitem__` or `pop` will be kept track of and preserved, while
implicit modification via object manipulation,
e.g. ``ctx.last_request.text = "new_text"``, will be discarded.

Turning the option on will enable calculating hashes for all items stored
locally and comparing them at the end of every turn, updating any
that were implicitly changed.

NB! Keeping track of the modified elements comes with a price of calculating
their hashes and comparing them, so in performance-critical environments
this feature can be disabled by setting the flag to False.
"""

# %%
# Any modifications done to the elements already present in storage
# will be preserved.
db.rewrite_existing = True

# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH)
    # This is a function for automatic
    # tutorial running (testing) with HAPPY_PATH

    # This runs tutorial in interactive mode if not in IPython env
    # and if `DISABLE_INTERACTIVE_MODE` is not set
    if is_interactive_mode():
        pipeline.run()  # This runs tutorial in interactive mode
