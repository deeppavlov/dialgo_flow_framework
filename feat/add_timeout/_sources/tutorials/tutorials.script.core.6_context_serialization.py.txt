# %% [markdown]
"""
# Core: 6. Context serialization
"""

# %pip install chatsky=={chatsky}

# %%
import logging

from chatsky import (
    TRANSITIONS,
    RESPONSE,
    Context,
    Pipeline,
    Transition as Tr,
    BaseResponse,
    MessageInitTypes,
)

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)


# %%
class RequestCounter(BaseResponse):
    async def call(self, ctx: Context) -> MessageInitTypes:
        return f"answer {len(ctx.requests)}"


# %%
toy_script = {
    "flow_start": {
        "node_start": {
            RESPONSE: RequestCounter(),
            TRANSITIONS: [Tr(dst=("flow_start", "node_start"))],
        }
    }
}

# testing
happy_path = (
    ("hi", "answer 1"),
    ("how are you?", "answer 2"),
    ("ok", "answer 3"),
    ("good", "answer 4"),
)

# %% [markdown]
"""
Draft function that performs serialization.
"""


# %%
def process_response(ctx: Context):
    ctx_json = ctx.model_dump_json()
    if isinstance(ctx_json, str):
        logging.info("context serialized to json str")
    else:
        raise Exception(f"ctx={ctx_json} has to be serialized to json string")

    ctx_dict = ctx.model_dump()
    if isinstance(ctx_dict, dict):
        logging.info("context serialized to dict")
    else:
        raise Exception(f"ctx={ctx_dict} has to be serialized to dict")

    if not isinstance(ctx, Context):
        raise Exception(f"ctx={ctx} has to have Context type")


# %%
pipeline = Pipeline(
    script=toy_script,
    start_label=("flow_start", "node_start"),
    post_services=[process_response],
)

if __name__ == "__main__":
    check_happy_path(pipeline, happy_path, printout=True)
    if is_interactive_mode():
        pipeline.run()
