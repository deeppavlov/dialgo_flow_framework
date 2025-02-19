# %% [markdown]
"""
# Core: 8. Misc

This tutorial shows `MISC` (miscellaneous) keyword usage.

See %mddoclink(api,core.script,MISC)
for more information.
"""

# %pip install chatsky=={chatsky}

# %%
from chatsky import (
    GLOBAL,
    LOCAL,
    RESPONSE,
    TRANSITIONS,
    MISC,
    Context,
    Message,
    Pipeline,
    MessageInitTypes,
    BaseResponse,
    Transition as Tr,
    destinations as dst,
)

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)


# %% [markdown]
"""
`MISC` is used to store custom node data.
It can be accessed via `ctx.current_node.misc`.
"""


# %%
class CustomResponse(BaseResponse):
    async def call(self, ctx: Context) -> MessageInitTypes:
        return (
            f"node_name={ctx.last_label.node_name}: "
            f"current_node.misc={ctx.current_node.misc}"
        )


# %%
toy_script = {
    "root": {
        "start": {
            TRANSITIONS: [Tr(dst=("flow", "step_0"))],
        },
        "fallback": {RESPONSE: "the end"},
    },
    GLOBAL: {
        MISC: {
            "var1": "global_data",
            "var2": "global_data",
            "var3": "global_data",
        }
    },
    "flow": {
        LOCAL: {
            MISC: {
                "var2": "global data is overwritten by local",
                "var3": "global data is overwritten by local",
            },
            TRANSITIONS: [Tr(dst=dst.Forward(loop=True))],
        },
        "step_0": {
            MISC: {"var3": "this overwrites local values - step_0"},
            RESPONSE: CustomResponse(),
        },
        "step_1": {
            MISC: {"var3": "this overwrites local values - step_1"},
            RESPONSE: CustomResponse(),
        },
        "step_2": {
            MISC: {"var3": "this overwrites local values - step_2"},
            RESPONSE: CustomResponse(),
        },
    },
}


# testing
happy_path = (
    (
        Message(),
        "node_name=step_0: current_node.misc="
        "{'var3': 'this overwrites local values - step_0', "
        "'var2': 'global data is overwritten by local', "
        "'var1': 'global_data'}",
    ),
    (
        Message(),
        "node_name=step_1: current_node.misc="
        "{'var3': 'this overwrites local values - step_1', "
        "'var2': 'global data is overwritten by local', "
        "'var1': 'global_data'}",
    ),
    (
        Message(),
        "node_name=step_2: current_node.misc="
        "{'var3': 'this overwrites local values - step_2', "
        "'var2': 'global data is overwritten by local', "
        "'var1': 'global_data'}",
    ),
    (
        Message(),
        "node_name=step_0: current_node.misc="
        "{'var3': 'this overwrites local values - step_0', "
        "'var2': 'global data is overwritten by local', "
        "'var1': 'global_data'}",
    ),
)


# %%
pipeline = Pipeline(
    script=toy_script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
)

if __name__ == "__main__":
    check_happy_path(pipeline, happy_path, printout=True)
    if is_interactive_mode():
        pipeline.run()
