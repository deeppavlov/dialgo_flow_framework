# %% [markdown]
"""
# Core: 7. Pre-response processing

Here, %mddoclink(api,core.script,PRE_RESPONSE)
is demonstrated which can be used for
additional context processing before response handlers.
"""

# %pip install chatsky=={chatsky}

# %%
from chatsky import (
    GLOBAL,
    LOCAL,
    RESPONSE,
    TRANSITIONS,
    PRE_RESPONSE,
    Context,
    Message,
    MessageInitTypes,
    BaseResponse,
    Transition as Tr,
    Pipeline,
    destinations as dst,
    processing as proc,
)

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)


# %% [markdown]
"""
Processing functions have the same signature as
conditions, responses or destinations
except they don't return anything:

.. python:

    class MyProcessing(BaseProcessing):
        async def call(self, ctx: Context) -> None:
            ...


The main way for processing functions to interact with the script
is modifying `ctx.current_node`, which is used by pipeline
to store a copy of the current node in script.
Any of its attributes can be safely edited, and these changes will
only have an effect during the current turn of the current context.
"""


# %% [markdown]
"""
In this tutorial we'll subclass
%mddoclink(api,processing.standard,ModifyResponse)
processing function so that it would modify response
of the current node to include a prefix.
"""


# %%
class AddPrefix(proc.ModifyResponse):
    prefix: str

    def __init__(self, prefix: str):
        # basemodel does not allow positional arguments by default
        super().__init__(prefix=prefix)

    async def modified_response(
        self, original_response: BaseResponse, ctx: Context
    ) -> MessageInitTypes:
        result = await original_response(ctx)

        if result.text is not None:
            result.text = f"{self.prefix}: {result.text}"
        return result


# %% [markdown]
"""
<div class="alert alert-info">

Tip

You can use `ModifyResponse` to catch exceptions in response functions:

.. python:

    class ExceptionHandler(proc.ModifyResponse):
        async def modified_response(self, original_response, ctx):
            try:
                return await original_response(ctx)
            except Exception as exc:
                return str(exc)

</div>
"""


# %%
toy_script = {
    "root": {
        "start": {
            TRANSITIONS: [Tr(dst=("flow", "step_0"))],
        },
        "fallback": {RESPONSE: "the end"},
    },
    GLOBAL: {
        PRE_RESPONSE: {
            "proc_name_1": AddPrefix("l1_global"),
            "proc_name_2": AddPrefix("l2_global"),
        }
    },
    "flow": {
        LOCAL: {
            PRE_RESPONSE: {
                "proc_name_2": AddPrefix("l2_local"),
                "proc_name_3": AddPrefix("l3_local"),
            },
            TRANSITIONS: [Tr(dst=dst.Forward(loop=True))],
        },
        "step_0": {
            RESPONSE: "first",
        },
        "step_1": {
            PRE_RESPONSE: {"proc_name_1": AddPrefix("l1_step_1")},
            RESPONSE: "second",
        },
        "step_2": {
            PRE_RESPONSE: {"proc_name_2": AddPrefix("l2_step_2")},
            RESPONSE: "third",
        },
        "step_3": {
            PRE_RESPONSE: {"proc_name_3": AddPrefix("l3_step_3")},
            RESPONSE: "fourth",
        },
        "step_4": {
            PRE_RESPONSE: {"proc_name_4": AddPrefix("l4_step_4")},
            RESPONSE: "fifth",
        },
    },
}


# %% [markdown]
"""
The order of execution for processing functions is as follows:

1. All node-specific functions are executed in the order of definition;
2. All local functions are executed in the order of definition except those with
   keys matching to previously executed functions;
3. All global functions are executed in the order of definition
   except those with keys matching to previously executed functions.

That means that if both global and local nodes
define a processing function with key "processing_name",
only the one inside the local node will be executed.

This demonstrated in the happy path below
(the first prefix in the text is the last one to execute):
"""


# %%
# testing
happy_path = (
    (Message(), "l1_global: l3_local: l2_local: first"),
    (Message(), "l3_local: l2_local: l1_step_1: second"),
    (Message(), "l1_global: l3_local: l2_step_2: third"),
    (Message(), "l1_global: l2_local: l3_step_3: fourth"),
    (Message(), "l1_global: l3_local: l2_local: l4_step_4: fifth"),
    (Message(), "l1_global: l3_local: l2_local: first"),
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
