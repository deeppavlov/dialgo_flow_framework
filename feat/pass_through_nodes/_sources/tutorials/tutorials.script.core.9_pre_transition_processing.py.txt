# %% [markdown]
"""
# Core: 9. Pre-transition processing

Here, %mddoclink(api,core.script,PRE_TRANSITION)
is demonstrated which can be used for additional context
processing before transitioning to the next step.
"""

# %pip install chatsky=={chatsky}

# %%
from chatsky import (
    GLOBAL,
    RESPONSE,
    TRANSITIONS,
    PRE_RESPONSE,
    PRE_TRANSITION,
    Context,
    Pipeline,
    BaseProcessing,
    BaseResponse,
    MessageInitTypes,
    Transition as Tr,
    destinations as dst,
    processing as proc,
)

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)


# %% [markdown]
"""
Processing functions can be used at two stages:

1. Pre-transition. Triggers after response is received but before
   the next node is considered.
2. Pre-response. Triggers after transition is chosen and current node is
   changed but before response of that node is calculated.

In this tutorial we'll save the response function of the current node
during pre-transition and extract it during pre-response
(at which point current node is already changed).
"""


# %%
class SavePreviousNodeResponse(BaseProcessing):
    async def call(self, ctx: Context) -> None:
        if ctx.current_node.response is not None:
            ctx.misc["previous_node_response"] = ctx.current_node.response
        # This function is called as Pre-transition
        # so current node is going to be the previous one
        # when we reach the Pre-response step


class PrependPreviousNodeResponse(proc.ModifyResponse):
    async def modified_response(
        self, original_response: BaseResponse, ctx: Context
    ) -> MessageInitTypes:
        result = await original_response(ctx)

        previous_node_response = ctx.misc.get("previous_node_response")
        if previous_node_response is None:
            return result
        else:
            previous_result = await previous_node_response(ctx)
        return f"previous={previous_result.text}: current={result.text}"


# %% [markdown]
"""
<div class="alert alert-info">

Note

Previous node can be accessed another way.

Instead of storing the node response in misc,
one can obtain previous label
with `dst.Previous()(ctx)` and then get the node from the
%mddoclink(api,core.script,Script) object:

```python
ctx.pipeline.script.get_inherited_node(dst.Previous()(ctx))
```

</div>
"""


# %%
# a dialog script
toy_script = {
    "root": {
        "start": {
            TRANSITIONS: [Tr(dst=("flow", "step_0"))],
        },
        "fallback": {RESPONSE: "the end"},
    },
    GLOBAL: {
        PRE_RESPONSE: {"proc_name_1": PrependPreviousNodeResponse()},
        PRE_TRANSITION: {"proc_name_1": SavePreviousNodeResponse()},
        TRANSITIONS: [Tr(dst=dst.Forward(loop=True))],
    },
    "flow": {
        "step_0": {RESPONSE: "first"},
        "step_1": {RESPONSE: "second"},
        "step_2": {RESPONSE: "third"},
        "step_3": {RESPONSE: "fourth"},
        "step_4": {RESPONSE: "fifth"},
    },
}


# testing
happy_path = (
    ("1", "first"),
    ("2", "previous=first: current=second"),
    ("3", "previous=second: current=third"),
    ("4", "previous=third: current=fourth"),
    ("5", "previous=fourth: current=fifth"),
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
