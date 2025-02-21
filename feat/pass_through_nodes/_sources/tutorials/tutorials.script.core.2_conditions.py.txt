# %% [markdown]
"""
# Core: 2. Conditions

This tutorial shows different options for
setting transition conditions from one node to another.

Here, [conditions](%doclink(api,conditions.standard))
for script transitions are shown.
"""

# %pip install chatsky=={chatsky}

# %%
import re

from chatsky import (
    Context,
    TRANSITIONS,
    RESPONSE,
    Message,
    Pipeline,
    BaseCondition,
    Transition as Tr,
    conditions as cnd,
)

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)

# %% [markdown]
"""
The transition condition is determined by
%mddoclink(api,core.script_function,BaseCondition).

If this function returns `True`,
then the corresponding transition is considered possible.

Condition functions have signature

    class MyCondition(BaseCondition):
        async def call(self, ctx: Context) -> bool:

This script covers the following pre-defined conditions:

- `ExactMatch` returns `True` if the user's request completely
    matches the value passed to the function.
- `Regexp` returns `True` if the pattern matches the user's request.
    `Regexp` has same signature as `re.compile` function.
- `Any` returns `True` if one element of input sequence of conditions is `True`.
- `All` returns `True` if All elements of input
    sequence of conditions are `True`.

For a full list of available conditions see
[here](%doclink(api,conditions.standard)).

The `cnd` field of `Transition` may also be a constant bool value.
"""


# %%
class HiLowerCase(BaseCondition):
    """
    Return True if `hi` in both uppercase and lowercase
    letters is contained in the user request.
    """

    async def call(self, ctx: Context) -> bool:
        request = ctx.last_request
        return "hi" in request.text.lower()


# %% [markdown]
"""
Conditions are subclasses of `pydantic.BaseModel`.

You can define custom fields to make them more customizable:
"""


# %%
class ComplexUserAnswer(BaseCondition):
    """
    Checks if the misc field of the last message is of a certain value.

    Messages are more complex than just strings.
    The misc field can be used to store metadata about the message.
    More on that in the next tutorial.
    """

    value: dict

    async def call(self, ctx: Context) -> bool:
        request = ctx.last_request
        return request.misc == self.value


customized_condition = ComplexUserAnswer(value={"some_key": "some_value"})


# %%
toy_script = {
    "greeting_flow": {
        "start_node": {
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
            # If "Hi" == request of user then we make the transition
        },
        "node1": {
            RESPONSE: "Hi, how are you?",
            TRANSITIONS: [
                Tr(
                    dst="node2",
                    cnd=cnd.Regexp(r".*how are you", flags=re.IGNORECASE),
                )
            ],
            # pattern matching
        },
        "node2": {
            RESPONSE: "Good. What do you want to talk about?",
            TRANSITIONS: [
                Tr(
                    dst="node3",
                    cnd=cnd.All(
                        cnd.Regexp(r"talk"), cnd.Regexp(r"about.*music")
                    ),
                )
            ],
            # Combine sequences of conditions with `cnd.All`
        },
        "node3": {
            RESPONSE: "Sorry, I can not talk about music now.",
            TRANSITIONS: [
                Tr(dst="node4", cnd=cnd.Regexp(re.compile(r"Ok, goodbye.")))
            ],
        },
        "node4": {
            RESPONSE: "bye",
            TRANSITIONS: [
                Tr(
                    dst="node1",
                    cnd=cnd.Any(
                        HiLowerCase(),
                        cnd.ExactMatch("hello"),
                    ),
                )
            ],
            # Combine sequences of conditions with `cnd.Any`
        },
        "fallback_node": {  # We get to this node
            # if no suitable transition was found
            RESPONSE: "Ooops",
            TRANSITIONS: [
                Tr(dst="node1", cnd=customized_condition),
                # use a previously instantiated condition here
                Tr(dst="start_node", cnd=False),
                # This transition will never be made
                Tr(dst="fallback_node"),
                # `True` is the default value of `cnd`
                # this transition will always be valid
            ],
        },
    }
}

# testing
happy_path = (
    (
        "Hi",
        "Hi, how are you?",
    ),  # start_node -> node1
    (
        "i'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),  # node1 -> node2
    (
        "Let's talk about music.",
        "Sorry, I can not talk about music now.",
    ),  # node2 -> node3
    ("Ok, goodbye.", "bye"),  # node3 -> node4
    ("Hi", "Hi, how are you?"),  # node4 -> node1
    ("stop", "Ooops"),  # node1 -> fallback_node
    (
        "one",
        "Ooops",
    ),  # fallback_node -> fallback_node
    (
        "help",
        "Ooops",
    ),  # fallback_node -> fallback_node
    (
        "nope",
        "Ooops",
    ),  # fallback_node -> fallback_node
    (
        Message(misc={"some_key": "some_value"}),
        "Hi, how are you?",
    ),  # fallback_node -> node1
    (
        "i'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),  # node1 -> node2
    (
        "Let's talk about music.",
        "Sorry, I can not talk about music now.",
    ),  # node2 -> node3
    ("Ok, goodbye.", "bye"),  # node3 -> node4
)

# %%
pipeline = Pipeline(
    script=toy_script,
    start_label=("greeting_flow", "start_node"),
    fallback_label=("greeting_flow", "fallback_node"),
)

if __name__ == "__main__":
    check_happy_path(pipeline, happy_path, printout=True)
    if is_interactive_mode():
        pipeline.run()
