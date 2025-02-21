# %% [markdown]
"""
# Core: 3. Responses

This tutorial shows different options for setting responses.

Here, [responses](%doclink(api,responses.standard))
that allow giving custom answers to users are shown.
"""

# %pip install chatsky=={chatsky}

# %%
import re
import random
from typing import Union

from chatsky import (
    TRANSITIONS,
    RESPONSE,
    Context,
    Message,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    responses as rsp,
    destinations as dst,
    BaseResponse,
    MessageInitTypes,
    AnyResponse,
    AbsoluteNodeLabel,
)

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)


# %% [markdown]
"""
Response of a node is determined by
%mddoclink(api,core.script_function,BaseResponse).

Response can be constant in which case it is an instance
of %mddoclink(api,core.message,Message).

`Message` has an option to be instantiated from a string
which is what we've been using so far.
Under the hood `RESPONSE: "text"` is converted into
`RESPONSE: Message(text="text")`.
This class should be used over simple strings when
some additional information needs to be sent such as images/metadata.

More information on that can be found in the [media tutorial](
%doclink(tutorial,script.responses.1_media)
).

Instances of this class are returned by
%mddoclink(api,core.context,Context.last_request) and
%mddoclink(api,core.context,Context.last_response).
In the previous tutorial we showed how to access fields of messages
to build custom conditions.

Node `RESPONSE` can also be set to a custom function.
This is demonstrated below:
"""


# %%
class CannotTalkAboutTopic(BaseResponse):
    async def call(self, ctx: Context) -> MessageInitTypes:
        request = ctx.last_request
        if request.text is None:
            topic = None
        else:
            topic_pattern = re.compile(r"(.*talk about )(.*)\.")
            topic = topic_pattern.findall(request.text)
            topic = topic and topic[0] and topic[0][-1]
        if topic:
            return f"Sorry, I can not talk about {topic} now."
        else:
            return "Sorry, I can not talk about that now."


class UpperCase(BaseResponse):
    response: AnyResponse  # either const response or another BaseResponse

    def __init__(self, response: Union[MessageInitTypes, BaseResponse]):
        # defining this allows passing response as a positional argument
        # and allows to make a more detailed type annotation:
        # AnyResponse cannot be a string but can be initialized from it,
        # so MessageInitTypes annotates that we can init from a string
        super().__init__(response=response)

    async def call(self, ctx: Context) -> MessageInitTypes:
        response = await self.response(ctx)
        # const response is converted to BaseResponse,
        # so we call it regardless of the response type

        if response.text is not None:
            response.text = response.text.upper()
        return response


class FallbackTrace(BaseResponse):
    async def call(self, ctx: Context) -> MessageInitTypes:
        return Message(
            misc={
                "previous_node": await dst.Previous()(ctx),
                "last_request": ctx.last_request,
            }
        )


# %% [markdown]
"""
Chatsky provides one basic response as part of
the %mddoclink(api,responses.standard) module:

- `RandomChoice` randomly chooses a message out of the ones passed to it.
"""


# %%
toy_script = {
    "greeting_flow": {
        "start_node": {
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
        },
        "node1": {
            RESPONSE: rsp.RandomChoice(
                "Hi, what is up?",
                "Hello, how are you?",
            ),
            # Random choice from candidate list.
            TRANSITIONS: [
                Tr(dst="node2", cnd=cnd.ExactMatch("I'm fine, how are you?"))
            ],
        },
        "node2": {
            RESPONSE: "Good. What do you want to talk about?",
            TRANSITIONS: [
                Tr(dst="node3", cnd=cnd.ExactMatch("Let's talk about music."))
            ],
        },
        "node3": {
            RESPONSE: CannotTalkAboutTopic(),
            TRANSITIONS: [Tr(dst="node4", cnd=cnd.ExactMatch("Ok, goodbye."))],
        },
        "node4": {
            RESPONSE: UpperCase("bye"),
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
        },
        "fallback_node": {
            RESPONSE: FallbackTrace(),
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
        },
    }
}

# testing
happy_path = (
    (
        "Hi",
        "Hello, how are you?",
    ),  # start_node -> node1
    (
        "I'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),  # node1 -> node2
    (
        "Let's talk about music.",
        "Sorry, I can not talk about music now.",
    ),  # node2 -> node3
    ("Ok, goodbye.", "BYE"),  # node3 -> node4
    ("Hi", "Hello, how are you?"),  # node4 -> node1
    (
        "stop",
        Message(
            misc={
                "previous_node": AbsoluteNodeLabel(
                    flow_name="greeting_flow", node_name="node1"
                ),
                "last_request": Message("stop"),
            }
        ),
    ),
    # node1 -> fallback_node
    (
        "one",
        Message(
            misc={
                "previous_node": AbsoluteNodeLabel(
                    flow_name="greeting_flow", node_name="fallback_node"
                ),
                "last_request": Message("one"),
            }
        ),
    ),  # f_n->f_n
    (
        "help",
        Message(
            misc={
                "previous_node": AbsoluteNodeLabel(
                    flow_name="greeting_flow", node_name="fallback_node"
                ),
                "last_request": Message("help"),
            }
        ),
    ),  # f_n->f_n
    (
        "nope",
        Message(
            misc={
                "previous_node": AbsoluteNodeLabel(
                    flow_name="greeting_flow", node_name="fallback_node"
                ),
                "last_request": Message("nope"),
            }
        ),
    ),  # f_n->f_n
    (
        "Hi",
        "Hi, what is up?",
    ),  # fallback_node -> node1
    (
        "I'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),  # node1 -> node2
    (
        "Let's talk about music.",
        "Sorry, I can not talk about music now.",
    ),  # node2 -> node3
    ("Ok, goodbye.", "BYE"),  # node3 -> node4
)

# %%
random.seed(31415)  # predestination of choice


pipeline = Pipeline(
    script=toy_script,
    start_label=("greeting_flow", "start_node"),
    fallback_label=("greeting_flow", "fallback_node"),
)

if __name__ == "__main__":
    check_happy_path(pipeline, happy_path, printout=True)
    if is_interactive_mode():
        pipeline.run()
