# %% [markdown]
"""
# Core: 4. Transitions

This tutorial shows settings for transitions between flows and nodes.

Here, [conditions](%doclink(api,conditions.standard))
for transition between many different script steps are shown.

Some of the destination steps can be set using
[destinations](%doclink(api,destinations.standard)).
"""


# %pip install chatsky=={chatsky}

# %%
import re

from chatsky import (
    TRANSITIONS,
    RESPONSE,
    Context,
    NodeLabelInitTypes,
    Pipeline,
    Transition as Tr,
    BaseDestination,
    conditions as cnd,
    destinations as dst,
)
from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)

# %% [markdown]
"""
The `TRANSITIONS` keyword is used to determine a list of transitions from
the current node. After receiving user request, Pipeline will choose the
next node relying on that list.
If no transition in the list is suitable, transition will be made
to the fallback node.

Each transition is represented by the %mddoclink(api,core.transition,Transition)
class.

It has three main fields:

- dst: Destination determines the node to which the transition is made.
- cnd: Condition determines if the transition is allowed.
- priority: Allows choosing one of the transitions if several are allowed.
    Higher priority transitions will be chosen over the rest.
    If priority is not set,
    %mddoclink(api,core.pipeline,Pipeline.default_priority)
    is used instead.
    Default priority is 1 by default (but may be set via Pipeline).

For more details on how the next node is chosen see
[here](%doclink(api,core.transition,get_next_label)).

Like conditions, all of these fields can be either constant values or
custom functions (%mddoclink(api,core.script_function,BaseDestination),
%mddoclink(api,core.script_function,BaseCondition),
%mddoclink(api,core.script_function,BasePriority)).
"""

# %% [markdown]
"""
## Destinations

Destination node is specified with a %mddoclink(api,core.node_label,NodeLabel)
class.

It contains two field:

- "flow_name": Name of the flow the node belongs to.
    Optional; if not set, will use the flow of the current node.
- "node_name": Name of the node inside the flow.

Instances of this class can be initialized from a tuple of two strings
(flow name and node name) or a single string (node name; relative flow name).
This happens automatically for return values of `BaseDestination`
and for the `dst` field of `Transition`.
"""


# %%
class GreetingFlowNode2(BaseDestination):
    async def call(self, ctx: Context) -> NodeLabelInitTypes:
        return "greeting_flow", "node2"


# %% [markdown]
"""
Chatsky provides several basic transitions as part of
the %mddoclink(api,destinations.standard) module:

- `FromHistory` returns a node from label history.
    `Current` and `Previous` are subclasses of it that return specific nodes
    (current node and previous node respectively).
- `Start` returns the start node.
- `Fallback` returns the fallback node.
- `Forward` returns the next node (in order of definition)
    in the current flow relative to the current node.
- `Backward` returns the previous node (in order of definition)
    in the current flow relative to the current node.
"""

# %%
toy_script = {
    "global_flow": {
        "start_node": {
            TRANSITIONS: [
                Tr(
                    dst=("music_flow", "node1"),
                    cnd=cnd.Regexp(r"talk about music"),
                    # this condition is checked first.
                    # if it fails, pipeline will try the next transition
                ),
                Tr(
                    dst=("greeting_flow", "node1"),
                    cnd=cnd.Regexp(r"hi|hello", flags=re.IGNORECASE),
                ),
                Tr(
                    dst="fallback_node",
                    # a single string references a node in the same flow
                ),
                # this transition will only be made if previous ones fail
            ]
        },
        "fallback_node": {
            RESPONSE: "Ooops",
            TRANSITIONS: [
                Tr(
                    dst=("music_flow", "node1"),
                    cnd=cnd.Regexp(r"talk about music"),
                ),
                Tr(
                    dst=("greeting_flow", "node1"),
                    cnd=cnd.Regexp(r"hi|hello", flags=re.IGNORECASE),
                ),
                Tr(
                    dst=dst.Previous(),
                    cnd=cnd.Regexp(r"previous", flags=re.IGNORECASE),
                ),
                Tr(
                    dst=dst.Current(),  # this goes to the current node
                    # i.e. fallback node
                ),
            ],
        },
    },
    "greeting_flow": {
        "node1": {
            RESPONSE: "Hi, how are you?",
            TRANSITIONS: [
                Tr(
                    dst=("global_flow", "fallback_node"),
                    priority=0.1,
                ),  # due to low priority (default priority is 1)
                # this transition will be made if the next one fails
                Tr(dst="node2", cnd=cnd.Regexp(r"how are you")),
            ],
        },
        "node2": {
            RESPONSE: "Good. What do you want to talk about?",
            TRANSITIONS: [
                Tr(
                    dst=dst.Fallback(),
                    priority=0.1,
                ),
                # there is no need to specify such transition:
                # For any node if all transitions fail,
                # fallback node becomes the next node.
                # Here, this transition exists for demonstration purposes.
                Tr(
                    dst=dst.Forward(),  # i.e. "node3" of this flow
                    cnd=cnd.Regexp(r"talk about"),
                    priority=0.5,
                ),  # this transition is the third candidate
                Tr(
                    dst=("music_flow", "node1"),
                    cnd=cnd.Regexp(r"talk about music"),
                ),  # this transition is the first candidate
                Tr(
                    dst=dst.Previous(),
                    cnd=cnd.Regexp(r"previous", flags=re.IGNORECASE),
                ),  # this transition is the second candidate
            ],
        },
        "node3": {
            RESPONSE: "Sorry, I can not talk about that now.",
            TRANSITIONS: [Tr(dst=dst.Forward(), cnd=cnd.Regexp(r"bye"))],
        },
        "node4": {
            RESPONSE: "Bye",
            TRANSITIONS: [
                Tr(
                    dst="node1",
                    cnd=cnd.Regexp(r"hi|hello", flags=re.IGNORECASE),
                )
            ],
        },
    },
    "music_flow": {
        "node1": {
            RESPONSE: "I love `System of a Down` group, "
            "would you like to talk about it?",
            TRANSITIONS: [
                Tr(
                    dst=dst.Forward(),
                    cnd=cnd.Regexp(r"yes|yep|ok", flags=re.IGNORECASE),
                )
            ],
        },
        "node2": {
            RESPONSE: "System of a Down is an Armenian-American "
            "heavy metal band formed in 1994.",
            TRANSITIONS: [
                Tr(
                    dst=dst.Forward(),
                    cnd=cnd.Regexp(r"next", flags=re.IGNORECASE),
                ),
                Tr(
                    dst=dst.Current(),
                    cnd=cnd.Regexp(r"repeat", flags=re.IGNORECASE),
                ),
            ],
        },
        "node3": {
            RESPONSE: "The band achieved commercial success "
            "with the release of five studio albums.",
            TRANSITIONS: [
                Tr(
                    dst=dst.Forward(),
                    cnd=cnd.Regexp(r"next", flags=re.IGNORECASE),
                ),
                Tr(
                    dst=dst.Backward(),
                    cnd=cnd.Regexp(r"back", flags=re.IGNORECASE),
                ),
                Tr(
                    dst=dst.Current(),
                    cnd=cnd.Regexp(r"repeat", flags=re.IGNORECASE),
                ),
            ],
        },
        "node4": {
            RESPONSE: "That's all I know.",
            TRANSITIONS: [
                Tr(
                    dst=GreetingFlowNode2(),
                    cnd=cnd.Regexp(r"next", flags=re.IGNORECASE),
                ),
                Tr(
                    dst=("greeting_flow", "node4"),
                    cnd=cnd.Regexp(r"next time", flags=re.IGNORECASE),
                    priority=2,
                ),  # "next" is contained in "next_time" so we need higher
                # priority here.
                # Otherwise, this transition will never be made
            ],
        },
    },
}

# testing
happy_path = (
    ("hi", "Hi, how are you?"),
    (
        "i'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),
    (
        "talk about music.",
        "I love `System of a Down` group, " "would you like to talk about it?",
    ),
    (
        "yes",
        "System of a Down is "
        "an Armenian-American heavy metal band formed in 1994.",
    ),
    (
        "next",
        "The band achieved commercial success "
        "with the release of five studio albums.",
    ),
    (
        "back",
        "System of a Down is "
        "an Armenian-American heavy metal band formed in 1994.",
    ),
    (
        "repeat",
        "System of a Down is "
        "an Armenian-American heavy metal band formed in 1994.",
    ),
    (
        "next",
        "The band achieved commercial success "
        "with the release of five studio albums.",
    ),
    ("next", "That's all I know."),
    (
        "next",
        "Good. What do you want to talk about?",
    ),
    ("previous", "That's all I know."),
    ("next time", "Bye"),
    ("stop", "Ooops"),
    ("previous", "Bye"),
    ("stop", "Ooops"),
    ("nope", "Ooops"),
    ("hi", "Hi, how are you?"),
    ("stop", "Ooops"),
    ("previous", "Hi, how are you?"),
    (
        "i'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),
    (
        "let's talk about something.",
        "Sorry, I can not talk about that now.",
    ),
    ("Ok, goodbye.", "Bye"),
)

# %%
pipeline = Pipeline(
    script=toy_script,
    start_label=("global_flow", "start_node"),
    fallback_label=("global_flow", "fallback_node"),
)

if __name__ == "__main__":
    check_happy_path(pipeline, happy_path, printout=True)
    if is_interactive_mode():
        pipeline.run()
