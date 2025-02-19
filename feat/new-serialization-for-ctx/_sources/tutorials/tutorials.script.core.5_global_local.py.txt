# %% [markdown]
"""
# Core: 5. Global and Local nodes
"""


# %pip install chatsky=={chatsky}

# %%
import re

from chatsky import (
    GLOBAL,
    TRANSITIONS,
    RESPONSE,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    destinations as dst,
)
from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)

# %% [markdown]
"""
Keywords `GLOBAL` and `LOCAL` are used to define global and local nodes
respectively. Global node is defined at the script level (along with flows)
and local node is defined at the flow level (along with nodes inside a flow).

Every local node inherits properties from the global node.
Every node inherits properties from the local node (of its flow).

For example, if we are to set list `A` as transitions for the
local node of a flow, then every node of that flow would effectively
have the `A` list extended with its own transitions.

<div class="alert alert-info">

To sum up transition priorities:

Transition A is of higher priority compared to Transition B:

1. If A.priority > B.priority; OR
2. If A is a node transition and B is a local or global transition;
    or A is a local transition and B is a global transition; OR
3. If A is defined in the transition list earlier than B.

</div>

For more information on node inheritance, see [here](
%doclink(api,core.script,Script.get_inherited_node)
).

<div class="alert alert-info">

Note

Property %mddoclink(api,core.context,Context.current_node) does not return
the current node as is. Instead it returns a node that is modified
by the global and local nodes.

</div>
"""

# %%
toy_script = {
    GLOBAL: {
        TRANSITIONS: [
            Tr(
                dst=("greeting_flow", "node1"),
                cnd=cnd.Regexp(r"\b(hi|hello)\b", flags=re.I),
                priority=1.1,
            ),
            Tr(
                dst=("music_flow", "node1"),
                cnd=cnd.Regexp(r"talk about music"),
                priority=1.1,
            ),
            Tr(
                dst=dst.Forward(),
                cnd=cnd.All(
                    cnd.Regexp(r"next\b"),
                    cnd.CheckLastLabels(
                        labels=[("music_flow", i) for i in ["node2", "node3"]]
                    ),  # this checks if the current node is
                    # music_flow.node2 or music_flow.node3
                ),
            ),
            Tr(
                dst=dst.Current(),
                cnd=cnd.All(
                    cnd.Regexp(r"repeat", flags=re.I),
                    cnd.Negation(
                        cnd.CheckLastLabels(flow_labels=["global_flow"])
                    ),
                ),
                priority=0.2,
            ),
        ],
    },
    "global_flow": {
        "start_node": {},
        "fallback_node": {
            RESPONSE: "Ooops",
            TRANSITIONS: [
                Tr(
                    dst=dst.Previous(),
                    cnd=cnd.Regexp(r"previous", flags=re.I),
                )
            ],
        },
    },
    "greeting_flow": {
        "node1": {
            RESPONSE: "Hi, how are you?",
            TRANSITIONS: [Tr(dst="node2", cnd=cnd.Regexp(r"how are you"))],
        },
        "node2": {
            RESPONSE: "Good. What do you want to talk about?",
            TRANSITIONS: [
                Tr(
                    dst=dst.Forward(),
                    cnd=cnd.Regexp(r"talk about"),
                    priority=0.5,
                ),
                Tr(
                    dst=dst.Previous(),
                    cnd=cnd.Regexp(r"previous", flags=re.I),
                ),
            ],
        },
        "node3": {
            RESPONSE: "Sorry, I can not talk about that now.",
            TRANSITIONS: [Tr(dst=dst.Forward(), cnd=cnd.Regexp(r"bye"))],
        },
        "node4": {RESPONSE: "bye"},
        # This node does not define its own transitions.
        # It will use global transitions only.
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
        },
        "node3": {
            RESPONSE: "The band achieved commercial success "
            "with the release of five studio albums.",
            TRANSITIONS: [
                Tr(
                    dst=dst.Backward(),
                    cnd=cnd.Regexp(r"back", flags=re.IGNORECASE),
                ),
            ],
        },
        "node4": {
            RESPONSE: "That's all I know.",
            TRANSITIONS: [
                Tr(
                    dst=("greeting_flow", "node4"),
                    cnd=cnd.Regexp(r"next time", flags=re.I),
                ),
                Tr(
                    dst=("greeting_flow", "node2"),
                    cnd=cnd.Regexp(r"next", flags=re.I),
                ),
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
    ("next time", "bye"),
    ("stop", "Ooops"),
    ("previous", "bye"),
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
    ("Ok, goodbye.", "bye"),
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
