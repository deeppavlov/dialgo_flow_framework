# %% [markdown]
"""
# Responses: 2. Multi Message

This tutorial shows how to store several messages inside a single one.
This might be useful if you want Chatsky Pipeline to send `response` candidates
to the messenger interface instead of a final response.
"""

# %pip install chatsky=={chatsky}

# %%

from chatsky import (
    TRANSITIONS,
    RESPONSE,
    Message,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
)

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)

# %%
toy_script = {
    "greeting_flow": {
        "start_node": {
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
        },
        "node1": {
            RESPONSE: Message(
                misc={
                    "messages": [
                        Message(
                            text="Hi, what is up?", misc={"confidences": 0.85}
                        ),
                        Message(
                            text="Hello, how are you?",
                            misc={"confidences": 0.9},
                        ),
                    ]
                }
            ),
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
            RESPONSE: "Sorry, I can not talk about that now.",
            TRANSITIONS: [Tr(dst="node4", cnd=cnd.ExactMatch("Ok, goodbye."))],
        },
        "node4": {
            RESPONSE: "bye",
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
        },
        "fallback_node": {
            RESPONSE: "Ooops",
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
        },
    }
}

# testing
happy_path = (
    (
        "Hi",
        Message(
            misc={
                "messages": [
                    Message("Hi, what is up?", misc={"confidences": 0.85}),
                    Message("Hello, how are you?", misc={"confidences": 0.9}),
                ]
            }
        ),
    ),  # start_node -> node1
    (
        "I'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),  # node1 -> node2
    (
        "Let's talk about music.",
        "Sorry, I can not talk about that now.",
    ),  # node2 -> node3
    ("Ok, goodbye.", "bye"),  # node3 -> node4
    (
        "Hi",
        Message(
            misc={
                "messages": [
                    Message("Hi, what is up?", misc={"confidences": 0.85}),
                    Message("Hello, how are you?", misc={"confidences": 0.9}),
                ]
            }
        ),
    ),  # node4 -> node1
    (
        "stop",
        "Ooops",
    ),
    # node1 -> fallback_node
    (
        "one",
        "Ooops",
    ),  # f_n->f_n
    (
        "help",
        "Ooops",
    ),  # f_n->f_n
    (
        "nope",
        "Ooops",
    ),  # f_n->f_n
    (
        "Hi",
        Message(
            misc={
                "messages": [
                    Message("Hi, what is up?", misc={"confidences": 0.85}),
                    Message("Hello, how are you?", misc={"confidences": 0.9}),
                ]
            }
        ),
    ),  # fallback_node -> node1
    (
        "I'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),  # node1 -> node2
    (
        "Let's talk about music.",
        "Sorry, I can not talk about that now.",
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
