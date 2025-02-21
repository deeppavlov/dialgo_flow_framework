# %% [markdown]
"""
# 1. Basic Example

The following tutorial shows basic usage of slots extraction
module packaged with `chatsky`.
"""

# %pip install chatsky=={chatsky}

# %%
from chatsky import (
    RESPONSE,
    TRANSITIONS,
    PRE_TRANSITION,
    PRE_RESPONSE,
    GLOBAL,
    LOCAL,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    processing as proc,
    responses as rsp,
)

from chatsky.slots import RegexpSlot

from chatsky.utils.testing import (
    check_happy_path,
    is_interactive_mode,
)

# %% [markdown]
"""
The slots fall into the following category groups:

- Value slots can be used to extract slot values from user utterances.
- Group slots can be used to split value slots into groups
    with an arbitrary level of nesting.

You can build the slot tree by passing the child slot instances as extra fields
of the parent slot. In the following cell, we define two slot groups:

    Group 1: person.username, person.email
    Group 2: friend.first_name, friend.last_name

Currently there are two types of value slots:

- %mddoclink(api,slots.slots,RegexpSlot):
    Extracts slot values via regexp.
- %mddoclink(api,slots.slots,FunctionSlot):
    Extracts slot values with the help of a user-defined function.
"""

# %%
SLOTS = {
    "person": {
        "username": RegexpSlot(
            regexp=r"username is ([a-zA-Z]+)",
            match_group_idx=1,
        ),
        "email": RegexpSlot(
            regexp=r"email is ([a-z@\.A-Z]+)",
            match_group_idx=1,
        ),
    },
    "friend": {
        "first_name": RegexpSlot(regexp=r"^[A-Z][a-z]+?(?= )"),
        "last_name": RegexpSlot(regexp=r"(?<= )[A-Z][a-z]+"),
    },
}

# %% [markdown]
"""
The slots module provides several functions for managing slots in-script:

- %mddoclink(api,conditions.slots,SlotsExtracted):
    Condition for checking if specified slots are extracted.
- %mddoclink(api,processing.slots,Extract):
    A processing function that extracts specified slots.
- %mddoclink(api,processing.slots,Unset):
    A processing function that marks specified slots as not extracted,
    effectively resetting their state.
- %mddoclink(api,processing.slots,UnsetAll):
    A processing function that marks all slots as not extracted.
- %mddoclink(api,processing.slots,FillTemplate):
    A processing function that fills the `response`
    Message text with extracted slot values.
- %mddoclink(api,responses.slots,FilledTemplate):
    A response function that takes a Message with a
    format-string text and returns Message
    with its text string filled with extracted slot values.

The usage of all the above functions is shown in the following script:
"""

# %%
script = {
    GLOBAL: {
        TRANSITIONS: [
            Tr(dst=("username_flow", "ask"), cnd=cnd.Regexp(r"^[sS]tart"))
        ]
    },
    "username_flow": {
        LOCAL: {
            PRE_TRANSITION: {"get_slot": proc.Extract("person.username")},
            TRANSITIONS: [
                Tr(
                    dst=("email_flow", "ask"),
                    cnd=cnd.SlotsExtracted("person.username"),
                    priority=1.2,
                ),
                Tr(dst=("username_flow", "repeat_question"), priority=0.8),
            ],
        },
        "ask": {
            RESPONSE: "Write your username (my username is ...):",
        },
        "repeat_question": {
            RESPONSE: "Please, type your username again (my username is ...):",
        },
    },
    "email_flow": {
        LOCAL: {
            PRE_TRANSITION: {"get_slot": proc.Extract("person.email")},
            TRANSITIONS: [
                Tr(
                    dst=("friend_flow", "ask"),
                    cnd=cnd.SlotsExtracted("person.username", "person.email"),
                    priority=1.2,
                ),
                Tr(dst=("email_flow", "repeat_question"), priority=0.8),
            ],
        },
        "ask": {
            RESPONSE: "Write your email (my email is ...):",
        },
        "repeat_question": {
            RESPONSE: "Please, write your email again (my email is ...):",
        },
    },
    "friend_flow": {
        LOCAL: {
            PRE_TRANSITION: {"get_slots": proc.Extract("friend")},
            TRANSITIONS: [
                Tr(
                    dst=("root", "utter"),
                    cnd=cnd.SlotsExtracted(
                        "friend.first_name", "friend.last_name", mode="any"
                    ),
                    priority=1.2,
                ),
                Tr(dst=("friend_flow", "repeat_question"), priority=0.8),
            ],
        },
        "ask": {RESPONSE: "Please, name me one of your friends: (John Doe)"},
        "repeat_question": {
            RESPONSE: "Please, name me one of your friends again: (John Doe)"
        },
    },
    "root": {
        "start": {
            TRANSITIONS: [Tr(dst=("username_flow", "ask"))],
        },
        "fallback": {
            RESPONSE: "Finishing query",
            TRANSITIONS: [Tr(dst=("username_flow", "ask"))],
        },
        "utter": {
            RESPONSE: rsp.FilledTemplate(
                "Your friend is {friend.first_name} {friend.last_name}"
            ),
            TRANSITIONS: [Tr(dst=("root", "utter_alternative"))],
        },
        "utter_alternative": {
            RESPONSE: "Your username is {person.username}. "
            "Your email is {person.email}.",
            PRE_RESPONSE: {"fill": proc.FillTemplate()},
        },
    },
}

# %%
HAPPY_PATH = [
    ("hi", "Write your username (my username is ...):"),
    ("my username is groot", "Write your email (my email is ...):"),
    (
        "my email is groot@gmail.com",
        "Please, name me one of your friends: (John Doe)",
    ),
    ("Bob Page", "Your friend is Bob Page"),
    ("ok", "Your username is groot. Your email is groot@gmail.com."),
    ("ok", "Finishing query"),
]

# %%
pipeline = Pipeline(
    script=script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
    slots=SLOTS,
)

if __name__ == "__main__":
    check_happy_path(
        pipeline, HAPPY_PATH, printout=True
    )  # This is a function for automatic tutorial running
    # (testing) with HAPPY_PATH

    if is_interactive_mode():
        pipeline.run()
