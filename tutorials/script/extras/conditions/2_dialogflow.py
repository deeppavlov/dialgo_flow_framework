# %% [markdown]
"""
# 2. Dialogflow

The tutorial below demonstrates, how to integrate Google Dialogflow into your script logic using `llm_conditions`.
The way of using the `GoogleDialogFlowModel` class is similar to that of other models.
Tutorials for other models can be found in the same section.
"""

# %pip install dff[ext,dialogflow]

# %%
import os

from chatsky import (
    Message,
    RESPONSE,
    GLOBAL,
    TRANSITIONS,
    LOCAL,
)
from chatsky import conditions as cnd

from chatsky.ml.models.remote_api.google_dialogflow_model import (
    GoogleDialogFlowModel,
)
from chatsky.ml import conditions as i_cnd
from chatsky import Pipeline
from chatsky.messengers.console import CLIMessengerInterface
from chatsky.utils.testing.common import (
    is_interactive_mode,
    check_happy_path,
    run_interactive_mode,
)


# %% [markdown]
"""
All you need to instantiate the `GoogleDialogFlowModel` class are
Google API credentials. You can obtain further instructions
on how to get those in the class documenation.

After you have received the credentials in the form of a json file,
you can use them to construct the class.
"""


# %%
gdf_model = GoogleDialogFlowModel.from_file(
    filename=os.getenv("GDF_ACCOUNT_JSON", "")
)

# %%
script = {
    GLOBAL: {
        # Intents from Google Dialogflow can be used in conditions to traverse your dialog graph
        TRANSITIONS: {
            ("root", "finish", 1.2): i_cnd.has_cls_label(gdf_model, "goodbye")
        },
    },
    "root": {
        LOCAL: {TRANSITIONS: {("mood", "ask", 1.2): cnd.true()}},
        "start": {RESPONSE: Message(text="Hi!")},
        "fallback": {
            RESPONSE: Message(text="I can't quite get what you mean.")
        },
        "finish": {RESPONSE: Message(text="Ok, see you soon!")},
    },
    "mood": {
        "ask": {
            RESPONSE: Message(text="How do you feel today?"),
            TRANSITIONS: {
                # Here, we use intents to decide which branch of dialog should be picked
                ("mood", "react_good"): i_cnd.has_cls_label(
                    gdf_model,
                    "mood_great",
                    threshold=0.95,
                    namespace="dialogflow",
                ),
                ("mood", "react_bad"): i_cnd.has_cls_label(
                    gdf_model, "mood_unhappy", namespace="dialogflow"
                ),
                ("mood", "assert"): cnd.true(),
            },
        },
        "assert": {
            RESPONSE: Message(
                text="What you mean is you're feeling down, isn't it?"
            ),
            TRANSITIONS: {
                ("mood", "react_good"): i_cnd.has_cls_label(
                    gdf_model, "deny", threshold=0.95, namespace="dialogflow"
                ),
                ("mood", "react_bad"): i_cnd.has_cls_label(
                    gdf_model, "affirm", namespace="dialogflow"
                ),
            },
        },
        "react_good": {
            RESPONSE: Message(
                text="Now that's the right talk! You'd better stay happy and stuff."
            ),
            TRANSITIONS: {("root", "finish"): cnd.true()},
        },
        "react_bad": {
            RESPONSE: Message(
                text="I feel you, fellow human. Watch a good movie, it might help."
            ),
            TRANSITIONS: {("root", "finish"): cnd.true()},
        },
    },
}


# %%
pipeline = Pipeline.from_script(
    script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
    messenger_interface=CLIMessengerInterface(intro="Starting Dff bot..."),
)


# %%
happy_path = [
    (Message(text="hi"), Message(text="How do you feel today?")),
    (
        Message(text="i'm sad"),
        Message(
            text="I feel you, fellow human. Watch a good movie, it might help."
        ),
    ),
    (Message(text="ok"), Message(text="Ok, see you soon!")),
    (Message(text="hi"), Message(text="How do you feel today?")),
    (
        Message(text="rather bad"),
        Message(text="What you mean is you're feeling down, isn't it?"),
    ),
    (
        Message(text="yes"),
        Message(
            text="I feel you, fellow human. Watch a good movie, it might help."
        ),
    ),
    (Message(text="good"), Message(text="Ok, see you soon!")),
    (Message(text="hi"), Message(text="How do you feel today?")),
    (
        Message(text="great"),
        Message(
            text="Now that's the right talk! You'd better stay happy and stuff."
        ),
    ),
]


# %%
if __name__ == "__main__":
    check_happy_path(
        pipeline,
        happy_path,
    )  # This is a function for automatic tutorial
    # running (testing tutorial) with `happy_path`.

    # Run tutorial in interactive mode if not in IPython env
    # and if `DISABLE_INTERACTIVE_MODE` is not set.
    if is_interactive_mode():
        run_interactive_mode(pipeline)
        # This runs tutorial in interactive mode.
