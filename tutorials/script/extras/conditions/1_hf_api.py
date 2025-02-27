# %% [markdown]
"""
# 1. HF API

This module explains, how to integrate web-hosted huggingface models in your conversational services.
"""

# %pip install dff[ext,async]

# %%
import os
from chatsky import (
    TRANSITIONS,
    RESPONSE,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    GLOBAL,
    LOCAL,
    Message,
    # all the aliases used in tutorials are available for direct import
    # e.g. you can do `from chatsky import Tr` instead
)

from chatsky.ml.models.hf_api_model import HFAPIModel
from chatsky.conditions.ml import HasLabel
from chatsky import Pipeline
from chatsky.messengers.console import CLIMessengerInterface
# from chatsky.utils.testing.common import (
#     is_interactive_mode,
#     check_happy_path,
#     run_interactive_mode,
# )


# %% [markdown]
"""
The HuggingFace inference API allows you to use any model
on HuggingFace hub that was made publicly available by its owners.
Pass the model address and an API key to construct the class.

We are using this open source model by Obsei-AI
to demonstrate, how custom classifiers can be easily adapted for use your script.
"""


# %%
api_model = HFAPIModel(
    model="obsei-ai/sell-buy-intent-classifier-bert-mini",
    api_key=os.getenv("HF_API_KEY") or input("Enter HF API key:"),
)


# %%
script = {
    GLOBAL: {
        TRANSITIONS: [
            # We get to one of the dialog branches depending on the annotation
            Tr(
                dst=("service", "buy"), priority=1.2, cnd=HasLabel(
                label="LABEL_1", model_name="my_hf_model", threshold=0.95)
            ),
            Tr(
                dst=("service", "sell"), priority=1.2, cnd=HasLabel(
                label="LABEL_0", model_name="my_hf_model", threshold=0.95)
            )
        ]
    },
    "root": {
        LOCAL: {TRANSITIONS: [Tr(dst=("service", "offer"), priority=1.2, cnd=True)]},
        "start": {RESPONSE: Message(text="Hi!")},
        "fallback": {
            RESPONSE: Message(text="I can't quite get what you mean.")
        },
        "finish": {
            RESPONSE: Message(text="Ok, see you soon!"),
            TRANSITIONS: [Tr(dst=("root", "start"), priority=1.3, cnd = True)],
        },
    },
    "service": {
        "offer": {
            RESPONSE: Message(
                text="Welcome to the e-marketplace. Tell us, what you would like to buy or sell."
            )
        },
        "buy": {
            RESPONSE: Message(
                text="Unfortunately, the item is out of stock at the moment."
            )
        },
        "sell": {
            RESPONSE: Message(text="Your advertisement has been registered.")
        },
    },
}

# %%
pipeline = Pipeline(
    script=script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
    messenger_interface=CLIMessengerInterface(intro="Starting Dff bot..."),
    models={"my_hf_model": api_model}
)


# %%
happy_path = [
    (
        Message(text="hi"),
        Message(
            text="Welcome to the e-marketplace. Tell us, what you would like to buy or sell."
        ),
    ),
    (
        Message(text="I would like to buy a car"),
        Message(text="Unfortunately, the item is out of stock at the moment."),
    ),
    (Message(text="ok"), Message(text="I can't quite get what you mean.")),
    (
        Message(text="ok"),
        Message(
            text="Welcome to the e-marketplace. Tell us, what you would like to buy or sell."
        ),
    ),
    (
        Message(text="sell a bike"),
        Message(text="Your advertisement has been registered."),
    ),
    (Message(text="goodbye"), Message(text="I can't quite get what you mean.")),
]


# %%
if __name__ == "__main__":
    # check_happy_path(
    #     pipeline,
    #     happy_path,
    # )  # This is a function for automatic tutorial
    # # running (testing tutorial) with `happy_path`.

    # # Run tutorial in interactive mode if not in IPython env
    # # and if `DISABLE_INTERACTIVE_MODE` is not set.
    pipeline.run()
        # This runs tutorial in interactive mode.
