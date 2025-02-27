# %% [markdown]
"""
# 3. HF Tutorial

The following tutorial shows how to use HuggingFace models
for annotating user phrases.
"""

# %pip install chatsky[huggingface]

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from chatsky.script import (
    Message,
    RESPONSE,
    PRE_TRANSITIONS_PROCESSING,
    GLOBAL,
    TRANSITIONS,
    LOCAL,
)
from chatsky.script import conditions as cnd

from chatsky.ml.models import HFClassifier
from chatsky.ml.models import HFMatcher
from chatsky.ml.dataset import Dataset
from chatsky.ml import conditions as i_cnd
from chatsky.pipeline import Pipeline
from chatsky.messengers.console import CLIMessengerInterface
from chatsky.utils.testing.common import (
    is_interactive_mode,
    check_happy_path,
    run_interactive_mode,
)
from torch import device


# %% [markdown]
"""
Thanks to the structure of the HuggingFace library, the `AutoTokenizer` and `AutoModel` classes
will handle downloading and deploying the model for you. You need to instantiante
these classes separately and then pass them to the `Classifier` / `Matcher` instance.

We are using an open source model by Obsei-AI
to demonstrate, how custom classifiers can be easily adapted for use in chatsky.ml
However, you can use any classification model that is accessible via the Hugging Face hub
Below, we list some of the most popular open-source models that can power your conversational logic

unitary/toxic-bert - toxic speech detection
ProsusAI/finbert - sentiment analysis of financial texts
twitter-roberta-base-irony - irony detection
"""


# %%
tokenizer = AutoTokenizer.from_pretrained(
    "obsei-ai/sell-buy-intent-classifier-bert-mini"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "obsei-ai/sell-buy-intent-classifier-bert-mini"
)

common_label_collection = Dataset.model_validate(
    {
        "items": [
            {
                "label": "hello",
                "samples": ["hello", "hi", "hi there", "hello there"],
            },
            {"label": "goodbye", "samples": ["bye", "see you", "goodbye"]},
            {
                "label": "food",
                "samples": ["something to eat", "have a snack", "have a meal"],
            },
        ]
    }
)
# You can also parse static files using the Dataset structure:
# from pathlib import Path
# common_label_collection = Dataset.parse_json(Path(__file__).parent.joinpath("data/example.json"))

model_1 = HFClassifier(
    namespace_key="hf_classifier",
    tokenizer=tokenizer,
    device=device("cpu"),
    model=model,
)

model_2 = HFMatcher(
    namespace_key="hf_matcher",
    dataset=common_label_collection,
    device=device("cpu"),
    tokenizer=tokenizer,
    model=model,
)


# %%
script = {
    GLOBAL: {
        PRE_TRANSITIONS_PROCESSING: {
            "get_intents_1": model_1,
            "get_intents_2": model_2,
        },
        TRANSITIONS: {
            ("service", "buy", 1.2): i_cnd.has_cls_label(
                "LABEL_1", threshold=0.95
            ),
            ("service", "sell", 1.2): i_cnd.has_cls_label(
                "LABEL_0", threshold=0.95
            ),
        },
    },
    "root": {
        LOCAL: {TRANSITIONS: {("service", "offer", 1.2): cnd.true()}},
        "start": {RESPONSE: Message(text="Hi!")},
        "fallback": {
            RESPONSE: Message(text="I can't quite get what you mean.")
        },
        "finish": {
            RESPONSE: Message(text="Ok, see you soon!"),
            TRANSITIONS: {("service", "offer", 1.3): cnd.true()},
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
            ),
            TRANSITIONS: {("root", "finish", 1.3): cnd.true()},
        },
        "sell": {
            RESPONSE: Message(text="Your advertisement has been registered."),
            TRANSITIONS: {("root", "finish", 1.3): cnd.true()},
        },
    },
}


# %%
pipeline = Pipeline.from_script(
    script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
    messenger_interface=CLIMessengerInterface(intro="Starting chatsky bot..."),
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
    (Message(text="ok"), Message(text="Ok, see you soon!")),
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
    (Message(text="goodbye"), Message(text="Ok, see you soon!")),
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
