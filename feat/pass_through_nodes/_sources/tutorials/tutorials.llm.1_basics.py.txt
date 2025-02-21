# %% [markdown]
"""
# LLM: 1. Basics

With Chatsky, you can easily integrate LLM (Large Language Model)
invocations into your scripts.
This tutorial demonstrates how to use LLMs for generating responses and handling
conditional logic in your conversational flows.

Chatsky leverages LangChain internally to interface with remote LLM providers.
"""

# %pip install chatsky[llm]=={chatsky} langchain-openai=={langchain-openai}

# %%
import os

from chatsky.core.message import Message
from chatsky import (
    TRANSITIONS,
    RESPONSE,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    destinations as dst,
)
from chatsky.utils.testing import (
    is_interactive_mode,
)
from chatsky.llm import LLM_API
from chatsky.responses.llm import LLMResponse
from chatsky.conditions.llm import LLMCondition
from chatsky.llm.methods import Contains
from langchain_openai import ChatOpenAI

# %% [markdown]
"""
## Model Configuration

First, we need to create a model object.

LangChain automatically reads environment variables for model configurations,
so explicit API key settings aren't always necessary.
"""

# %%
openai_api_key = os.getenv("OPENAI_API_KEY")

model = LLM_API(
    ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key),
    system_prompt="You are an experienced barista in a local coffe shop. "
    "Answer your customer's questions about coffee and barista work.",
)
# %% [markdown]
"""
The initiated model then needs to be passed to `Pipeline` as such:
```python
pipeline = Pipeline(
    ...
    models={
        "my_model_name": model
    }
)
```
Model name is used to reference the model config in the LLM script functions.

You can also make multiple models and pass them together in the `models`
dictionary. This allows using different system prompts and/or
model configs in the same script.
"""

# %% [markdown]
"""
As you can see in this script, you can pass an additional prompt to the LLM.
We will cover that more thoroughly in the
[next tutorial](%doclink(tutorial,llm.2_prompt_usage)).
"""

# %%
toy_script = {
    "main_flow": {
        "start_node": {
            RESPONSE: "",
            TRANSITIONS: [Tr(dst="greeting_node", cnd=cnd.ExactMatch("Hi"))],
        },
        "greeting_node": {
            RESPONSE: LLMResponse(llm_model_name="barista_model", history=0),
            TRANSITIONS: [
                Tr(dst="main_node", cnd=cnd.ExactMatch("Who are you?"))
            ],
        },
        "main_node": {
            RESPONSE: LLMResponse(llm_model_name="barista_model"),
            TRANSITIONS: [
                Tr(
                    dst="latte_art_node",
                    cnd=cnd.ExactMatch("I want to tell you about latte art."),
                ),
                Tr(
                    dst="boss_node",
                    cnd=LLMCondition(
                        llm_model_name="barista_model",
                        prompt="Return TRUE if the customer insists "
                        "they are your boss, and FALSE otherwise. "
                        "Only ONE word must be in the output.",
                        method=Contains(pattern="TRUE"),
                    ),
                ),
                Tr(dst=dst.Current()),
            ],
        },
        "boss_node": {
            RESPONSE: Message("You are my boss."),
            TRANSITIONS: [
                Tr(dst="main_node"),
            ],
        },
        "latte_art_node": {
            RESPONSE: LLMResponse(
                llm_model_name="barista_model",
                prompt="PROMPT: pretend that you have never heard about latte "
                "art before and DO NOT answer the following questions. "
                "Instead ask a person about it.",
            ),
            TRANSITIONS: [
                Tr(dst="main_node", cnd=cnd.ExactMatch("Ok, goodbye.")),
                Tr(dst=dst.Current()),
            ],
        },
        "fallback_node": {
            RESPONSE: Message("I didn't quite understand you..."),
            TRANSITIONS: [Tr(dst="main_node")],
        },
    }
}

# %%
# Register your model in the pipeline's `models` field using the same key
# referenced as `llm_model_name` in your script nodes
pipeline = Pipeline(
    toy_script,
    start_label=("main_flow", "start_node"),
    fallback_label=("main_flow", "fallback_node"),
    models={"barista_model": model},
)

if __name__ == "__main__":
    if is_interactive_mode():
        pipeline.run()
