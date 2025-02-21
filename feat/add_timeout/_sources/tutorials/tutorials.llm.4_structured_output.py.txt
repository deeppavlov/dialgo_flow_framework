# %% [markdown]
"""
# LLM: 4. Structured Output

Chatsky provides two powerful ways to get structured output from LLMs:

1. **Using BaseModel**: To get structured text content (like JSON).
2. **Using Message subclass**: To add metadata to messages.

This tutorial demonstrates both approaches with practical examples.
"""

# %pip install chatsky[llm]=={chatsky} langchain-openai=={langchain-openai} langchain-anthropic=={langchain-anthropic} # noqa: E501
# %%
import os

from pydantic import BaseModel, Field

from chatsky import (
    TRANSITIONS,
    RESPONSE,
    GLOBAL,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
)
from chatsky.core.message import Message
from chatsky.utils.testing import is_interactive_mode
from chatsky.llm import LLM_API
from chatsky.responses.llm import LLMResponse

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Load API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize our models
movie_model = LLM_API(
    ChatAnthropic(
        model="claude-3.5-sonnet", api_key=anthropic_api_key, temperature=0
    ),
)
review_model = LLM_API(
    ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0),
)


# Define structured output schemas
class Movie(BaseModel):
    """Schema for movie details."""

    name: str = Field(description="Name of the movie")
    genre: str = Field(description="Genre of the movie")
    plot: str = Field(description="Plot of the movie in chapters")
    cast: list = Field(description="List of the actors")


class MovieReview(Message):
    """Schema for movie reviews (uses `Message.misc` for metadata)."""

    text: str = Field(description="The actual review text")
    misc: dict = Field(
        description="A dictionary with the following keys and values: "
        "k: rating v [int]: number between 0 and 5, "
        "k: spoiler_alert v [boolean]: is there a spoiler in this review"
    )


# %%

script = {
    GLOBAL: {
        TRANSITIONS: [
            Tr(
                dst=("greeting_flow", "start_node"),
                cnd=cnd.ExactMatch("/start"),
            ),
            Tr(dst=("movie_flow", "create"), cnd=cnd.ExactMatch("/create")),
            Tr(dst=("movie_flow", "review"), cnd=cnd.Regexp("/review .*")),
        ]
    },
    "greeting_flow": {
        "start_node": {
            RESPONSE: Message(
                "Welcome to MovieBot! Try:\n"
                "/create - Create a movie idea\n"
                "/review - Write a movie review"
            ),
        },
        "fallback_node": {
            RESPONSE: Message("I didn't understand. Try /create or /review"),
            TRANSITIONS: [Tr(dst="start_node")],
        },
    },
    "movie_flow": {
        "create": {
            RESPONSE: LLMResponse(
                llm_model_name="movie_model",
                prompt="Create a movie idea for the user.",
                message_schema=Movie,
            ),
            TRANSITIONS: [Tr(dst=("greeting_flow", "start_node"))],
        },
        "review": {
            RESPONSE: LLMResponse(
                llm_model_name="review_model",
                prompt="Generate a movie review based on user's input. "
                "Include rating, and mark if it contains spoilers. "
                "Use JSON with the `text` and `misc` fields "
                "to produce the output.",
                message_schema=MovieReview,
            ),
            TRANSITIONS: [Tr(dst=("greeting_flow", "start_node"))],
        },
    },
}

# %%
pipeline = Pipeline(
    script=script,
    start_label=("greeting_flow", "start_node"),
    fallback_label=("greeting_flow", "fallback_node"),
    models={"movie_model": movie_model, "review_model": review_model},
)

if __name__ == "__main__":
    if is_interactive_mode():
        pipeline.run()
