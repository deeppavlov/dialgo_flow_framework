# %% [markdown]
# # Web API: 4. Streamlit chat interface
#
# This tutorial shows how to use an API endpoint created in the FastAPI tutorial
# in a Streamlit chat.
#
# A demonstration of the chat:
# ![demo](https://user-images.githubusercontent.com/61429541/238721597-ef88261d-e9e6-497d-ba68-0bcc9a765808.png)

# %pip install chatsky=={chatsky} streamlit=={streamlit} streamlit-chat=={streamlit-chat} # noqa: E501

# %% [markdown]
# ## Running Streamlit:
#
# ```bash
# streamlit run {file_name}
# ```


# %% [markdown]
# ## Module and package import


# %%
###########################################################
# This patch is only needed to import Message from chatsky.
# Streamlit Chat interface can be written without using it.
import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
###########################################################


# %%
import uuid
import itertools

import requests
import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components
from chatsky import Message


# %% [markdown]
# ## API configuration
#
# Here we define methods to contact the API endpoint.


# %%
API_URL = "http://localhost:8000/chat"


def query(payload, user_id) -> requests.Response:
    response = requests.post(
        API_URL + f"?user_id={user_id}",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    return response


# %% [markdown]
# ## Streamlit configuration
#
# Here we configure Streamlit page and initialize some session variables:
#
# 1. `user_id` -- stores user_id to be used in pipeline.
# 2. `bot_responses` -- a list of bot responses.
# 3. `user_requests` -- a list of user requests.


# %%
st.set_page_config(page_title="Streamlit Chatsky Chat", page_icon=":robot:")

st.header("Streamlit Chatsky Chat")

if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

if "bot_responses" not in st.session_state:
    st.session_state["bot_responses"] = []

if "user_requests" not in st.session_state:
    st.session_state["user_requests"] = []


# %% [markdown]
# ## UI setup
#
# Here we configure elements that will be used
# in Streamlit to interact with the API.
#
# First we define a text input field which
# a user is supposed to type his requests into.
# Then we define a button that sends a query
# to the API, logs requests and responses,
# and clears the text field.


# %%
def send_and_receive():
    """
    Send text inside the input field. Receive response from API endpoint.

    Add both the request and response to `user_requests` and `bot_responses`.

    We do not call this function inside the `text_input.on_change` because then
    we'd call it whenever the text field loses focus
    (e.g. when a browser tab is switched).
    """
    user_request = st.session_state["input"]

    if user_request == "":
        return

    st.session_state["user_requests"].append(user_request)

    bot_response = query(
        Message(user_request).model_dump(),
        user_id=st.session_state["user_id"],
    )
    bot_response.raise_for_status()

    bot_message = Message.model_validate(bot_response.json()).text

    # # Implementation without using Message:
    # bot_response = query(
    #     {"text": user_request},
    #     user_id=st.session_state["user_id"]
    # )
    # bot_response.raise_for_status()
    #
    # bot_message = bot_response.json()["response"]["text"]

    st.session_state["bot_responses"].append(bot_message)

    st.session_state["input"] = ""


# %%
st.text_input("You: ", key="input")
st.button("Send", on_click=send_and_receive)


# %% [markdown]
# ### Component patch
#
# Here we add a component that presses the
# `Send` button whenever user presses the `Enter` key.


# %%
components.html(
    """
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=secondary]'));
const send_button = buttons.find(el => el.innerText === 'Send');
doc.addEventListener('keypress', function(e) {
    switch (e.keyCode) {
        case 13: // (13 = Enter key)
            send_button.click();
            break;
    }
});
</script>
""",
    height=0,
    width=0,
)


# %% [markdown]
# ### Message display
#
# Here we use the `streamlit-chat` package to
# display user requests and bot responses.


# %%
for i, bot_response, user_request in zip(
    itertools.count(0),
    st.session_state.get("bot_responses", []),
    st.session_state.get("user_requests", []),
):
    message(user_request, key=f"{i}_user", is_user=True)
    message(bot_response, key=f"{i}_bot")
