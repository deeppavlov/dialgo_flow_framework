# %% [markdown]
"""
# Responses: 1. Media

Here, %mddoclink(api,core.message,Attachment) class is shown.
Attachments can be used for attaching different media elements
(such as %mddoclink(api,core.message,Image),
%mddoclink(api,core.message,Document)
or %mddoclink(api,core.message,Audio)).

They can be attached to any message but will only work if the chosen
[messenger interface](%doclink(api,index_messenger_interfaces)) supports them.
"""

# %pip install chatsky=={chatsky}

# %%
from chatsky import (
    RESPONSE,
    TRANSITIONS,
    Message,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    destinations as dst,
)
from chatsky.core.message import Image

from chatsky.utils.testing import (
    check_happy_path,
    is_interactive_mode,
)


# %%
img_url = "https://www.python.org/static/img/python-logo.png"
toy_script = {
    "root": {
        "start": {
            TRANSITIONS: [Tr(dst=("pics", "ask_picture"))],
        },
        "fallback": {
            RESPONSE: "Final node reached, send any message to restart.",
            TRANSITIONS: [Tr(dst=("pics", "ask_picture"))],
        },
    },
    "pics": {
        "ask_picture": {
            RESPONSE: "Please, send me a picture url",
            TRANSITIONS: [
                Tr(
                    dst=("pics", "send_one"),
                    cnd=cnd.Regexp(r"^http.+\.png$"),
                ),
                Tr(
                    dst=("pics", "send_many"),
                    cnd=cnd.Regexp(f"{img_url} repeat 10 times"),
                ),
                Tr(
                    dst=dst.Current(),
                ),
            ],
        },
        "send_one": {
            RESPONSE: Message(  # need to use the Message class to send images
                text="here's my picture!",
                attachments=[Image(source=img_url)],
            ),
        },
        "send_many": {
            RESPONSE: Message(
                text="Look at my pictures",
                attachments=[Image(source=img_url)] * 10,
            ),
        },
    },
}

happy_path = (
    ("Hi", "Please, send me a picture url"),
    ("no", "Please, send me a picture url"),
    (
        img_url,
        Message(
            text="here's my picture!",
            attachments=[Image(source=img_url)],
        ),
    ),
    ("ok", "Final node reached, send any message to restart."),
    ("ok", "Please, send me a picture url"),
    (
        f"{img_url} repeat 10 times",
        Message(
            text="Look at my pictures",
            attachments=[Image(source=img_url)] * 10,
        ),
    ),
    ("ok", "Final node reached, send any message to restart."),
)


# %%
pipeline = Pipeline(
    script=toy_script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
)

if __name__ == "__main__":
    check_happy_path(pipeline, happy_path, printout=True)
    if is_interactive_mode():
        pipeline.run()
