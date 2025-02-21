# %% [markdown]
"""
# Telegram: 2. Attachments

The following tutorial shows how to send different attachments using
telegram interfaces.

Here, %mddoclink(api,messengers.telegram.interface,LongpollingInterface)
class and [python-telegram-bot](https://docs.python-telegram-bot.org/)
library are used for accessing telegram API in polling mode.

Telegram API token is required to access telegram API.
"""

# %pip install chatsky[telegram]=={chatsky}

# %%
import os

from pydantic import HttpUrl

from chatsky import (
    GLOBAL,
    RESPONSE,
    TRANSITIONS,
    Message,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
)
from chatsky.messengers.telegram import LongpollingInterface
from chatsky.core.message import (
    Animation,
    Audio,
    Contact,
    Document,
    Location,
    Image,
    MediaGroup,
    Poll,
    PollOption,
    Sticker,
    Video,
    VideoMessage,
    VoiceMessage,
)
from chatsky.utils.testing.common import is_interactive_mode


# %% [markdown]
"""
Example attachment data is specified below in form of dictionaries.
List of attachments that telegram messenger interface can send can
be found here:
%mddoclink(api,messengers.telegram.abstract,_AbstractTelegramInterface.supported_request_attachment_types).
"""

# %%

EXAMPLE_ATTACHMENT_SOURCE = (
    "https://github.com/deeppavlov/chatsky/wiki/example_attachments"
)

location_data = {"latitude": 50.65, "longitude": 3.916667}

contact_data = {
    "phone_number": "8-900-555-35-35",
    "first_name": "Hope",
    "last_name": "Credit",
}

sticker_data = {
    "id": (
        "CAACAgIAAxkBAAErAAFXZibO5ksphCKS"
        + "XSe1CYiw5588yqsAAkEAAzyKVxogmx2BPCogYDQE"
    ),
    "caption": "A sticker I've just found",
}

audio_data = {
    "source": HttpUrl(
        f"{EXAMPLE_ATTACHMENT_SOURCE}/separation-william-king.mp3"
    ),
    "caption": "Separation melody by William King",
    "filename": "separation-william-king.mp3",
}

video_data = {
    "source": HttpUrl(
        f"{EXAMPLE_ATTACHMENT_SOURCE}/crownfall-lags-nkognit0.mp4"
    ),
    "caption": "Epic Dota2 gameplay by Nkognit0",
    "filename": "crownfall-lags-nkognit0.mp4",
}

animation_data = {
    # For some reason, if we don't define filename explicitly,
    # animation is sent as file.
    "source": HttpUrl(
        f"{EXAMPLE_ATTACHMENT_SOURCE}/hong-kong-simplyart4794.gif"
    ),
    "caption": "Hong Kong skyscraper views by Simplyart4794",
    "filename": "hong-kong-simplyart4794.gif",
}

image_data = {
    "source": HttpUrl(f"{EXAMPLE_ATTACHMENT_SOURCE}/deeppavlov.png"),
    "caption": "DeepPavlov logo",
    "filename": "deeppavlov.png",
}

document_data = {
    "source": HttpUrl(f"{EXAMPLE_ATTACHMENT_SOURCE}/deeppavlov-article.pdf"),
    "caption": "DeepPavlov article",
    "filename": "deeppavlov-article.pdf",
}

ATTACHMENTS = [
    "location",
    "contact",
    "poll",
    "sticker",
    "audio",
    "video",
    "animation",
    "image",
    "document",
    "voice_message",
    "video_message",
    "media_group",
]

QUOTED_ATTACHMENTS = [f'"{attachment}"' for attachment in ATTACHMENTS]


# %% [markdown]
"""
The bot below sends different attachments on request.

[Here](%doclink(api,core.message)) you can find
all the attachment options available.
"""

# %%
script = {
    GLOBAL: {
        TRANSITIONS: [
            Tr(
                dst=("main_flow", f"{attachment}_node"),
                cnd=cnd.ExactMatch(attachment),
            )
            for attachment in ATTACHMENTS
        ]
    },
    "main_flow": {
        "start_node": {
            TRANSITIONS: [Tr(dst="intro_node", cnd=cnd.ExactMatch("/start"))],
        },
        "intro_node": {
            RESPONSE: f'Type {", ".join(QUOTED_ATTACHMENTS[:-1])}'
            f" or {QUOTED_ATTACHMENTS[-1]}"
            f" to receive a corresponding attachment!",
        },
        "location_node": {
            RESPONSE: Message(
                text="Here's your location!",
                attachments=[Location(**location_data)],
            ),
        },
        "contact_node": {
            RESPONSE: Message(
                text="Here's your contact!",
                attachments=[Contact(**contact_data)],
            ),
        },
        "poll_node": {
            RESPONSE: Message(
                text="Here's your poll!",
                attachments=[
                    Poll(
                        question="What is the poll question?",
                        options=[
                            PollOption(text="This one!"),
                            PollOption(text="Not this one :("),
                        ],
                    ),
                ],
            ),
        },
        "sticker_node": {
            RESPONSE: Message(
                text="Here's your sticker!",
                attachments=[Sticker(**sticker_data)],
            ),
        },
        "audio_node": {
            RESPONSE: Message(
                text="Here's your audio!",
                attachments=[Audio(**audio_data)],
            ),
        },
        "video_node": {
            RESPONSE: Message(
                text="Here's your video!",
                attachments=[Video(**video_data)],
            ),
        },
        "animation_node": {
            RESPONSE: Message(
                text="Here's your animation!",
                attachments=[Animation(**animation_data)],
            ),
        },
        "image_node": {
            RESPONSE: Message(
                text="Here's your image!",
                attachments=[Image(**image_data)],
            ),
        },
        "document_node": {
            RESPONSE: Message(
                text="Here's your document!",
                attachments=[Document(**document_data)],
            ),
        },
        "voice_message_node": {
            RESPONSE: Message(
                text="Here's your voice message!",
                attachments=[VoiceMessage(source=audio_data["source"])],
            ),
        },
        "video_message_node": {
            RESPONSE: Message(
                text="Here's your video message!",
                attachments=[VideoMessage(source=video_data["source"])],
            ),
        },
        "media_group_node": {
            RESPONSE: Message(
                text="Here's your media group!",
                attachments=[
                    MediaGroup(
                        group=[
                            Image(**image_data),
                            Video(**video_data),
                        ],
                    )
                ],
            ),
        },
        "fallback_node": {
            RESPONSE: f"Unknown attachment type, try again! "
            f"Supported attachments are: "
            f'{", ".join(QUOTED_ATTACHMENTS[:-1])} '
            f"and {QUOTED_ATTACHMENTS[-1]}.",
        },
    },
}


# %%
interface = LongpollingInterface(token=os.environ["TG_BOT_TOKEN"])


# %%
pipeline = Pipeline(
    script=script,
    start_label=("main_flow", "start_node"),
    fallback_label=("main_flow", "fallback_node"),
    messenger_interface=interface,
    # The interface can be passed as a pipeline argument.
)


if __name__ == "__main__":
    if is_interactive_mode():
        # prevent run during doc building
        pipeline.run()
