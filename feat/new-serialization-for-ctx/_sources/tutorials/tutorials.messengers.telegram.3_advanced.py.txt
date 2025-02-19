# %% [markdown]
"""
# Telegram: 3. Advanced

The following tutorial shows several advanced cases of user-to-bot interaction.

Here, %mddoclink(api,messengers.telegram.interface,LongpollingInterface)
class and [python-telegram-bot](https://docs.python-telegram-bot.org/)
library are used for accessing telegram API in polling mode.

Telegram API token is required to access telegram API.
"""

# %pip install chatsky[telegram]=={chatsky}

# %%
import os
from hashlib import sha256
from urllib.request import urlopen

from pydantic import HttpUrl
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from chatsky import (
    RESPONSE,
    TRANSITIONS,
    LOCAL,
    Message,
    Pipeline,
    BaseResponse,
    Context,
    Transition as Tr,
    conditions as cnd,
)
from chatsky.messengers.telegram import LongpollingInterface
from chatsky.core.message import (
    DataAttachment,
    Document,
    Image,
    Location,
    Sticker,
    MessageInitTypes,
)
from chatsky.utils.testing.common import is_interactive_mode


# %% [markdown]
"""
This bot shows different special telegram messenger interface use cases,
such as:

1. Interactive keyboard with buttons.
2. Text formatted with Markdown V2.
3. Multiple attachments of different kind handling.
4. Image with a spoiler.
5. Document with a thumbnail.
6. Attachment bytes hash.

Last option ("Raw attachments!") button might be especially interesting,
because it shows how bot percepts different telegram attachments sent by user
in terms and datastructures of Chatsky.

<div class="alert alert-info">

Tip

Check out
[this](https://docs.python-telegram-bot.org/en/v21.3/telegram.bot.html#telegram.Bot)
class for information about different arguments
for sending attachments, `send_...` methods.

For example, documentation for `Image` extra fields can be found in the
[send_photo](https://docs.python-telegram-bot.org/en/v21.3/telegram.bot.html#telegram.Bot.send_photo)
method.

The `Message` class also supports extra keywords as described in
[send_message](https://docs.python-telegram-bot.org/en/v21.3/telegram.bot.html#telegram.Bot.send_message).

</div>
"""

# %%

EXAMPLE_ATTACHMENT_SOURCE = (
    "https://github.com/deeppavlov/chatsky/wiki/example_attachments"
)

image_url = HttpUrl(f"{EXAMPLE_ATTACHMENT_SOURCE}/deeppavlov.png")

formatted_text = """
Visit [this link](https://core.telegram.org/bots/api#formatting-options)
for more information about formatting options in telegram\\.
"""  # noqa: W605

location_data = {"latitude": 59.9386, "longitude": 30.3141}

sticker_data = {
    "id": (
        "CAACAgIAAxkBAAErBZ1mKAbZvEOmhscojaIL5q0u8v"
        + "gp1wACRygAAiSjCUtLa7RHZy76ezQE"
    ),
}

image_data = {
    "source": image_url,
    "caption": "DeepPavlov logo",
    "has_spoiler": True,
    "filename": "deeppavlov_logo.png",
}
# last 3 fields are extra keywords passed directly to the
# telegram.Bot.send_photo method

document_data = {
    "source": HttpUrl(f"{EXAMPLE_ATTACHMENT_SOURCE}/deeppavlov-article.pdf"),
    "caption": "DeepPavlov article",
    "filename": "deeppavlov_article.pdf",
    "thumbnail": urlopen(str(image_url)).read(),
}


# %%
class DataAttachmentHash(BaseResponse):
    async def call(self, ctx: Context) -> MessageInitTypes:
        attachment = [
            a
            for a in ctx.last_request.attachments
            if isinstance(a, DataAttachment)
        ]
        if len(attachment) > 0:
            attachment_bytes = await attachment[0].get_bytes(
                ctx.pipeline.messenger_interface
            )
            attachment_hash = sha256(attachment_bytes).hexdigest()
            response = (
                "Here's your previous request first attachment sha256 hash:\n"
                f"```\n{attachment_hash}\n```"
            )
            return Message(text=response, parse_mode=ParseMode.MARKDOWN_V2)
        else:
            return "Last request did not contain any data attachment!"


# %%
script = {
    "main_flow": {
        LOCAL: {
            TRANSITIONS: [
                Tr(dst="main_node", cnd=cnd.ExactMatch("/start")),
                Tr(dst="formatted_node", cnd=cnd.HasCallbackQuery("formatted")),
                Tr(
                    dst="attachments_node",
                    cnd=cnd.HasCallbackQuery("attachments"),
                ),
                Tr(dst="secret_node", cnd=cnd.HasCallbackQuery("secret")),
                Tr(dst="thumbnail_node", cnd=cnd.HasCallbackQuery("thumbnail")),
                Tr(dst="hash_init_node", cnd=cnd.HasCallbackQuery("hash")),
                Tr(dst="main_node", cnd=cnd.HasCallbackQuery("restart")),
            ]
        },
        "start_node": {},
        "main_node": {
            RESPONSE: Message(
                text="Welcome! Choose what you want to receive.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Cute formatted text!",
                                callback_data="formatted",
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Multiple attachments!",
                                callback_data="attachments",
                            ),
                            InlineKeyboardButton(
                                "Secret image!", callback_data="secret"
                            ),
                            InlineKeyboardButton(
                                "Document with thumbnail!",
                                callback_data="thumbnail",
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "First attachment bytes hash!",
                                callback_data="hash",
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Restart!", callback_data="restart"
                            ),
                        ],
                    ],
                ),
                # you can add extra fields to the message itself;
                # they are passed to telegram.Bot.send_message
            ),
        },
        "formatted_node": {
            RESPONSE: Message(
                text=formatted_text, parse_mode=ParseMode.MARKDOWN_V2
            ),
        },
        "attachments_node": {
            RESPONSE: Message(
                text="Here's your message with multiple attachments "
                + "(a location and a sticker)!",
                attachments=[
                    Location(**location_data),
                    Sticker(**sticker_data),
                ],
            ),
        },
        "secret_node": {
            RESPONSE: Message(
                text="Here's your secret image!",
                attachments=[Image(**image_data)],
            ),
        },
        "thumbnail_node": {
            RESPONSE: Message(
                text="Here's your document with thumbnail!",
                attachments=[Document(**document_data)],
            ),
        },
        "hash_init_node": {
            RESPONSE: Message(
                text="Alright! Now send me a message with data attachment "
                + "(audio, video, animation, image, sticker or document)!"
            ),
            TRANSITIONS: [Tr(dst="hash_request_node")],
        },
        "hash_request_node": {
            RESPONSE: DataAttachmentHash(),
        },
        "fallback_node": {
            RESPONSE: Message(
                text="Bot has entered unrecoverable state:"
                + "\nRun /start command again to restart."
            ),
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
