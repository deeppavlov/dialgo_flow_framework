# %% [markdown]
"""
# Telegram: 1. Basic

The following tutorial shows how to run a regular Chatsky script in Telegram.
It asks users for the '/start' command and then loops in one place.

Here, %mddoclink(api,messengers.telegram.interface,LongpollingInterface)
class and [python-telegram-bot](https://docs.python-telegram-bot.org/)
library are used for accessing telegram API in polling mode.

Telegram API token is required to access telegram API.
"""

# %pip install chatsky[telegram]=={chatsky}

# %%
import os

from chatsky import (
    RESPONSE,
    TRANSITIONS,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    destinations as dst,
)
from chatsky.messengers.telegram import LongpollingInterface
from chatsky.utils.testing.common import is_interactive_mode


# %% [markdown]
"""
In order to integrate your script with Telegram, you need an instance of the
%mddoclink(api,messengers.telegram.abstract,_AbstractTelegramInterface) class.
Two of its child subclasses that can be instantiated
are %mddoclink(api,messengers.telegram.interface,LongpollingInterface) and
%mddoclink(api,messengers.telegram.interface,WebhookInterface).
The latter requires a webserver, so here we use long polling interface.

%mddoclink(api,messengers.telegram.abstract,_AbstractTelegramInterface)
encapsulates the bot logic. The only required
argument for a bot to run is a token. Some other parameters
(such as host, port, interval, etc.) can be passed as keyword arguments
(for their specs see [documentations of the child subclasses](
%doclink(api,messengers.telegram.interface)
).

Either of the two interfaces connect the bot to Telegram.
They can be passed directly to a Chatsky `Pipeline` instance.


<div class="alert alert-info">

Note

You can also import `LongpollingInterface`
under the alias of `TelegramInterface` from `chatsky.messengers`:

```python
from chatsky.messengers import TelegramInterface
```

</div>
"""


# %%
script = {
    "greeting_flow": {
        "start_node": {
            TRANSITIONS: [
                Tr(dst="greeting_node", cnd=cnd.ExactMatch("/start"))
            ],
        },
        "greeting_node": {
            RESPONSE: "Hi",
            TRANSITIONS: [Tr(dst=dst.Current())],
        },
        "fallback_node": {
            RESPONSE: "Please, repeat the request",
            TRANSITIONS: [
                Tr(dst="greeting_node", cnd=cnd.ExactMatch("/start"))
            ],
        },
    }
}


# %%
interface = LongpollingInterface(token=os.environ["TG_BOT_TOKEN"])


# %%
pipeline = Pipeline(
    script=script,
    start_label=("greeting_flow", "start_node"),
    fallback_label=("greeting_flow", "fallback_node"),
    messenger_interface=interface,
    # The interface can be passed as a pipeline argument.
)


if __name__ == "__main__":
    if is_interactive_mode():
        # prevent run during doc building
        pipeline.run()
