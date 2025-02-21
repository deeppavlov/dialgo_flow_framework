# %% [markdown]
"""
# 1. Basic services

This tutorial shows basics of using services.

Here, %mddoclink(api,core.ctx_utils,ContextMainInfo.misc)
dictionary of context is used for storing additional data.
"""

# %pip install chatsky=={chatsky}

# %%
import logging
import sys
from importlib import reload

from chatsky import Context, Pipeline

from chatsky.utils.testing import (
    check_happy_path,
    is_interactive_mode,
    HAPPY_PATH,
    TOY_SCRIPT_KWARGS,
)

reload(logging)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="")
# fix jupyter logs display

logger = logging.getLogger(__name__)


# %% [markdown]
"""
When Pipeline is created, additional pre-
and post-services can be defined. These run before and after
`Actor` respectively.

<div class="alert alert-info">

Reminder

`Actor` is a Pipeline component that processes user request, determines
the next node and generates a response from that node.

</div>

Services can be used to access external APIs, annotate user input, etc.

Service callables only take one parameter: `ctx` (Context) and have no return.

Here a pre-service ("ping") and
a post-service ("pong") are added to the pipeline.
"""


# %%
def ping_processor(ctx: Context):
    logger.info("ping - ...")
    ctx.misc["ping"] = True


# services can be both sync and async
async def pong_processor(ctx: Context):
    ping_pong = ctx.misc.get("ping", False)
    logger.info("... - pong")
    logger.info(
        f"Ping-pong exchange: " f"{'completed' if ping_pong else 'failed'}."
    )


pipeline = Pipeline(
    **TOY_SCRIPT_KWARGS,  # contains script, start and fallback labels
    pre_services=[ping_processor],
    post_services=[pong_processor],
    # To add a service simply add it to the corresponding service list
)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH[:1], printout=True)
    if is_interactive_mode():
        pipeline.run()
