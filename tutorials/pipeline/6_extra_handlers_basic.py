# %% [markdown]
"""
# 6. Extra Handlers (basic)

The following tutorial shows extra handlers possibilities and use cases.

Here, extra handlers %mddoclink(api,core.service.extra,BeforeHandler)
and %mddoclink(api,core.service.extra,AfterHandler)
are shown as additional means of data processing, attached to services.
"""

# %pip install chatsky

# %%
import asyncio
import json
import random
import logging
import sys
from importlib import reload
from datetime import datetime

from chatsky.core.service import (
    ServiceGroup,
    ExtraHandlerRuntimeInfo,
)
from chatsky import Context, Pipeline
from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)
from chatsky.utils.testing.toy_script import HAPPY_PATH, TOY_SCRIPT

reload(logging)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="")
logger = logging.getLogger(__name__)

# %% [markdown]
"""
Extra handlers are additional function
    lists (before-functions and/or after-functions)
    that can be added to any `pipeline` components (service and service groups).
Extra handlers main purpose should be service
and service groups statistics collection.
Extra handlers can be attached to pipeline component using
`before_handler` and `after_handler` constructor parameter.

Here 5 `heavy_service`s are run in single asynchronous service group.
Each of them sleeps for random amount of seconds (between 0 and 0.05).
To each of them (as well as to group)
    time measurement extra handler is attached,
    that writes execution time to `ctx.misc`.
In the end `ctx.misc` is logged to info channel.
"""


# %%
def collect_timestamp_before(ctx: Context, info: ExtraHandlerRuntimeInfo):
    ctx.misc.update({f"{info.component.name}": datetime.now()})


def collect_timestamp_after(ctx: Context, info: ExtraHandlerRuntimeInfo):
    ctx.misc.update(
        {
            f"{info.component.name}": datetime.now()
            - ctx.misc[f"{info.component.name}"]
        }
    )


async def heavy_service(_):
    await asyncio.sleep(random.randint(0, 5) / 100)


def logging_service(ctx: Context):
    logger.info(f"Context misc: {json.dumps(ctx.misc, indent=4, default=str)}")


# %%
pipeline_dict = {
    "script": TOY_SCRIPT,
    "start_label": ("greeting_flow", "start_node"),
    "fallback_label": ("greeting_flow", "fallback_node"),
    "pre_services": ServiceGroup(
        before_handler=[collect_timestamp_before],
        after_handler=[collect_timestamp_after],
        components=[
            {
                "handler": heavy_service,
                "before_handler": [collect_timestamp_before],
                "after_handler": [collect_timestamp_after],
            },
            {
                "handler": heavy_service,
                "before_handler": [collect_timestamp_before],
                "after_handler": [collect_timestamp_after],
            },
            {
                "handler": heavy_service,
                "before_handler": [collect_timestamp_before],
                "after_handler": [collect_timestamp_after],
            },
            {
                "handler": heavy_service,
                "before_handler": [collect_timestamp_before],
                "after_handler": [collect_timestamp_after],
            },
            {
                "handler": heavy_service,
                "before_handler": [collect_timestamp_before],
                "after_handler": [collect_timestamp_after],
            },
        ],
    ),
    "post_services": logging_service,
}

# %%
pipeline = Pipeline(**pipeline_dict)

if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    if is_interactive_mode():
        pipeline.run()
