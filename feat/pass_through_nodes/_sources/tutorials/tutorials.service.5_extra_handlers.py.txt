# %% [markdown]
"""
# 5. Extra handlers

This tutorial shows usage of extra handlers:
functions that run before/after components.

For API ref, see:

* %mddoclink(api,core.service.extra,BeforeHandler)
* %mddoclink(api,core.service.extra,AfterHandler).
"""

# %pip install chatsky=={chatsky}

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
from chatsky.utils.testing.toy_script import HAPPY_PATH, TOY_SCRIPT_KWARGS

reload(logging)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="")
logger = logging.getLogger(__name__)

random.seed(0)

# %% [markdown]
"""
## Intro

Extra handlers are additional function
lists (before-functions and/or after-functions)
that can be added to any pipeline components (service and service groups).

Extra handlers main purpose should be statistics collection.

## Usage

Extra handlers can be attached to pipeline component using
`before_handler` and `after_handler` constructor parameter.

Extra handler callable signature can be one of the following:
`[ctx]` or `[ctx, info]`, where:

* `ctx` - `Context` of the current dialog.
* `info` - Dictionary, containing information about current extra handler
    and the pipeline component that called this.

    For example, `info.stage` will tell you if this Extra Handler is a
    BeforeHandler or AfterHandler; `info.component.name` will give
    you the component's name; `info.component.get_state(ctx)` will
    return the component's execution state (which is `NOT_RUN` for
    before handlers and `FINISHED` for after handlers).

### Extra Handler configuration

Instead of passing a list of functions as extra handler you can pass an instance
of either %mddoclink(api,core.service.extra,BeforeHandler) or
%mddoclink(api,core.service.extra,AfterHandler).

This allows changing the `timeout` and `concurrent` options to change
the way extra handlers are executed.

### Mass extra handler addition

You can use %mddoclink(api,core.service.group,ServiceGroup.add_extra_handler)
to add a function as an extra handler to a service group and if you pass
a `condition` function it will also add the extra handler to all its
subcomponents that satisfy the condition function.

## Code explanation

Here 5 `heavy_service`s are run in a single concurrent service group.
Each of them sleeps for random a amount of seconds (between 0 and 0.05).

To each of them (as well as the group)
time measurement extra handler is attached,
that writes execution time to `ctx.misc`.

In the end `ctx.misc` is logged to info channel.
"""


# %%
def collect_timestamp_before(ctx: Context, info: ExtraHandlerRuntimeInfo):
    ctx.misc.update({f"{info.component.path}": datetime.now()})


def collect_timestamp_after(ctx: Context, info: ExtraHandlerRuntimeInfo):
    ctx.misc.update(
        {
            f"{info.component.path}": datetime.now()
            - ctx.misc[f"{info.component.path}"]
        }
    )


async def heavy_service(_):
    await asyncio.sleep(random.randint(0, 5) / 100)


def logging_service(ctx: Context):
    logger.info(f"Context misc: {json.dumps(ctx.misc, indent=4, default=str)}")


# %%
pipeline = Pipeline(
    **TOY_SCRIPT_KWARGS,
    pre_services=ServiceGroup(
        before_handler=[collect_timestamp_before],
        after_handler=[collect_timestamp_after],
        components=[
            {
                "handler": heavy_service,
                "before_handler": [collect_timestamp_before],
                "after_handler": [collect_timestamp_after],
            }
        ]
        * 5,
    ),
    post_services=[logging_service],
)

# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH[:1], printout=True)
    if is_interactive_mode():
        pipeline.run()
