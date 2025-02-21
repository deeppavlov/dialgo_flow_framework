# %% [markdown]
"""
# 3. Service Groups

The following tutorial shows how to group multiple services.

For more information, see
[API ref](%doclink(api,core.service.group,ServiceGroup)).
"""

# %pip install chatsky=={chatsky}

# %%
import asyncio
import logging
import sys
from importlib import reload

from chatsky.core import Context, Pipeline
from chatsky.core.service import ServiceGroup

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)
from chatsky.utils.testing.toy_script import HAPPY_PATH, TOY_SCRIPT_KWARGS

reload(logging)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="")
logger = logging.getLogger(__name__)

# %% [markdown]
"""
## Intro

Service groups are used to combine several services
(or service groups) into one.

Both services and service groups inherit interface from `PipelineComponent`
class which defines all the fields described in the [previous tutorial](
%doclink(tutorial,service.2_advanced))
except `handler`.

Instead of `handler` service group defines `components`:
a list of services or service groups.

Pipeline pre-services and post-services are actually service groups
and you can pass a ServiceGroup instead of a list when initializing Pipeline.

## Component execution

Components inside a service group are executed sequentially, except for
components with the `concurrent` attribute set to `True`:
Continuous sequences of concurrent components are executed concurrently
(via `asyncio.gather`).

For example, if components are `[1, 1, 0, 0, 1, 1, 1, 0]` where
"1" indicates a concurrent component, the components are executed as follows:

1. Components 1 and 2 (concurrently);
2. Component 3;
3. Component 4;
4. Components 5, 6 and 7 (concurrently);
5. Component 8.

<div class="alert alert-info">

Note

Components processing different contexts are always executed independently
of each other.

</div>

### Fully concurrent flag

Service groups have a `fully_concurrent` flag which makes it treat
every component inside it as concurrent,
running all components simultaneously.

This is convenient if you have a bunch of functions,
that you want to run simultaneously,
but don't want to make a service for each of them.

## Code explanation

In this example, we define `pre_services` as a `ServiceGroup` instead of a list.
This allows us to set the `fully_concurrent` flag to `True`.
The service group consists of 10 services that sleep 0.01 seconds each.
But since they are executed concurrently, the entire service group
takes much less time than 0.1 seconds.

To further demonstrate ServiceGroup's execution logic,
`post_services` is a ServiceGroup with concurrent components 'A' and 'B',
which execute simultaneously, and also one regular component 'C' at the end.

If 'A' and 'B' weren't concurrent, all steps for component 'A' would complete
before component 'B' begins its execution, but instead they start
at the same time. Only after both of them have finished,
does component 'C' start working.
"""


# %%
async def time_consuming_service(_):
    await asyncio.sleep(0.01)


def interact(stage: str, service: str):
    async def slow_service(_: Context):
        logger.info(f"{stage} with service {service}")
        await asyncio.sleep(0.1)

    return slow_service


pipeline = Pipeline(
    **TOY_SCRIPT_KWARGS,
    pre_services=ServiceGroup(
        components=[time_consuming_service for _ in range(0, 10)],
        fully_concurrent=True,
    ),
    post_services=[
        ServiceGroup(
            name="InteractWithServiceA",
            components=[
                interact("Starting interaction", "A"),
                interact("Finishing interaction", "A"),
            ],
            concurrent=True,
        ),
        ServiceGroup(
            name="InteractWithServiceB",
            components=[
                interact("Starting interaction", "B"),
                interact("Finishing interaction", "B"),
            ],
            concurrent=True,
        ),
        ServiceGroup(
            name="InteractWithServiceC",
            components=[
                interact("Starting interaction", "C"),
                interact("Finishing interaction", "C"),
            ],
            concurrent=False,
        ),
    ],
)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH[:1], printout=True)
    if is_interactive_mode():
        pipeline.run()
