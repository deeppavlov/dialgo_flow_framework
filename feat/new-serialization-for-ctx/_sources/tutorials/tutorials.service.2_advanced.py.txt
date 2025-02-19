# %% [markdown]
"""
# 2. Advanced services

This tutorial demonstrates various configuration options for services.

For more information, see
[API ref](%doclink(api,core.service.service,Service)).
"""

# %pip install chatsky=={chatsky}

# %%
import logging
import sys
from importlib import reload

from chatsky import Context, Pipeline, BaseProcessing
from chatsky.core.service import Service
from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)

from chatsky.utils.testing.toy_script import TOY_SCRIPT_KWARGS, HAPPY_PATH

reload(logging)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="")
logger = logging.getLogger(__name__)

# %% [markdown]
"""
## Intro

In the previous tutorial we used a function as a Service.
Under the hood a function is converted to a `Service` object
with the function as its `handler` argument.

The `Service` model has other arguments that modify its execution.

## Service Arguments

* `handler` - Function or `BaseProcessing`.
* `before_handler` - a list of functions that run before the service.
        You can read more about the handlers in this [tutorial](
        %doclink(tutorial,service.5_extra_handlers)
        ).
* `after_handler` - a list of functions that run after the service.
        You can read more about the handlers in this [tutorial](
        %doclink(tutorial,service.5_extra_handlers)
        ).
* `timeout` - service timeout.
* `concurrent` - whether this service can run concurrently,
        see [tutorial 3](
        %doclink(tutorial,service.3_groups)).
* `start_condition` - service start condition, see [tutorial 4](
        %doclink(tutorial,service.4_conditions_and_paths)).
* `name` - name of the service,
        see [tutorial 4](
        %doclink(tutorial,service.4_conditions_and_paths)).

## Service subclassing

Services can also be defined as subclasses of `Service`,
allowing access to all the fields described above via `self`.

To do this, derive your class from `Service`,
then implement an async `call` method which will
now replace the `handler` (see the `PreProcess` example below).

<div class="alert alert-info">

Tip

When defining a service as a subclass of `Service`, you can also change
default parameters such as `timeout` or `start_condition`.

</div>

## Code explanation

In this example, pipeline contains three services,
defined in three different ways.

The first is defined as a Service with a function handler.

The second derives from the `Service` class.

The third is defined as a Service with a processing handler.
"""


# %%
async def function_handler(ctx: Context):
    logger.info(
        "function_handler running:\n"
        "timeout of this service cannot be determined"
    )


class ServiceSubclass(Service):
    async def call(self, ctx: Context):
        logger.info(
            f"{self.name or self.computed_name} running:\n"
            f"timeout: {self.timeout}"
        )

    timeout: float = 1.0
    # this overrides the default `None` timeout,
    # but can still be overridden in class instances


class ProcessingService(BaseProcessing):
    async def call(self, ctx: Context) -> None:
        try:
            logger.info(self.timeout)
        except AttributeError:
            # this is BaseProcessing not Service so there's no `timeout` field
            logger.info(
                "ProcessingService running:\n"
                "timeout of this service cannot be determined"
            )


pipeline = Pipeline(
    **TOY_SCRIPT_KWARGS,
    pre_services=[
        Service(
            handler=function_handler,
            timeout=0.5,
        ),
        ServiceSubclass(name="ServiceSubclassWithCustomName", timeout=100),
        ServiceSubclass(),
        Service(handler=ProcessingService(), timeout=4),
    ],
)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH[:1], printout=True)
    if is_interactive_mode():
        pipeline.run()
