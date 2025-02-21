# %% [markdown]
"""
# 4. Conditions and paths

This tutorial explains how a unique path is generated for each component
and how to add a condition for component execution.

[API ref for service conditions](%doclink(api,conditions.service))
"""

# %pip install chatsky=={chatsky}

# %%
import logging
import sys
from importlib import reload

from chatsky.conditions import Not, All, ServiceFinished
from chatsky.core.service import Service, ServiceGroup
from chatsky import Pipeline, Context, AnyCondition

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
## Component paths

Each component has a unique path that can be used as an ID for that component.

A path of a component is a "." concatenation of names of service groups this
component is in and the name of this component.

For example, if component "c" is in group "b" which itself is in group "a"
then path of "c" is "a.b.c".

<div class="alert alert-warning">

Naming services

When choosing a name for a component, keep in mind that
it should be unique among other components in the same service group.

</div>

### Computed names

If a component does not have a name, one is provided to it:

1. Each component has `computed_name` property which returns a default name.
    All service groups have "service_group" as their computed name.
    For services the computed name is the name of the handler **or** of the
    service class if handler is not present.

    E.g. A service with handler `my_func` has computed name "my_func";
    and if service `MyService` is a subclass of `Service` and does not
    define `handler` its computed name is "MyService".
2. If there are two or more components in a service group with the same
    computed name, they are assigned names with an incrementing postfix.
    For example, if there are two service groups with no name in pre-services
    they will be named "service_group#0" and "service_group#1".

### Names of basic components

* Main component (pipeline) has an empty name;
* Name of the pre services group is "pre";
* Name of the post services group is "post";
* Name of the actor service is "actor".

For example, if you have `pre_services=[my_func]`, the full path of the
`my_func` component is ".pre.my_func".

## Component start condition

Any component (service or service group) can have a `start_condition`.

Start condition is a `BaseCondition` that determines whether the component
should be executed.

For more information about conditions, see the [condition tutorial](
%doclink(tutorial,script.core.2_conditions)).

### Component status

At any time each component has a certain status:

* `NOT_RUN` - Component hasn't bee executed yet or
    start condition returned False.
* `RUNNING` - Component is currently being executed.
* `FINISHED` - Component finished successfully.
* `FAILED` - Component execution failed.

For more information, see
%mddoclink(api,core.ctx_utils,FrameworkData.service_states).

### ServiceFinished condition

`ServiceFinished` is a condition that returns `True` if another service
has the `FINISHED` status.

`ServiceFinished` accepts the following constructor parameters:

* `path` (required) - a path of another component.
* `wait` - whether it should wait for the component to complete.
    This means that the component status cannot be
    `NOT_RUN` or `RUNNING` at the time of the check.
    Defaults to `False`.

<div class="alert alert-warning">

Warning!

It is possible for the pipeline to get stuck in infinite waiting
with the `ServiceFinished` condition.

Either disable `wait` or set a timeout for the service.

</div>

For more information about `ServiceFinished`, see [API ref](
%doclink(api,conditions.service,ServiceFinished)).

## Code explanation

In this example, two conditionally executed services are illustrated.

The service named `running_service` is executed
only if both `SimpleServices` in pre service group
have finished successfully.

`never_running_service` is executed only if `running_service` is not finished,
which should never happen.

Lastly, `context_printing_service` prints pipeline runtime information,
that contains execution state of all previously run services.
"""


# %%
class SimpleService(Service):
    async def call(self, _: Context):
        logger.info(f"Service '{self.name}' is running...")


class NeverRunningService(Service):
    async def call(self, _: Context):
        raise Exception(f"Oh no! The '{self.name}' service is running!")

    start_condition: AnyCondition = Not(
        ServiceFinished(".post.named_group.running_service", wait=True)
    )


class RuntimeInfoPrintingService(Service):
    async def call(self, _: Context):
        logger.info(
            f"Service '{self.name}' runtime execution info:"
            f"{self.model_dump_json(indent=4)}"
        )


# %%
pipeline = Pipeline(
    **TOY_SCRIPT_KWARGS,
    pre_services=[
        SimpleService(),
        # This service will be named "SimpleService#0"
        SimpleService(),
        # This service will be named "SimpleService#1"
    ],
    # this group is named "pre"
    post_services=[
        ServiceGroup(
            name="named_group",
            components=[
                SimpleService(
                    start_condition=All(
                        ServiceFinished(".pre.SimpleService#0"),
                        ServiceFinished(".pre.SimpleService#1"),
                    ),
                    # Alternative:
                    # ServiceFinished(".pre")
                    name="running_service",
                ),
                # This simple service is named "running_service"
                NeverRunningService(),
                # this service will be named "NeverRunningService"
            ],
            fully_concurrent=True,
            # Makes components in the group run asynchronously,
            # unless one is waiting for another to complete,
            # which is what happens with NeverRunningService.
        ),
        RuntimeInfoPrintingService(),
    ],
)

# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH[:1], printout=True)
    if is_interactive_mode():
        pipeline.run()
