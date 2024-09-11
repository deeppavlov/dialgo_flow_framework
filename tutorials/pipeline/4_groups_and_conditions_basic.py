# %% [markdown]
"""
# 4. Groups and conditions (basic)

The following example shows `pipeline` service group usage and start conditions.

Here, %mddoclink(api,core.service.service,Service)s
and %mddoclink(api,core.service.group,ServiceGroup)s
are shown for advanced data pre- and postprocessing based on conditions.
"""

# %pip install chatsky

# %%
import json
import logging

from chatsky.core import Context, BaseProcessing
from chatsky.core.service import Service, ServiceFinishedCondition
from chatsky.conditions import Not
from chatsky import Pipeline

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)
from chatsky.utils.testing.toy_script import HAPPY_PATH, TOY_SCRIPT

logger = logging.getLogger(__name__)


# %% [markdown]
"""
Pipeline can contain not only single services, but also service groups.
Service groups can be defined as `ServiceGroup` objects:
      lists of `Service` or more `ServiceGroup` objects.
`ServiceGroup` objects should contain `components` -
a list of `Service` and `ServiceGroup` objects.

To receive serialized information about service,
    service group or pipeline a property `info_dict` can be used,
    it returns important object properties as a dict.

Services and service groups can be executed conditionally.
Conditions are functions passed to `start_condition` argument.
They have the following signature

    class MyCondition(BaseCondition):
        async def call(self, ctx: Context) -> bool:

Service is only executed if its `start_condition` returned `True`.
By default all the services start unconditionally.
There are number of built-in condition functions. (see `Script` tutorial 2)
Though there is also a built-in condition `ServiceFinishedCondition`
that returns `True` if a `Service` with a given path completed successfully,
returns `False` otherwise.

`ServiceFinishedCondition` accepts the following constructor parameters:

* `path` (required) - a path to the `Service`.
* `wait` - whether it should wait for the said `Service` to complete.
        defaults to `False`.

Here there is a conditionally executed service named
`never_running_service` is always executed.
It is executed only if `always_running_service`
is not finished, that should never happen.
The service named `context_printing_service`
prints pipeline runtime information,
that contains execution state of all previously run services.
"""


# %%
class AlwaysRunningService(Service):
    def call(self, _: Context):
        logger.info(f"Service '{self.name}' is running...")


class NeverRunningService(Service):
    def call(self, _: Context):
        raise Exception(f"Oh no! The '{self.name}' service is running!")


class RuntimeInfoPrintingService(Service):
    def call(self, _: Context):
        logger.info(
            f"Service '{self.name}' runtime execution info:"
            f"{json.dumps(self.info_dict, indent=4, default=str)}"
        )


# %%
pipeline_dict = {
    "script": TOY_SCRIPT,
    "start_label": ("greeting_flow", "start_node"),
    "fallback_label": ("greeting_flow", "fallback_node"),
    "pre_services": Service(
        handler=AlwaysRunningService(), name="always_running_service"
    ),
    "post_services": [
        Service(
            handler=NeverRunningService(),
            start_condition=Not(
                ServiceFinishedCondition(
                    path=".pipeline.pre.AlwaysRunningService"
                )  # pre services belong to the "pre" group; post -- to "post"
            ),
        ),
        Service(
            handler=RuntimeInfoPrintingService,
            name="runtime_info_printing_service",
        ),
    ],
}


# %%
pipeline = Pipeline.model_validate(pipeline_dict)

if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    if is_interactive_mode():
        pipeline.run()
