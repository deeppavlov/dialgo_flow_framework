"""
Service Group
-------------
The Service Group module contains the
:py:class:`~.ServiceGroup` class, which is used to represent a group of related services.
This class provides a way to organize and manage multiple services as a single unit,
allowing for easier management and organization of the services within the pipeline.
The :py:class:`~.ServiceGroup` serves the important function of grouping services to work together in parallel.
"""

from __future__ import annotations
import asyncio
import logging
from typing import List, Union, Awaitable, TYPE_CHECKING, Any
from pydantic import model_validator

from chatsky.script import Context
from ..pipeline.actor import Actor

from ..pipeline.component import PipelineComponent
from ..types import (
    ComponentExecutionState,
    GlobalExtraHandlerType,
    ExtraHandlerConditionFunction,
    ExtraHandlerFunction,
)
from .service import Service

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from chatsky.pipeline.pipeline.pipeline import Pipeline


# I think it's fine calling this a `Service` group, even though really it's a `PipelineComponent` group.
# The user only sees this as a `Service` group like they should.
# arbitrary_types_allowed for testing, will remove later
class ServiceGroup(PipelineComponent, extra="forbid", arbitrary_types_allowed=True):
    """
    A service group class.
    Service group can be included into pipeline as an object or a pipeline component list.
    Service group can be synchronous or asynchronous.
    Components in synchronous groups are executed consequently (no matter is they are synchronous or asynchronous).
    Components in asynchronous groups are executed simultaneously.
    Group can be asynchronous only if all components in it are asynchronous.

    :param components: A `ServiceGroup` object, that will be added to the group.
    :type components: :py:data:`~.ServiceGroup`
    :param before_handler: List of `_ComponentExtraHandler` to add to the group.
    :type before_handler: Optional[:py:data:`~._ComponentExtraHandler`]
    :param after_handler: List of `_ComponentExtraHandler` to add to the group.
    :type after_handler: Optional[:py:data:`~._ComponentExtraHandler`]
    :param timeout: Timeout to add to the group.
    :param requested_async_flag: Requested asynchronous property.
    :param start_condition: :py:data:`~.StartConditionCheckerFunction` that is invoked before each group execution;
        group is executed only if it returns `True`.
    :param name: Requested group name.
    """

    # If this is a list of PipelineComponents, why would the program know this is supposed to be a Service in the end?
    # It's kind of logical it would try to match the best one fitting, but there are no guarantees, right?
    # components: List[PipelineComponent]
    components: List[
        Union[
            Actor,
            Service,
            ServiceGroup,
        ]
    ]

    # Whenever data isn't passed like a dictionary, this tries to cast it to the right dictionary
    # This includes List, PipelineComponent and Callable.
    @model_validator(mode="before")
    @classmethod
    # Here Script class has "@validate_call". Is it needed here?
    def components_constructor(cls, data: Any):
        # Question: I don't think shallow copy() could be a problem for this, right?
        # Pydantic is already rather recursively checking types.
        # print(data)
        if not isinstance(data, dict):
            result = {"components": data}
        else:
            result = data.copy()
        # When it's a dictionary, data is cast to a list.
        # We don't need to check if it's a list of Services or anything else: Pydantic does that for us.
        if ("components" in result) and (not isinstance(result["components"], list)):
            result["components"] = [result["components"]]
        return result

    # Is there a better way to do this? calculated_async_flag is exposed to the user right now.
    # Maybe I could just make this a 'private' field, like '_calc_async'
    @model_validator(mode="after")
    def calculate_async_flag(self):
        self.calculated_async_flag = all([service.asynchronous for service in self.components])
        return self

    async def run_component(self, ctx: Context, pipeline: Pipeline) -> None:
        """
        Method for running this service group. Catches runtime exceptions and logs them.
        It doesn't include extra handlers execution, start condition checking or error handling - pure execution only.
        Executes components inside the group based on its `asynchronous` property.
        Collects information about their execution state - group is finished successfully
        only if all components in it finished successfully.

        :param ctx: Current dialog context.
        :param pipeline: The current pipeline.

        :param ctx: Current dialog context.
        :param pipeline: The current pipeline.
        """
        if self.asynchronous:
            service_futures = [service(ctx, pipeline) for service in self.components]
            for service, future in zip(self.components, await asyncio.gather(*service_futures, return_exceptions=True)):
                service_result = future
                if service.asynchronous and isinstance(service_result, Awaitable):
                    await service_result
                elif isinstance(service_result, asyncio.TimeoutError):
                    logger.warning(f"{type(service).__name__} '{service.name}' timed out!")

        else:
            for service in self.components:
                service_result = await service(ctx, pipeline)
                if service.asynchronous and isinstance(service_result, Awaitable):
                    await service_result

        failed = any([service.get_state(ctx) == ComponentExecutionState.FAILED for service in self.components])
        self._set_state(ctx, ComponentExecutionState.FAILED if failed else ComponentExecutionState.FINISHED)

    def log_optimization_warnings(self):
        """
        Method for logging service group optimization warnings for all this groups inner components.
        (NOT this group itself!).
        Warnings are basically messages,
        that indicate service group inefficiency or explicitly defined parameters mismatch.
        These are cases for warnings issuing:

        - Service can be asynchronous, however is marked synchronous explicitly.
        - Service is not asynchronous, however has a timeout defined.
        - Group is not marked synchronous explicitly and contains both synchronous and asynchronous components.

        :return: `None`
        """
        for service in self.components:
            if isinstance(service, Service):
                if (
                    service.calculated_async_flag
                    and service.requested_async_flag is not None
                    and not service.requested_async_flag
                ):
                    logger.warning(f"Service '{service.name}' could be asynchronous!")
                if not service.asynchronous and service.timeout is not None:
                    logger.warning(f"Timeout can not be applied for Service '{service.name}': it's not asynchronous!")
            else:
                if not service.calculated_async_flag:
                    if service.requested_async_flag is None and any(
                        [sub_service.asynchronous for sub_service in service.components]
                    ):
                        logger.warning(
                            f"ServiceGroup '{service.name}' contains both sync and async services, "
                            "it should be split or marked as synchronous explicitly!",
                        )
                service.log_optimization_warnings()

    def add_extra_handler(
        self,
        global_extra_handler_type: GlobalExtraHandlerType,
        extra_handler: ExtraHandlerFunction,
        condition: ExtraHandlerConditionFunction = lambda _: False,
    ):
        """
        Method for adding a global extra handler to this group.
        Adds extra handler to itself and propagates it to all inner components.
        Uses a special condition function to determine whether to add extra handler to any particular inner component.
        Condition checks components path to be in whitelist (if defined) and not to be in blacklist (if defined).

        :param global_extra_handler_type: A type of extra handler to add.
        :param extra_handler: A `ExtraHandlerFunction` to add as an extra handler.
        :type extra_handler: :py:data:`~.ExtraHandlerFunction`
        :param condition: A condition function.
        :return: `None`
        """
        super().add_extra_handler(global_extra_handler_type, extra_handler)
        for service in self.components:
            if not condition(service.path):
                continue
            if isinstance(service, ServiceGroup):
                service.add_extra_handler(global_extra_handler_type, extra_handler, condition)
            else:
                service.add_extra_handler(global_extra_handler_type, extra_handler)

    @property
    def info_dict(self) -> dict:
        """
        See `Component.info_dict` property.
        Adds `services` key to base info dictionary.
        """
        representation = super(ServiceGroup, self).info_dict
        representation.update({"services": [service.info_dict for service in self.components]})
        return representation
