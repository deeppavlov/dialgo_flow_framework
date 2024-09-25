"""
Component
---------
The Component module defines a :py:class:`.PipelineComponent` class.

This is a base class for pipeline processing and is responsible for performing a specific task.
"""

from __future__ import annotations

import logging
import abc
import asyncio
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator

from chatsky.core.service.extra import BeforeHandler, AfterHandler
from chatsky.core.script_function import AnyCondition
from chatsky.core.service.types import (
    ComponentExecutionState,
    GlobalExtraHandlerType,
    ExtraHandlerFunction,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from chatsky.core.context import Context


class PipelineComponent(abc.ABC, BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """
    Base class for a single task processed by :py:class:`.Pipeline`.
    """

    before_handler: BeforeHandler = Field(default_factory=BeforeHandler)
    """
    :py:class:`~.BeforeHandler`, associated with this component.
    """
    after_handler: AfterHandler = Field(default_factory=AfterHandler)
    """
    :py:class:`~.AfterHandler`, associated with this component.
    """
    timeout: Optional[float] = None
    """
    Maximum component execution time (in seconds),
    if it exceeds this time, it is interrupted.
    """
    asynchronous: bool = False
    """
    Optional flag that indicates whether this component
    should be executed asynchronously with adjacent async components.
    """
    start_condition: AnyCondition = Field(default=True, validate_default=True)
    """
    :py:data:`~.AnyCondition` that is invoked before each component execution;
    component is executed only if it returns ``True``.
    """
    name: Optional[str] = None
    """
    Component name (should be unique in a single :py:class:`~chatsky.core.service.group.ServiceGroup`),
    should not be blank or contain the ``.`` character.
    """
    path: Optional[str] = None
    """
    Separated by dots path to component, is universally unique.
    """

    @field_validator("name")
    @classmethod
    def __pipeline_component_name_validator__(cls, name: str):
        """
        Validate this component's name.

        :raises ValueError: If component's name is blank or if it contains dots.
        """
        if name is not None:
            if name == "":
                raise ValueError("Name cannot be blank.")
            if "." in name:
                raise ValueError(f"Name cannot contain '.': {name!r}.")

        return name

    def _set_state(self, ctx: Context, value: ComponentExecutionState):
        """
        Method for component runtime state setting, state is preserved in :py:attr:`.Context.framework_data`.

        :param ctx: :py:class:`~.Context` to keep state in.
        :param value: State to set.
        """
        ctx.framework_data.service_states[self.path].execution_status = value

    def get_state(self, ctx: Context) -> ComponentExecutionState:
        """
        Method for component runtime state getting, state is preserved in :py:attr:`.Context.framework_data`.

        :param ctx: :py:class:`~.Context` to get state from.
        :return: :py:class:`.ComponentExecutionState` of this service.
        """
        return ctx.framework_data.service_states[self.path].execution_status

    @abc.abstractmethod
    async def run_component(self, ctx: Context) -> Optional[ComponentExecutionState]:
        """
        Run this component.

        :param ctx: Current dialog :py:class:`~.Context`.
        """
        raise NotImplementedError

    @property
    def computed_name(self) -> str:
        """
        Default name that is used if :py:attr:`~.PipelineComponent.name` is not defined.
        In case two components in a :py:class:`~chatsky.core.service.group.ServiceGroup` have the same
        :py:attr:`.computed_name` an incrementing number is appended to the name.
        """
        return "noname_service"

    async def _run(self, ctx: Context) -> None:
        """
        A method for running a pipeline component. Executes extra handlers before and after execution,
        launches :py:meth:`.run_component` method. This method is run after the component's timeout is set (if needed).

        :param ctx: Current dialog :py:class:`~.Context`.
        """

        async def _inner_run():
            if await self.start_condition(ctx):
                await self.before_handler(ctx, self)

                self._set_state(ctx, ComponentExecutionState.RUNNING)
                result = await self.run_component(ctx)
                if isinstance(result, ComponentExecutionState):
                    self._set_state(ctx, result)
                else:
                    self._set_state(ctx, ComponentExecutionState.FINISHED)

                await self.after_handler(ctx, self)
            else:
                self._set_state(ctx, ComponentExecutionState.NOT_RUN)

        try:
            await asyncio.wait_for(_inner_run(), timeout=self.timeout)
        except Exception as exc:
            self._set_state(ctx, ComponentExecutionState.FAILED)
            logger.error(f"Service '{self.name}' execution failed!", exc_info=exc)
        finally:
            ctx.framework_data.service_states[self.path].finished_event.set()

    async def __call__(self, ctx: Context) -> None:
        """
        A method for calling pipeline components.
        It sets up timeout and executes it using :py:meth:`_run` method.

        :param ctx: Current dialog :py:class:`~.Context`.
        :return: ``None``
        """
        await self._run(ctx)

    def add_extra_handler(self, global_extra_handler_type: GlobalExtraHandlerType, extra_handler: ExtraHandlerFunction):
        """
        Method for adding a global extra handler to this particular component.

        :param global_extra_handler_type: A type of extra handler to add.
        :param extra_handler: A :py:class:`~.GlobalExtraHandlerType` to add to the component as an extra handler.
        :type extra_handler: :py:data:`~.ExtraHandlerFunction`
        :return: `None`
        """
        target = (
            self.before_handler if global_extra_handler_type is GlobalExtraHandlerType.BEFORE else self.after_handler
        )
        target.functions.append(extra_handler)
