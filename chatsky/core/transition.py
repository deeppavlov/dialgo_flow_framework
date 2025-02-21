"""
Transition
----------
This module defines a transition class that is used to
specify conditions and destinations for transitions to nodes.
"""

from __future__ import annotations

from typing import Union, List, TYPE_CHECKING, Optional, Tuple
import logging
import asyncio

from pydantic import BaseModel, Field

from chatsky.core.script_function import AnyCondition, AnyDestination, AnyPriority
from chatsky.core.script_function import BaseCondition, BaseDestination, BasePriority
from chatsky.core.node_label import AbsoluteNodeLabel, NodeLabelInitTypes

if TYPE_CHECKING:
    from chatsky.core.context import Context


logger = logging.getLogger(__name__)


class Transition(BaseModel):
    """
    A basic class for a transition to a node.
    """

    cnd: AnyCondition = Field(default=True, validate_default=True)
    """A condition that determines if transition is allowed to happen."""
    dst: AnyDestination
    """Destination node of the transition."""
    priority: AnyPriority = Field(default=None, validate_default=True)
    """Priority of the transition. Higher priority transitions are resolved first."""

    def __init__(
        self,
        *,
        cnd: Union[bool, BaseCondition] = True,
        dst: Union[NodeLabelInitTypes, BaseDestination],
        priority: Union[Optional[float], BasePriority] = None,
    ):
        super().__init__(cnd=cnd, dst=dst, priority=priority)


async def get_next_label(
    ctx: Context, transitions: List[Transition], default_priority: float
) -> Tuple[Optional[AbsoluteNodeLabel], Optional[Transition]]:
    """
    Determine the next node based on ``transitions`` and ``ctx``.

    The process is as follows:

    1. Condition result is calculated for every transition.
    2. Transitions are filtered by the calculated condition.
    3. Priority result is calculated for every transition that is left.
       ``default_priority`` is used for priorities that return ``True`` or ``None``
       as per :py:class:`.BasePriority`.
       Those that return ``False`` are filtered out.
    4. Destination result is calculated for every transition that is left.
    5. The highest priority transition is chosen.
       If there are multiple transition of the higher priority,
       choose the first one of that priority in the ``transitions`` list.
       Order of ``transitions`` is as follows:
       ``node transitions, local transitions, global transitions``.

    If at any point any :py:class:`.BaseCondition`, :py:class:`.BaseDestination` or :py:class:`.BasePriority`
    produces an exception, the corresponding transition is filtered out.

    :return: Label of the next node or ``None`` if no transition is left by the end of the process and
    the transition that leads to the next node.
    """
    
    filtered_transitions: List[Transition] = transitions.copy()
    condition_results = await asyncio.gather(*[transition.cnd.wrapped_call(ctx) for transition in filtered_transitions])

    filtered_transitions = [
        transition for transition, condition in zip(filtered_transitions, condition_results) if condition is True
    ]

    priority_results = await asyncio.gather(
        *[transition.priority.wrapped_call(ctx) for transition in filtered_transitions]
    )

    transitions_with_priorities: List[Tuple[Transition, float]] = [
        (transition, (priority_result if isinstance(priority_result, float) else default_priority))
        for transition, priority_result in zip(filtered_transitions, priority_results)
        if (priority_result is True or priority_result is None or isinstance(priority_result, float))
    ]
    logger.debug(f"Possible transitions: {transitions_with_priorities!r}")

    transitions_with_priorities = sorted(transitions_with_priorities, key=lambda x: x[1], reverse=True)

    destination_results: List[Union[AbsoluteNodeLabel, Exception]] = await asyncio.gather(
        *[transition.dst.wrapped_call(ctx) for transition, _ in transitions_with_priorities]
    )

    for destination, transition in zip(destination_results, transitions_with_priorities):
        if isinstance(destination, AbsoluteNodeLabel):
            return destination, transition[0]
    return (None, None)
