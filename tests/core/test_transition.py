from typing import Union

import pytest

from chatsky.core import Transition as Tr, BaseDestination, BaseCondition, BasePriority, Context
from chatsky.core.transition import get_next_label, AbsoluteNodeLabel
from chatsky.core.node_label import NodeLabelInitTypes


class FaultyDestination(BaseDestination):
    async def call(self, ctx: Context) -> NodeLabelInitTypes:
        raise RuntimeError()


class FaultyCondition(BaseCondition):
    async def call(self, ctx: Context) -> bool:
        raise RuntimeError()


class FaultyPriority(BasePriority):
    async def call(self, ctx: Context) -> Union[float, bool, None]:
        raise RuntimeError()


class TruePriority(BasePriority):
    async def call(self, ctx: Context) -> Union[float, bool, None]:
        return True


class FalsePriority(BasePriority):
    async def call(self, ctx: Context) -> Union[float, bool, None]:
        return False


@pytest.mark.parametrize(
    "transitions,default_priority,result",
    [
        ([Tr(dst=("service", "start"))], 0, ("service", "start")),
        ([Tr(dst="node1")], 0, ("flow", "node1")),
        ([Tr(dst="node1"), Tr(dst="node2")], 0, ("flow", "node1")),
        ([Tr(dst="node1"), Tr(dst="node2", priority=1)], 0, ("flow", "node2")),
        ([Tr(dst="node1"), Tr(dst="node2", priority=1)], 2, ("flow", "node1")),
        ([Tr(dst="node1", cnd=False), Tr(dst="node2")], 0, ("flow", "node2")),
        ([Tr(dst="node1", cnd=False), Tr(dst="node2", cnd=False)], 0, None),
        ([Tr(dst="non_existent")], 0, None),
        ([Tr(dst=FaultyDestination())], 0, None),
        ([Tr(dst="node1", priority=FaultyPriority())], 0, None),
        ([Tr(dst="node1", cnd=FaultyCondition())], 0, None),
        ([Tr(dst="node1", priority=FalsePriority())], 0, None),
        ([Tr(dst="node1", priority=TruePriority()), Tr(dst="node2", priority=1)], 0, ("flow", "node2")),
        ([Tr(dst="node1", priority=TruePriority()), Tr(dst="node2", priority=1)], 2, ("flow", "node1")),
        ([Tr(dst="node1", priority=1), Tr(dst="node2", priority=2), Tr(dst="node3", priority=3)], 0, ("flow", "node3")),
    ],
)
async def test_get_next_label(context_factory, transitions, default_priority, result):
    ctx = context_factory(start_label=("flow", "node1"))

    next_label, _ = await get_next_label(ctx, transitions, default_priority)

    assert next_label == (AbsoluteNodeLabel.model_validate(result) if result is not None else None)


@pytest.mark.parametrize(
    "transitions,result",
    [
        ([Tr(dst=("service", "start"))], Tr(dst=("service", "start"))),
        (
            [Tr(dst="node1", priority=1), Tr(dst="node2", priority=2), Tr(dst="node3", priority=3)],
            Tr(dst="node3", priority=3),
        ),
        ([Tr(dst="node1", cnd=False), Tr(dst="node2", cnd=False)], None),
    ],
)
async def test_transition_for_next_label(context_factory, transitions, result):
    ctx = context_factory(start_label=("flow", "node1"))

    _, transition = await get_next_label(ctx, transitions, default_priority=0)

    assert transition == result
