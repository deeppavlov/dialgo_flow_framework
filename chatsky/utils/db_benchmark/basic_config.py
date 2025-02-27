"""
Basic Config
------------
This module contains basic benchmark configurations.

It defines a simple configurations class (:py:class:`~.BasicBenchmarkConfig`)
as well as a set of configurations that covers different dialogs a user might have and some edge-cases
(:py:data:`~.basic_configurations`).
"""

from typing import Tuple, Optional
import string
import random

from humanize import naturalsize
from pympler import asizeof

from chatsky.core import Message, Context, AbsoluteNodeLabel
from chatsky.context_storages import MemoryContextStorage
from chatsky.utils.db_benchmark.benchmark import BenchmarkConfig


def get_dict(dimensions: Tuple[int, ...]):
    """
    Return misc dictionary build in `dimensions` dimensions.

    :param dimensions:
        Dimensions of the dictionary.
        Each element of the dimensions tuple is the number of keys on the corresponding level of the dictionary.
        The last element of the dimensions tuple is the length of the string values of the dict.

        e.g. dimensions=(1, 2) returns a dictionary with 1 key that points to a string of len 2.
        whereas dimensions=(1, 2, 3) returns a dictionary with 1 key that points to a dictionary
        with 2 keys each of which points to a string of len 3.

        So, the len of dimensions is the depth of the dictionary, while its values are
        the width of the dictionary at each level.
    """

    def _get_dict(dimensions: Tuple[int, ...]):
        if len(dimensions) < 2:
            # get a random string of length dimensions[0]
            return "".join(random.choice(string.printable) for _ in range(dimensions[0]))
        return {str(i): _get_dict(dimensions[1:]) for i in range(dimensions[0])}

    if len(dimensions) > 1:
        return _get_dict(dimensions)
    elif len(dimensions) == 1:
        return _get_dict((dimensions[0], 0))
    else:
        return _get_dict((0, 0))


def get_message(message_dimensions: Tuple[int, ...]):
    """
    Return message with a non-empty misc field.

    :param message_dimensions: Dimensions of the misc field of the message. See :py:func:`~.get_dict`.
    """
    return Message(misc=get_dict(message_dimensions))


async def get_context(
    db,
    dialog_len: int,
    message_dimensions: Tuple[int, ...],
    misc_dimensions: Tuple[int, ...],
) -> Context:
    """
    Return context with a non-empty misc, labels, requests, responses fields.

    :param dialog_len: Number of labels, requests and responses.
    :param message_dimensions:
        A parameter used to generate messages for requests and responses. See :py:func:`~.get_message`.
    :param misc_dimensions:
        A parameter used to generate misc field. See :py:func:`~.get_dict`.
    """
    ctx = await Context.connected(db, start_label=("flow", "node"))
    ctx.current_turn_id = -1
    for i in range(dialog_len):
        ctx.current_turn_id += 1
        ctx.labels[ctx.current_turn_id] = AbsoluteNodeLabel(flow_name=f"flow_{i}", node_name=f"node_{i}")
        ctx.requests[ctx.current_turn_id] = get_message(message_dimensions)
        ctx.responses[ctx.current_turn_id] = get_message(message_dimensions)
    ctx.misc.update(get_dict(misc_dimensions))

    return ctx


class BasicBenchmarkConfig(BenchmarkConfig, frozen=True):
    """
    A simple benchmark configuration that generates contexts using two parameters:

    - `message_dimensions` -- to configure the way messages are generated.
    - `misc_dimensions` -- to configure size of context's misc field.

    Dialog length is configured using `from_dialog_len`, `to_dialog_len`, `step_dialog_len`.
    """

    context_num: int = 1
    """
    Number of times the contexts will be benchmarked.
    Increasing this number decreases standard error of the mean for benchmarked data.
    """
    from_dialog_len: int = 50
    """Starting dialog len of a context."""
    to_dialog_len: int = 75
    """
    Final dialog len of a context.
    :py:meth:`~.BasicBenchmarkConfig.context_updater` will return contexts
    until their dialog len is less then `to_dialog_len`.
    """
    step_dialog_len: int = 1
    """
    Increment step for dialog len.
    :py:meth:`~.BasicBenchmarkConfig.context_updater` will return contexts
    increasing dialog len by `step_dialog_len`.
    """
    message_dimensions: Tuple[int, ...] = (10, 10)
    """
    Dimensions of misc dictionaries inside messages.
    See :py:func:`~.get_message`.
    """
    misc_dimensions: Tuple[int, ...] = (10, 10)
    """
    Dimensions of misc dictionary.
    See :py:func:`~.get_dict`.
    """

    async def get_context(self, db) -> Context:
        """
        Return context with `from_dialog_len`, `message_dimensions`, `misc_dimensions`.

        Wraps :py:func:`~.get_context`.
        """
        return await get_context(db, self.from_dialog_len, self.message_dimensions, self.misc_dimensions)

    async def info(self):
        """
        Return fields of this instance and sizes of objects defined by this config.

        :return:
            A dictionary with two keys.
            Key "params" stores fields of this configuration.
            Key "sizes" stores string representation of following values:

                - "starting_context_size" -- size of a context with `from_dialog_len`.
                - "final_context_size" -- size of a context with `to_dialog_len`.
                  A context of this size will never actually be benchmarked.
                - "misc_size" -- size of a misc field of a context.
                - "message_size" -- size of a misc field of a message.
        """

        def remove_db_from_context(ctx: Context):
            ctx._storage = None
            ctx.requests._storage = None
            ctx.responses._storage = None
            ctx.labels._storage = None

        starting_context = await get_context(
            MemoryContextStorage(), self.from_dialog_len, self.message_dimensions, self.misc_dimensions
        )
        final_contex = await get_context(
            MemoryContextStorage(), self.to_dialog_len, self.message_dimensions, self.misc_dimensions
        )
        remove_db_from_context(starting_context)
        remove_db_from_context(final_contex)
        return {
            "params": self.model_dump(),
            "sizes": {
                "starting_context_size": naturalsize(
                    asizeof.asizeof(starting_context.model_dump(mode="python")), gnu=True
                ),
                "final_context_size": naturalsize(asizeof.asizeof(final_contex.model_dump(mode="python")), gnu=True),
                "misc_size": naturalsize(asizeof.asizeof(get_dict(self.misc_dimensions)), gnu=True),
                "message_size": naturalsize(asizeof.asizeof(get_message(self.message_dimensions)), gnu=True),
            },
        }

    async def context_updater(self, context: Context) -> Optional[Context]:
        """
        Update context to have `step_dialog_len` more labels, requests and responses,
        unless such dialog len would be equal to `to_dialog_len` or exceed than it,
        in which case None is returned.
        """
        start_len = len(context.labels)
        if start_len + self.step_dialog_len < self.to_dialog_len:
            for i in range(start_len, start_len + self.step_dialog_len):
                context.current_turn_id += 1
                context.labels[context.current_turn_id] = AbsoluteNodeLabel(
                    flow_name=f"flow_{i}", node_name=f"node_{i}"
                )
                context.requests[context.current_turn_id] = get_message(self.message_dimensions)
                context.responses[context.current_turn_id] = get_message(self.message_dimensions)
            return context
        else:
            return None


basic_configurations = {
    "large-misc": BasicBenchmarkConfig(
        from_dialog_len=1,
        to_dialog_len=50,
        message_dimensions=(3, 5, 6, 5, 3),
        misc_dimensions=(2, 4, 3, 8, 100),
    ),
    "short-messages": BasicBenchmarkConfig(
        message_dimensions=(2, 30),
        misc_dimensions=(0, 0),
    ),
    "default": BasicBenchmarkConfig(),
    "large-misc-long-dialog": BasicBenchmarkConfig(
        from_dialog_len=500,
        to_dialog_len=510,
        message_dimensions=(3, 5, 6, 5, 3),
        misc_dimensions=(2, 4, 3, 8, 100),
    ),
    "very-long-dialog-len": BasicBenchmarkConfig(
        from_dialog_len=10000,
        to_dialog_len=10010,
    ),
    "very-long-message-len": BasicBenchmarkConfig(
        from_dialog_len=1,
        to_dialog_len=3,
        message_dimensions=(10000, 1),
    ),
    "very-long-misc-len": BasicBenchmarkConfig(
        from_dialog_len=1,
        to_dialog_len=3,
        misc_dimensions=(10000, 1),
    ),
}
"""
Configuration that covers many dialog cases (as well as some edge-cases).

:meta hide-value:
"""
