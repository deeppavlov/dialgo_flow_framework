"""
Actor
-----
Actor is a component of :py:class:`.Pipeline`, that processes the :py:class:`.Script`.

It is responsible for determining the next node and getting response from it.

The actor acts as a bridge between the user's input and the dialog graph,
making sure that the conversation follows the expected flow.

More details on the processing can be found in the documentation for
:py:meth:`Actor.run_component`.
"""

from __future__ import annotations
import logging
import asyncio
from typing import Dict

from chatsky.core.service.component import PipelineComponent
from chatsky.core.transition import get_next_label
from chatsky.core.message import Message

from chatsky.core.context import Context
from chatsky.core.script_function import BaseProcessing

logger = logging.getLogger(__name__)


class Actor(PipelineComponent):
    """
    The class which is used to process :py:class:`~chatsky.core.context.Context`
    according to the :py:class:`~chatsky.core.script.Script`.
    """

    @property
    def computed_name(self) -> str:
        """
        "actor"
        """
        return "actor"

    async def run_component(self, ctx: Context) -> None:
        """
        Process the context in the following way:

        1. Run pre-transition of the :py:attr:`.Context.current_node`.
        2. Determine and save the next node based on :py:attr:`~chatsky.core.script.Node.transitions`
           of the :py:attr:`.Context.current_node`.
        3. Run pre-response of the :py:attr:`.Context.current_node`.
        4. Determine and save the response of the :py:attr:`.Context.current_node`
        """
        next_label = ctx.pipeline.fallback_label

        try:
            ctx.framework_data.current_node = ctx.pipeline.script.get_inherited_node(ctx.last_label)

            logger.debug("Running pre_transition")
            await self._run_processing(ctx.current_node.pre_transition, ctx)

            logger.debug("Running transitions")

            destination_result, transition = await get_next_label(
                ctx, ctx.current_node.transitions, ctx.pipeline.default_priority
            )
            if destination_result is not None:
                next_label = destination_result
            ctx.framework_data.transition = transition
        except Exception as exc:
            logger.exception("Exception occurred during transition processing.", exc_info=exc)

        logger.debug(f"Next label: {next_label}")

        ctx.labels[ctx.current_turn_id] = next_label

        response = Message()

        try:
            ctx.framework_data.current_node = ctx.pipeline.script.get_inherited_node(next_label)

            logger.debug("Running pre_response")
            await self._run_processing(ctx.current_node.pre_response, ctx)

            node_response = ctx.current_node.response
            if node_response is not None:
                response_result = await node_response.wrapped_call(ctx)
                if isinstance(response_result, Message):
                    response = response_result
                    logger.debug(f"Produced response {response}.")
                else:
                    logger.debug("Response was not produced.")
            else:
                logger.debug("Node has empty response.")
        except Exception as exc:
            logger.exception("Exception occurred during response processing.", exc_info=exc)

        ctx.responses[ctx.current_turn_id] = response

    @staticmethod
    async def _run_processing_parallel(processing: Dict[str, BaseProcessing], ctx: Context) -> None:
        """
        Execute :py:class:`.BaseProcessing` functions simultaneously, independent of the order.

        Picked depending on the value of the :py:class:`.Pipeline`'s `parallelize_processing` flag.
        """
        await asyncio.gather(
            *[func.wrapped_call(ctx, info=f"processing_name={name!r}") for name, func in processing.items()]
        )

    @staticmethod
    async def _run_processing_sequential(processing: Dict[str, BaseProcessing], ctx: Context) -> None:
        """
        Execute :py:class:`.BaseProcessing` functions in-order.

        Picked depending on the value of the :py:class:`.Pipeline`'s `parallelize_processing` flag.
        """
        for name, func in processing.items():
            await func.wrapped_call(ctx, info=f"processing_name={name!r}")

    @staticmethod
    async def _run_processing(processing: Dict[str, BaseProcessing], ctx: Context) -> None:
        """
        Run :py:class:`.BaseProcessing` functions.

        The execution order depends on the value of the :py:class:`.Pipeline`'s
        `parallelize_processing` flag.
        """
        if ctx.pipeline.parallelize_processing:
            await Actor._run_processing_parallel(processing, ctx)
        else:
            await Actor._run_processing_sequential(processing, ctx)
