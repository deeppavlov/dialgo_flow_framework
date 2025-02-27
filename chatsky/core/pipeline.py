"""
Pipeline
--------
Pipeline is the main element of the Chatsky framework.

Pipeline is responsible for managing and executing the various components
(:py:class:`~chatsky.core.service.component.PipelineComponent`)
including :py:class:`.Actor`.
"""

from __future__ import annotations
import asyncio
import logging
from functools import cached_property
from typing import Union, List, Dict, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator, computed_field

from chatsky.core.script import Script
from chatsky.core.context import Context
from chatsky.core.message import Message

from chatsky.context_storages import DBContextStorage, MemoryContextStorage
from chatsky.messengers.console import CLIMessengerInterface
from chatsky.messengers.common import MessengerInterface
from chatsky.slots.slots import GroupSlot
from chatsky.core.service.group import ServiceGroup, ServiceGroupInitTypes
from chatsky.core.service.extra import ComponentExtraHandlerInitTypes, BeforeHandler, AfterHandler
from .service import Service
from .utils import finalize_service_group, initialize_service_states
from chatsky.core.service.actor import Actor
from chatsky.core.node_label import AbsoluteNodeLabel, AbsoluteNodeLabelInitTypes
from chatsky.core.script_parsing import JSONImporter, Path

if TYPE_CHECKING:
    from chatsky.ml.models.base_model import ExtrasBaseAPIModel
    from chatsky.llm.llm_api import LLM_API

logger = logging.getLogger(__name__)


class PipelineServiceGroup(ServiceGroup):
    """A service group that allows actor inside."""

    components: List[Union[Actor, Service, ServiceGroup]]


class Pipeline(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """
    Class that automates service execution and creates service pipeline.
    """

    pre_services: ServiceGroup = Field(default_factory=list, validate_default=True)
    """
    :py:class:`~.ServiceGroup` that will be executed before Actor.
    """
    post_services: ServiceGroup = Field(default_factory=list, validate_default=True)
    """
    :py:class:`~.ServiceGroup` that will be executed after :py:class:`~.Actor`.
    """
    script: Script
    """
    (required) A :py:class:`~.Script` instance (object or dict).
    """
    start_label: AbsoluteNodeLabel
    """
    (required) The first node of every context.
    """
    fallback_label: AbsoluteNodeLabel
    """
    Node which will is used if :py:class:`Actor` cannot find the next node.

    This most commonly happens when there are not suitable transitions.

    Defaults to :py:attr:`start_label`.
    """
    default_priority: float = 1.0
    """
    Default priority value for :py:class:`~chatsky.core.transition.Transition`.

    Defaults to ``1.0``.
    """
    slots: GroupSlot = Field(default_factory=GroupSlot)
    """
    Slots configuration.
    """
    models: Dict[str, Union[LLM_API, ExtrasBaseAPIModel]] = Field(default_factory=dict)
    """
    LLM models to be made available in custom functions.
    """
    messenger_interface: MessengerInterface = Field(default_factory=CLIMessengerInterface)
    """
    A `MessengerInterface` instance for this pipeline.

    It handles connections to interfaces that provide user requests and accept bot responses.
    """
    context_storage: DBContextStorage = Field(default_factory=MemoryContextStorage)
    """
    A :py:class:`~.DBContextStorage` instance for this pipeline or
    a dict to store dialog :py:class:`~.Context`.
    """
    before_handler: BeforeHandler = Field(default_factory=list, validate_default=True)
    """
    :py:class:`~.BeforeHandler` to add to the pipeline service.
    """
    after_handler: AfterHandler = Field(default_factory=list, validate_default=True)
    """
    :py:class:`~.AfterHandler` to add to the pipeline service.
    """
    timeout: Optional[float] = None
    """
    Timeout to add to pipeline root service group.
    """
    parallelize_processing: bool = False
    """
    This flag determines whether or not the functions
    defined in the ``PRE_RESPONSE_PROCESSING`` and ``PRE_TRANSITIONS_PROCESSING`` sections
    of the script should be parallelized over respective groups.
    """

    def __init__(
        self,
        script: Union[Script, dict],
        start_label: AbsoluteNodeLabelInitTypes,
        fallback_label: AbsoluteNodeLabelInitTypes = None,
        *,
        default_priority: float = None,
        slots: GroupSlot = None,
        models: dict = None,
        messenger_interface: MessengerInterface = None,
        context_storage: DBContextStorage = None,
        pre_services: ServiceGroupInitTypes = None,
        post_services: ServiceGroupInitTypes = None,
        before_handler: ComponentExtraHandlerInitTypes = None,
        after_handler: ComponentExtraHandlerInitTypes = None,
        timeout: float = None,
        parallelize_processing: bool = None,
    ):
        if fallback_label is None:
            fallback_label = start_label
        init_dict = {
            "script": script,
            "start_label": start_label,
            "fallback_label": fallback_label,
            "default_priority": default_priority,
            "slots": slots,
            "models": models,
            "messenger_interface": messenger_interface,
            "context_storage": context_storage,
            "pre_services": pre_services,
            "post_services": post_services,
            "before_handler": before_handler,
            "after_handler": after_handler,
            "timeout": timeout,
            "parallelize_processing": parallelize_processing,
        }
        empty_fields = set()
        for k, v in init_dict.items():
            if k not in self.model_fields:
                raise NotImplementedError("Init method contains a field not in model fields.")
            if v is None:
                empty_fields.add(k)
        for field in empty_fields:
            del init_dict[field]
        super().__init__(**init_dict)
        self.services_pipeline  # cache services

    @classmethod
    def from_file(
        cls,
        file: Union[str, Path],
        custom_dir: Union[str, Path] = "custom",
        **overrides,
    ) -> "Pipeline":
        """
        Create Pipeline by importing it from a file.
        A file (json or yaml) should contain a dictionary with keys being a subset of pipeline init parameters.

        See :py:meth:`.JSONImporter.import_pipeline_file` for more information.

        :param file: Path to a file containing pipeline init parameters.
        :param custom_dir: Path to a directory containing custom code.
            Defaults to "./custom".
            If ``file`` does not use custom code, this parameter will not have any effect.
        :param overrides: You can pass init parameters to override those imported from the ``file``.
        """
        pipeline = JSONImporter(custom_dir=custom_dir).import_pipeline_file(file)

        pipeline.update(overrides)

        return cls(**pipeline)

    @computed_field
    @cached_property
    def actor(self) -> Actor:
        """An actor instance of the pipeline."""
        return Actor()

    @computed_field
    @cached_property
    def services_pipeline(self) -> PipelineServiceGroup:
        """
        A group containing :py:attr:`.Pipeline.pre_services`, :py:class:`~.Actor`
        and :py:attr:`.Pipeline.post_services`.
        It has :py:attr:`.Pipeline.before_handler` and :py:attr:`.Pipeline.after_handler` applied to it.
        """
        components = [self.pre_services, self.actor, self.post_services]
        self.pre_services.name = "pre"
        self.post_services.name = "post"
        services_pipeline = PipelineServiceGroup(
            components=components,
            before_handler=self.before_handler,
            after_handler=self.after_handler,
            timeout=self.timeout,
        )
        services_pipeline.name = ""
        services_pipeline.path = ""

        finalize_service_group(services_pipeline, path=services_pipeline.path)

        return services_pipeline

    @model_validator(mode="after")
    def validate_start_label(self):
        """Validate :py:attr:`start_label` is in :py:attr:`script`."""
        if self.script.get_node(self.start_label) is None:
            raise ValueError(f"Unknown start_label={self.start_label}")
        return self

    @model_validator(mode="after")
    def validate_fallback_label(self):
        """Validate :py:attr:`fallback_label` is in :py:attr:`script`."""
        if self.script.get_node(self.fallback_label) is None:
            raise ValueError(f"Unknown fallback_label={self.fallback_label}")
        return self

    async def _run_pipeline(
        self, request: Message, ctx_id: Optional[str] = None, update_ctx_misc: Optional[dict] = None
    ) -> Context:
        """
        Method that should be invoked on user input.
        This method has the same signature as :py:class:`~chatsky.core.service.types.PipelineRunnerFunction`.

        This method does:

        1. Retrieve from :py:attr:`context_storage` or initialize context ``ctx_id``.
        2. Update :py:attr:`.Context.misc` with ``update_ctx_misc``.
        3. Set up :py:attr:`.Context.framework_data` fields.
        4. Add ``request`` to the context.
        5. Execute :py:attr:`services_pipeline`.
           This includes :py:class:`.Actor` (read :py:meth:`.Actor.run_component` for more information).
        6. Save context in the :py:attr:`context_storage`.

        :return: Modified context ``ctx_id``.
        """
        logger.info(f"Running pipeline for context {ctx_id}.")
        logger.debug(f"Received request: {request}.")
        ctx = await Context.connected(self.context_storage, self.start_label, ctx_id)

        if update_ctx_misc is not None:
            ctx.misc.update(update_ctx_misc)

        if self.slots is not None:
            ctx.framework_data.slot_manager.set_root_slot(self.slots)

        ctx.framework_data.pipeline = self
        initialize_service_states(ctx, self.services_pipeline)

        ctx.current_turn_id = ctx.current_turn_id + 1

        ctx.requests[ctx.current_turn_id] = request
        await self.services_pipeline(ctx)

        ctx.framework_data.service_states.clear()
        ctx.framework_data.pipeline = None

        await ctx.store()

        return ctx

    def run(self):
        """
        Method that starts a pipeline and connects to :py:attr:`messenger_interface`.
        It also connects to the :py:attr:`context_storage` (if it's not already connected).

        It passes :py:meth:`_run_pipeline` to :py:attr:`messenger_interface` as a callback,
        so every time user request is received, :py:meth:`_run_pipeline` will be called.

        This method can be both blocking and non-blocking. It depends on current :py:attr:`messenger_interface` nature.
        Message interfaces that run in a loop block current thread.
        """
        if not self.context_storage.connected:
            asyncio.run(self.context_storage.connect())
        logger.info("Pipeline is accepting requests.")
        asyncio.run(self.messenger_interface.connect(self._run_pipeline))

    def __call__(
        self, request: Message, ctx_id: Optional[str] = None, update_ctx_misc: Optional[dict] = None
    ) -> Context:
        """
        Method that executes pipeline once.
        Basically, it is a shortcut for :py:meth:`_run_pipeline`.
        NB! When pipeline is executed this way, :py:attr:`messenger_interface` won't be initiated nor connected.
        Still, it connects to the :py:attr:`context_storage` (if it's not already connected) to avoid sync issues.

        This method has the same signature as :py:class:`~chatsky.core.service.types.PipelineRunnerFunction`.
        """
        if not self.context_storage.connected:
            asyncio.run(self.context_storage.connect())
        return asyncio.run(self._run_pipeline(request, ctx_id, update_ctx_misc))
