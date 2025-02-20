"""
Context Utils
-------------
This module defines classes used by :py:class:`~chatsky.core.context.Context`.

The most important ones here are :py:class:`FrameworkData` and :py:class:`ContextMainInfo`
that define all non-turn related data stored in contexts.
"""

from __future__ import annotations
from asyncio import Event
from json import loads
from time import time_ns
from typing import Any, Callable, Optional, Dict, TYPE_CHECKING

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from chatsky.slots.slots import SlotManager

if TYPE_CHECKING:
    from chatsky.core.service import ComponentExecutionState
    from chatsky.core.script import Node
    from chatsky.core.pipeline import Pipeline


class ContextError(Exception):
    """Raised when context methods are not used correctly."""


class ServiceState(BaseModel, arbitrary_types_allowed=True):
    execution_status: ComponentExecutionState = Field(default="NOT_RUN")
    """
    :py:class:`.ComponentExecutionState` of this pipeline service.
    Cleared at the end of every turn.
    """
    finished_event: Event = Field(default_factory=Event)
    """
    Asyncio `Event` which can be awaited until this service finishes.
    Cleared at the end of every turn.
    """


class FrameworkData(BaseModel, arbitrary_types_allowed=True):
    """
    Framework uses this to store data related to any of its modules.
    """

    service_states: Dict[str, ServiceState] = Field(default_factory=dict, exclude=True)
    """
    Dictionary containing :py:class:`.ServiceState` of all the pipeline components.
    Cleared at the end of every turn.
    """
    current_node: Optional[Node] = Field(default=None, exclude=True)
    """
    A copy of the current node provided by :py:meth:`~chatsky.core.script.Script.get_inherited_node`.
    This node can be safely modified by Processing functions to alter current node fields.
    """
    pipeline: Optional[Pipeline] = Field(default=None, exclude=True)
    """
    Instance of the pipeline that manages this context.
    Can be used to obtain run configuration such as script or fallback label.
    """
    stats: Dict[str, Any] = Field(default_factory=dict)
    "Enables complex stats collection across multiple turns."
    slot_manager: SlotManager = Field(default_factory=SlotManager)
    "Stores extracted slots."


class ContextMainInfo(BaseModel):
    """
    Main context fields, that are stored in `MAIN` table.
    For most of the database backends, it will be serialized to json.
    For SQL database backends, it will be written to different table columns.
    For memory context storage, it won't be serialized at all.
    """

    current_turn_id: int = Field(default=0)
    """
    Current turn number, specifies the last turn number,
    that is also the last turn available in `labels`, `requests`, and `responses`.
    """
    _created_at: int = PrivateAttr(default_factory=time_ns)
    """
    Timestamp when the context was **first time saved to database**.
    It is set (and managed) by :py:class:`~chatsky.context_storages.DBContextStorage`.
    """
    _updated_at: int = PrivateAttr(default_factory=time_ns)
    """
    Timestamp when the context was **last time saved to database**.
    It is set (and managed) by :py:class:`~chatsky.context_storages.DBContextStorage`.
    """
    misc: Dict[str, Any] = Field(default_factory=dict)
    """
    `misc` stores any custom data. The framework doesn't use this dictionary,
    so storage of any data won't reflect on the work of the internal Chatsky functions.

        - key - Arbitrary data name.
        - value - Arbitrary data.
    """
    framework_data: FrameworkData = Field(default_factory=FrameworkData, validate_default=True)
    """
    This attribute is used for storing custom data required for pipeline execution.
    It is meant to be used by the framework only. Accessing it may result in pipeline breakage.
    """
    origin_interface: Optional[str] = Field(default=None)
    """
    Name of the interface that produced the first request in this context.
    """

    _misc_adaptor: TypeAdapter[Dict[str, Any]] = PrivateAttr(default=TypeAdapter(Dict[str, Any]))

    @field_validator("framework_data", "misc", mode="before")
    @classmethod
    def _validate_framework_data(cls, value: Any) -> Dict:
        if isinstance(value, bytes) or isinstance(value, str):
            value = loads(value)
        return value

    @field_serializer("misc", when_used="always")
    def _serialize_misc(self, misc: Dict[str, Any]) -> bytes:
        return self._misc_adaptor.dump_json(misc)

    @field_serializer("framework_data", when_used="always")
    def _serialize_framework_data(self, framework_data: FrameworkData) -> bytes:
        return framework_data.model_dump_json().encode()

    @model_serializer(mode="wrap", when_used="always")
    def _serialize_model(self, original_serializer) -> Dict[str, Any]:
        model_dict = original_serializer(self)
        model_dict.update({"created_at": self._created_at, "updated_at": self._updated_at})
        return model_dict

    @model_validator(mode="wrap")
    def _validate_model(value: Any, handler: Callable[[Any], "ContextMainInfo"], _) -> "ContextMainInfo":
        if isinstance(value, ContextMainInfo):
            return value
        elif isinstance(value, Dict):
            created_at = value.pop("created_at", time_ns())
            updated_at = value.pop("updated_at", time_ns())
            instance = handler(value)
            instance._created_at = created_at
            instance._updated_at = updated_at
            return instance
        else:
            raise ValueError(f"Unknown type of ContextMainInfo value: {type(value).__name__}!")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ContextMainInfo):
            return (
                self.misc == other.misc
                and self.framework_data == other.framework_data
                and self.origin_interface == other.origin_interface
            )
        return super().__eq__(other)
