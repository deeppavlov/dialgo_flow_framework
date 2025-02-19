"""
Context
-------
Context is a data structure that is used to store information about the current state of a conversation.

It is used to keep track of the user's input, the current stage of the conversation, and any other
information that is relevant to the current context of a dialog.

The Context data structure provides several key features to make working with data easier.
Developers can use the context to store any information that is relevant to the current conversation,
such as user data, session data, conversation history, e.t.c.
This allows developers to easily access and use this data throughout the conversation flow.

Another important feature of the context is data serialization.
The context can be easily serialized to a format that can be stored or transmitted, such as JSON.
This allows developers to save the context data and resume the conversation later.
"""

from __future__ import annotations
from asyncio import gather
from uuid import uuid4
from time import time_ns
from typing import Any, Callable, Iterable, List, Optional, Dict, TYPE_CHECKING, Tuple, overload
import logging

from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter, model_validator

from chatsky.context_storages.database import DBContextStorage, NameConfig
from chatsky.core.message import Message
from chatsky.core.node_label import AbsoluteNodeLabel
from chatsky.core.ctx_dict import LabelContextDict, MessageContextDict
from chatsky.core.ctx_utils import ContextError, FrameworkData, ContextMainInfo

if TYPE_CHECKING:
    from chatsky.core.script import Node
    from chatsky.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Context(ContextMainInfo):
    """
    A structure that is used to store data about the context of a dialog.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), exclude=True, frozen=True)
    """
    `id` is the unique context identifier. By default, randomly generated using `uuid4` is used.
    """
    labels: LabelContextDict = Field(default_factory=LabelContextDict)
    """
    `labels` stores dialog labels.
    A new label is stored in the dictionary on every turn, the keys are consecutive integers.
    The first ever (initial) has key `0`.

        - key - Label identification numbers.
        - value - Label data: `AbsoluteNodeLabel`.
    """
    requests: MessageContextDict = Field(default_factory=MessageContextDict)
    """
    `requests` stores dialog requests.
    A new request is stored in the dictionary on every turn, the keys are consecutive integers.
    The first ever (initial) has key `1`.

        - key - Request identification numbers.
        - value - Request data: `Message`.
    """
    responses: MessageContextDict = Field(default_factory=MessageContextDict)
    """
    `responses` stores dialog responses.
    A new response is stored in the dictionary on every turn, the keys are consecutive integers.
    The first ever (initial) has key `1`.

        - key - Response identification numbers.
        - value - Response data: `Message`.
    """
    _storage: Optional[DBContextStorage] = PrivateAttr(None)
    """
    Context storage this context is connected to (if any).
    """

    @classmethod
    async def connected(
        cls, storage: DBContextStorage, start_label: Optional[AbsoluteNodeLabel] = None, id: Optional[str] = None
    ) -> Context:
        """
        Create context **connected** to the given database storage.
        If context ID is given, the corresponding context is loaded from the database.
        If the context does not exist in database or ID is `None`, a new context with new ID is created.
        A connected context can be later stored in the database.

        :param storage: context storage to connect to.
        :param start_label: new context start label (will be set only if the context is created).
        :param id: context ID.
        :return: context, connected to the database.
        """

        if id is None:
            uid = str(uuid4())
            logger.debug(f"Disconnected context created with uid: {uid}")
            instance = cls(id=uid)
            instance.requests = await MessageContextDict.new(storage, uid, NameConfig._requests_field)
            instance.responses = await MessageContextDict.new(storage, uid, NameConfig._responses_field)
            instance.labels = await LabelContextDict.new(storage, uid, NameConfig._labels_field)
            await instance.labels.update({0: start_label})
            instance._storage = storage
            return instance
        else:
            if not isinstance(id, str):
                logger.warning(f"Id is not a string: {id}. Converting to string.")
                id = str(id)
            logger.debug(f"Connected context created with uid: {id}")
            main, labels, requests, responses = await gather(
                storage.load_main_info(id),
                LabelContextDict.connected(storage, id, NameConfig._labels_field),
                MessageContextDict.connected(storage, id, NameConfig._requests_field),
                MessageContextDict.connected(storage, id, NameConfig._responses_field),
            )
            if main is None:
                crt_at = upd_at = time_ns()
                current_turn_id = 0
                misc = dict()
                fw_data = FrameworkData()
                labels[0] = start_label
            else:
                current_turn_id = main.current_turn_id
                crt_at = main._created_at
                upd_at = main._updated_at
                misc = main.misc
                fw_data = main.framework_data
            logger.debug(f"Context loaded with turns number: {len(labels)}")
            instance = cls(
                id=id,
                current_turn_id=current_turn_id,
                misc=misc,
                framework_data=fw_data,
                labels=labels,
                requests=requests,
                responses=responses,
            )
            instance._created_at = crt_at
            instance._updated_at = upd_at
            instance._storage = storage
            return instance

    async def delete(self) -> None:
        """
        Delete connected context from the context storage and disconnect it.
        Throw an error if the context is not connected.
        No local context fields will be affected.
        If the context is not connected, throw a runtime error.
        """

        if self._storage is not None:
            await self._storage.delete_context(self.id)
            self._storage = None
        else:
            raise RuntimeError(f"{type(self).__name__} is not attached to any context storage.")

    @property
    def last_label(self) -> AbsoluteNodeLabel:
        """
        Return label with the highest turn id that is present in :py:attr:`labels`.

        This is not always the label of the transition made during the current turn.
        For that, use ``ctx.labels[ctx.current_turn_id]``.

        :return: Label with the highest turn id.
        :raises ContextError: If there are no labels.
        """

        if len(self.labels) == 0:
            raise ContextError("Labels are empty.")
        return self.labels._items[self.labels.keys()[-1]]

    @property
    def last_response(self) -> Optional[Message]:
        """
        Return response with the highest turn id that is present in :py:attr:`responses`.

        This is not always the response produced during the current turn.
        For that, use ``ctx.responses[ctx.current_turn_id]``.

        :return: Response with the highest turn id or ``None`` if there are no responses.
        """

        if len(self.responses) == 0:
            return None
        return self.responses._items[self.responses.keys()[-1]]

    @property
    def last_request(self) -> Message:
        """
        Return request with the highest turn id that is present in :py:attr:`requests`.

        This is not always the request that initiated the current turn.
        For that, use ``ctx.requests[ctx.current_turn_id]``.

        :return: Request with the highest turn id.
        :raises ContextError: If there are no requests.
        """

        if len(self.requests) == 0:
            raise ContextError("Requests are empty.")
        return self.requests._items[self.requests.keys()[-1]]

    @property
    def pipeline(self) -> Pipeline:
        """
        Return :py:attr:`.FrameworkData.pipeline`.
        """

        pipeline = self.framework_data.pipeline
        if pipeline is None:
            raise ContextError("Pipeline is not set.")
        return pipeline

    @property
    def current_node(self) -> Node:
        """
        Return :py:attr:`.FrameworkData.current_node`.
        """

        node = self.framework_data.current_node
        if node is None:
            raise ContextError("Current node is not set.")
        return node

    class _Turns:
        """
        An instance of class is returned by :py:attr:`~Context.turns`.

        This class only defines a ``__getitem__`` method.

        Key for the method may be a single value or a slice.

        Keep in mind that the turn id 0 is reserved for
        ``start_label`` and does not have a request or response.

        If key is a single value, a tuple of the following is returned:

        1. Request that initiated that turn;
        2. Label of the destination made due to the request;
        3. Response generated by the destination node.

        If any of the items are missing, they are replaced by ``None``.
        If key is negative, corresponding turn is counted from the end
        (e.g. ``ctx.turns[-1]`` is the last turn).

        If key is a slice, the slice is applied to the range of all turn ids from
        0 to the current turn id, and an iterable of tuples described above is returned.

        **Examples:**

            1. ``await ctx.turns[0] == None, start_label, None``;
            2. ``await ctx.turns[-2]`` -- request, label, response of the second to last turn;
            3. ``for request, label, response in await ctx.turns[-5:]`` -- iterate over the last 5 turns.
        """

        def __init__(self, ctx: Context):
            self.ctx = ctx

        @overload
        async def __getitem__(self, key: int) -> Tuple[Message, AbsoluteNodeLabel, Message]:
            pass

        @overload
        async def __getitem__(self, key: slice) -> Iterable[Tuple[Message, AbsoluteNodeLabel, Message]]:
            pass

        async def __getitem__(self, key):
            turn_ids = range(self.ctx.current_turn_id + 1)[key]
            result = await gather(
                self.ctx.requests.get(turn_ids), self.ctx.labels.get(turn_ids), self.ctx.responses.get(turn_ids)
            )
            if isinstance(key, slice):
                return zip(*result)
            else:
                return tuple(result)

    @property
    def turns(self) -> _Turns:
        """
        Return a :py:class:`~Context._Turns` object used to slice turns in the context.
        See the Turn class documentation for more details.
        """
        return self._Turns(self)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Context):
            return (
                self.id == value.id
                and self.current_turn_id == value.current_turn_id
                and self.labels == value.labels
                and self.requests == value.requests
                and self.responses == value.responses
                and self.misc == value.misc
                and self.framework_data == value.framework_data
                and self._storage == value._storage
            )
        else:
            return False

    def __copy__(self):
        storage = self._storage
        self._storage = None
        copy = BaseModel.__copy__(self)
        copy._storage = self._storage = storage
        return copy

    def __deepcopy__(self, memo: dict[int, Any] | None = None):
        storage = self._storage
        self._storage = None
        copy = BaseModel.__deepcopy__(self, memo)
        copy._storage = self._storage = storage
        return copy

    @model_validator(mode="wrap")
    def _validate_model(value: Any, handler: Callable[[Any], "Context"], _) -> "Context":
        if isinstance(value, Context):
            return value
        elif isinstance(value, Dict):
            instance = handler(value)
            labels_obj = value.get("labels", dict())
            if isinstance(labels_obj, Dict):
                labels_obj = TypeAdapter(Dict[int, AbsoluteNodeLabel]).validate_python(labels_obj)
            instance.labels = LabelContextDict.model_validate(labels_obj)
            instance.labels._ctx_id = instance.id
            requests_obj = value.get("requests", dict())
            if isinstance(requests_obj, Dict):
                requests_obj = TypeAdapter(Dict[int, Message]).validate_python(requests_obj)
            instance.requests = MessageContextDict.model_validate(requests_obj)
            instance.requests._ctx_id = instance.id
            responses_obj = value.get("responses", dict())
            if isinstance(responses_obj, Dict):
                responses_obj = TypeAdapter(Dict[int, Message]).validate_python(responses_obj)
            instance.responses = MessageContextDict.model_validate(responses_obj)
            instance.responses._ctx_id = instance.id
            return instance
        else:
            raise ValueError(f"Unknown type of Context value: {type(value).__name__}.")

    async def store(self) -> None:
        """
        Store connected context in the context storage.
        Depending on the context storage settings ("rewrite_existing" flag in particular),
        either only write new and deleted values or also modify the changed ones.
        All the context storage tables are updated asynchronously and simultaneously.
        """

        if self._storage is not None:
            logger.debug(f"Storing context: {self.id}...")
            self._updated_at = time_ns()
            labels_data = self.labels.extract_items()
            requests_data = self.requests.extract_items()
            responses_data = self.responses.extract_items()
            await self._storage.update_context(self.id, self, [labels_data, requests_data, responses_data])
            logger.debug(f"Context stored: {self.id}")
        else:
            raise RuntimeError(f"{type(self).__name__} is not attached to any context storage.")
