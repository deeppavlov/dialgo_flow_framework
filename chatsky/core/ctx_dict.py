"""
Context Dict
------------
This module defines classes for lazy context data loading.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from asyncio import gather
from hashlib import sha256
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    overload,
    TYPE_CHECKING,
)

from pydantic import BaseModel, PrivateAttr, TypeAdapter, model_serializer, model_validator

from chatsky.core.message import Message
from chatsky.core.node_label import AbsoluteNodeLabel
from chatsky.utils.logging import collapse_num_list

if TYPE_CHECKING:
    from chatsky.context_storages.database import DBContextStorage

logger = logging.getLogger(__name__)


def _get_hash(string: bytes) -> bytes:
    return sha256(string).digest()


class ContextDict(ABC, BaseModel):
    """
    Dictionary-like structure for storing dialog data spanning multiple turns in a context storage.
    It holds all the possible keys, but may not store all the values locally.
    Values not stored locally will be loaded upon querying.

    Keys of the dictionary are turn ids.
    Get, set and delete operations support slices in addition to regular single-key operation.
    Examples:

    1. ``await ctx.labels[0]`` returns label at turn 0 (start label).
    2. ``await ctx.requests[1:4]`` returns the first 3 requests (requests at turns 1, 2, 3).
    3. ``ctx.responses[2:5:2] = Message("2"), Message("4")`` sets responses at turns 2 and 4.

    Get operation is asynchronous (and requires ``await``) since it calls
    :py:meth:`_load_items` in order to get items that need to be loaded from DB.
    """

    _items: Dict[int, BaseModel] = PrivateAttr(default_factory=dict)
    """
    Dictionary of already loaded from storage items.
    """

    _hashes: Dict[int, int] = PrivateAttr(default_factory=dict)
    """
    Hashes of the loaded items (as they were upon loading), only used if `rewrite_existing` flag is enabled.
    """

    _keys: Set[int] = PrivateAttr(default_factory=set)
    """
    All the item keys available either in storage or locally.
    """

    _added: Set[int] = PrivateAttr(default_factory=set)
    """
    Keys added locally (need to be synchronized with the storage).
    Synchronization happens whenever `store` is called (which is done at
    the end of every turn).
    """

    _removed: Set[int] = PrivateAttr(default_factory=set)
    """
    Keys removed locally (need to be synchronized with the storage).
    Synchronization happens whenever `store` is called (which is done at
    the end of every turn).
    """

    _storage: Optional[DBContextStorage] = PrivateAttr(None)
    """
    Context storage for item synchronization.
    """

    _ctx_id: str = PrivateAttr(default_factory=str)
    """
    Corresponding context ID.
    """

    _field_name: str = PrivateAttr(default_factory=str)
    """
    Name of the field in the context storage that is represented by the given dict.
    """

    @property
    @abstractmethod
    def _value_type(self) -> TypeAdapter[BaseModel]:
        raise NotImplementedError

    @classmethod
    async def new(cls, storage: DBContextStorage, id: str, field: str) -> "ContextDict":
        """
        Create a new context dict, without connecting it to the context storage.
        No keys or items will be loaded, but any newly added items will be available for synchronization.
        Should be used when we are *sure* that context with given ID does not exist in the storage.

        :param storage: Context storage, where the new items will be added.
        :param id: Newly created context ID.
        :param field: Current dict field name.
        :return: New "disconnected" context dict.
        """

        instance = cls()
        logger.debug(f"Disconnected context dict created for id {id} and field name: {field}")
        instance._ctx_id = id
        instance._field_name = field
        instance._storage = storage
        return instance

    @classmethod
    async def connected(cls, storage: DBContextStorage, id: str, field: str) -> "ContextDict":
        """
        Create a new context dict, connecting it to the context storage.
        All the keys and some items will be loaded, all the other items will be available for synchronization.
        Also hashes will be calculated for the initially loaded items for modification tracking.

        :param storage: Context storage, keeping the current context.
        :param id: Newly created context ID.
        :param field: Current dict field name.
        :return: New "connected" context dict.
        """

        logger.debug(f"Connected context dict created for {id}, {field}")
        keys, items = await gather(storage.load_field_keys(id, field), storage.load_field_latest(id, field))
        val_key_items = [(k, v) for k, v in items if v is not None]
        logger.debug(f"Context dict for {id}, {field} loaded: {collapse_num_list(keys)}")
        instance = cls()
        instance._storage = storage
        instance._ctx_id = id
        instance._field_name = field
        instance._keys = set(keys)
        instance._items = {k: instance._value_type.validate_json(v) for k, v in val_key_items}
        instance._hashes = {k: _get_hash(v) for k, v in val_key_items} if storage.rewrite_existing else dict()
        return instance

    async def _load_items(self, keys: List[int]) -> None:
        """
        Load items for the given keys from the connected context storage.
        Update the `_items` and `_hashes` fields if necessary.
        NB! If not all the requested items are available,
        only the successfully loaded will be updated and no error will be raised.

        :param keys: The requested key array.
        """

        logger.debug(
            f"Context dict for {self._ctx_id}, {self._field_name} loading extra items: {collapse_num_list(keys)}..."
        )
        items = await self._storage.load_field_items(self._ctx_id, self._field_name, keys)
        logger.debug(
            f"Context dict for {self._ctx_id}, {self._field_name} extra items loaded: {collapse_num_list(keys)}"
        )
        for key, value in items:
            self._items[key] = self._value_type.validate_json(value)
            if self._storage.rewrite_existing:
                self._hashes[key] = _get_hash(value)

    @overload
    async def __getitem__(self, key: int) -> BaseModel: ...  # noqa: E704

    @overload
    async def __getitem__(self, key: slice) -> Tuple[BaseModel]: ...  # noqa: E704

    async def __getitem__(self, key):
        if isinstance(key, int):
            if key not in self:
                raise KeyError(f"Key {key} does not exist.")
            if self._storage is not None and key not in self._items:
                await self._load_items([key])
            return self._items[key]

        elif isinstance(key, slice):
            slice_keys = list(range(key.start, key.stop, key.step if key.step is not None else 1))

            invalid_keys = [k for k in slice_keys if k not in self]
            if invalid_keys:
                raise KeyError(f"The following keys do not exist: {invalid_keys}.")

            keys_to_load = [k for k in slice_keys if k not in self._items]

            if self._storage is not None and keys_to_load:
                await self._load_items(keys_to_load)

            return tuple(self._items[k] for k in slice_keys)
        else:
            raise TypeError(f"Key must be either an integer or an iterable of integers: {key}.")

    @overload
    def __setitem__(self, key: int, value: BaseModel) -> None:
        pass

    @overload
    def __setitem__(self, key: slice, value: Sequence[BaseModel]) -> None:
        pass

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._keys.add(key)
            self._added.add(key)
            self._removed.discard(key)

            self._items[key] = self._value_type.validate_python(value)
        elif isinstance(key, slice):
            if isinstance(value, Sequence):
                slice_keys = list(range(key.start, key.stop, key.step if key.step is not None else 1))

                if len(slice_keys) != len(value):
                    raise ValueError("Key slice and value sequence must have the same length.")

                for k, v in zip(slice_keys, value):
                    self[k] = v
            else:
                raise ValueError("Key slice must have sequence value.")
        else:
            raise TypeError(f"Key must be either an integer or an iterable of integers: {key}.")

    def __delitem__(self, key: Union[int, slice]) -> None:
        if isinstance(key, int):
            self._removed.add(key)
            self._added.discard(key)
            self._keys.discard(key)

            del self._items[key]
        elif isinstance(key, slice):
            slice_keys = list(range(key.start, key.stop, key.step if key.step is not None else 1))

            for k in slice_keys:
                del self[k]
        else:
            raise TypeError(f"Key must be either an integer or an iterable of integers: {key}.")

    def __iter__(self) -> Iterable[int]:
        yield from self.keys()

    def __len__(self) -> int:
        return len(self._keys)

    @overload
    async def get(self, key: int, default=None) -> BaseModel: ...  # noqa: E704

    @overload
    async def get(self, key: Iterable[int], default=None) -> Tuple[BaseModel]: ...  # noqa: E704

    async def get(self, key, default=None):
        """
        Get one or many items from the dict.
        Asynchronously load missing ones, if context storage is connected.

        :param key: Key or an iterable of keys for item retrieving.
        :param default:
            Default value.
            Default is returned when `key` is a single key that is not in the dict.
            If `key` is an iterable of keys, default is returned instead of all
            the values that are not in the dict.
        :return:
            A single value if `key` is a single key.
            Tuple of values if `key` is an iterable.
        """
        if isinstance(key, int):
            if self._storage is not None and key in self and key not in self._items:
                await self._load_items([key])
            return self._items.get(key, default)

        if isinstance(key, Iterable) and all([isinstance(k, int) for k in key]):
            keys_to_load = [k for k in key if k in self and k not in self._items]

            if self._storage is not None and keys_to_load:
                await self._load_items(keys_to_load)

            return tuple(self._items.get(k, default) for k in key)
        else:
            raise TypeError(f"Key must be either an integer or an iterable of integers: {key}.")

    def __contains__(self, key: int) -> bool:
        return key in self._keys

    def keys(self) -> List[int]:
        return sorted(self._keys)

    async def values(self) -> List[BaseModel]:
        keys_to_load = [k for k in self.keys() if k not in self._items]

        if self._storage is not None and keys_to_load:
            await self._load_items(keys_to_load)
        return [self._items[key] for key in self.keys()]

    async def items(self) -> List[Tuple[int, BaseModel]]:
        return [(k, v) for k, v in zip(self.keys(), await self.values())]

    async def pop(self, key: int, default=None) -> BaseModel:
        try:
            value = await self[key]
        except KeyError:
            return default
        else:
            del self[key]
            return value

    def clear(self) -> None:
        for key in self.keys():
            del self[key]

    async def update(self, other: Any = (), /, **kwds) -> None:
        if isinstance(other, ContextDict):
            await self.update(zip(other.keys(), await other.values()))
        elif isinstance(other, Mapping):
            for key in other:
                self[key] = other[key]
        elif hasattr(other, "keys"):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value
        for key, value in kwds.items():
            self[key] = value

    async def setdefault(self, key: int, default=None) -> BaseModel:
        try:
            return await self[key]
        except KeyError:
            self[key] = default
        return default

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ContextDict):
            return self._items == value._items
        elif isinstance(value, Dict):
            return self._items == value
        else:
            return False

    def __repr__(self) -> str:
        return (
            f"ContextDict(items={self._items}, "
            f"keys={list(self.keys())}, "
            f"hashes={self._hashes}, "
            f"added={self._added}, "
            f"removed={self._removed}, "
            f"storage={self._storage}, "
            f"ctx_id={self._ctx_id}, "
            f"field_name={self._field_name})"
        )

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
    def _validate_model(value: Any, handler: Callable[[Any], "ContextDict"], _) -> "ContextDict":
        if isinstance(value, ContextDict):
            return value
        elif isinstance(value, Dict):
            instance = handler(dict())
            if len(value) != 0:
                instance._items = TypeAdapter(Dict[int, BaseModel]).validate_python(value.get("items", dict()))
                instance._hashes = TypeAdapter(Dict[int, int]).validate_python(value.get("hashes", dict()))
                instance._keys = TypeAdapter(Set[int]).validate_python(value.get("keys", set(instance._items.keys())))
                instance._added = TypeAdapter(Set[int]).validate_python(value.get("added", set()))
                instance._removed = TypeAdapter(Set[int]).validate_python(value.get("removed", set()))
                instance._ctx_id = TypeAdapter(str).validate_python(value.get("ctx_id", str()))
                instance._field_name = TypeAdapter(str).validate_python(value.get("field_name", str()))
            else:
                instance._items = dict()
                instance._keys = set()
            return instance
        else:
            raise ValueError(f"Unknown type of ContextDict value: {type(value).__name__}.")

    @model_serializer(mode="wrap", when_used="always")
    def _serialize_model(self, original_serializer) -> Dict[str, Any]:
        model_dict = original_serializer(self)
        model_dict.update(
            {
                "items": self._items,
                "hashes": self._hashes,
                "keys": self._keys,
                "added": self._added,
                "removed": self._removed,
                "ctx_id": self._ctx_id,
                "field_name": self._field_name,
            }
        )
        return model_dict

    @classmethod
    def import_items(cls, value: Any) -> "ContextDict":
        if isinstance(value, ContextDict):
            return value
        elif isinstance(value, Dict):
            instance = cls()
            instance._items = value.copy()
            instance._keys = set(value.keys())
            return instance
        else:
            raise ValueError(f"Unknown type of ContextDict value: {type(value).__name__}!")

    def extract_items(self) -> Tuple[str, List[Tuple[int, bytes]], List[int]]:
        """
        Synchronize dict state with the connected storage, extract the data that should be updated.
        Update added and removed elements, also update modified ones if `rewrite_existing` flag is enabled.
        Raise an error if no storage is connected.
        """

        if self._storage is not None:
            logger.debug(f"Extracting items for {self._ctx_id}...")
            if self._storage.rewrite_existing:
                added_items = list()
                for k, v in self._items.items():
                    value = self._value_type.dump_json(v)
                    if _get_hash(value) != self._hashes.get(k):
                        added_items += [(k, value)]
            else:
                added_items = [(k, self._value_type.dump_json(self._items[k])) for k in self._added]
            logger.debug(f"Storing context dict for {self._ctx_id}, {self._field_name}...")
            removed_items = list(self._removed - self._added)
            logger.debug(
                f"Context dict for {self._ctx_id}, {self._field_name} stored: "
                f"{collapse_num_list([k for k, _ in added_items])}"
            )
            self._added, self._removed = set(), set()
            if self._storage.rewrite_existing:
                for k, v in self._items.items():
                    self._hashes[k] = _get_hash(self._value_type.dump_json(v))
            return self._field_name, added_items, removed_items
        else:
            raise RuntimeError(f"{type(self).__name__} is not attached to any context storage.")


class LabelContextDict(ContextDict):
    """
    Context dictionary for storing `AbsoluteNodeLabel` types.
    """

    _items: Dict[int, AbsoluteNodeLabel]

    @property
    def _value_type(self) -> TypeAdapter[AbsoluteNodeLabel]:
        return TypeAdapter(AbsoluteNodeLabel)

    @overload
    async def __getitem__(self, key: int) -> AbsoluteNodeLabel: ...  # noqa: E704

    @overload
    async def __getitem__(self, key: slice) -> List[AbsoluteNodeLabel]: ...  # noqa: E704

    async def __getitem__(self, key):
        return await super().__getitem__(key)

    @overload
    def __setitem__(self, key: int, value: AbsoluteNodeLabel) -> None:
        pass

    @overload
    def __setitem__(self, key: slice, value: Sequence[AbsoluteNodeLabel]) -> None:
        pass

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    @overload
    async def get(self, key: int, default=None) -> AbsoluteNodeLabel: ...  # noqa: E704

    @overload
    async def get(self, key: Iterable[int], default=None) -> List[AbsoluteNodeLabel]: ...  # noqa: E704

    async def get(self, key, default=None):
        return await super().get(key, default)

    async def values(self) -> List[AbsoluteNodeLabel]:
        return await super().values()

    async def items(self) -> List[Tuple[int, AbsoluteNodeLabel]]:
        return await super().items()

    async def pop(self, key: int, default=None) -> AbsoluteNodeLabel:
        return await super().pop(key, default)

    async def setdefault(self, key: int, default=None) -> AbsoluteNodeLabel:
        return await super().setdefault(key, default)


class MessageContextDict(ContextDict):
    """
    Context dictionary for storing `Message` types.
    """

    _items: Dict[int, Message]

    @property
    def _value_type(self) -> TypeAdapter[Message]:
        return TypeAdapter(Message)

    @overload
    async def __getitem__(self, key: int) -> Message: ...  # noqa: E704

    @overload
    async def __getitem__(self, key: slice) -> List[Message]: ...  # noqa: E704

    async def __getitem__(self, key):
        return await super().__getitem__(key)

    @overload
    def __setitem__(self, key: int, value: Message) -> None:
        pass

    @overload
    def __setitem__(self, key: slice, value: Sequence[Message]) -> None:
        pass

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    @overload
    async def get(self, key: int, default=None) -> Message: ...  # noqa: E704

    @overload
    async def get(self, key: Iterable[int], default=None) -> List[Message]: ...  # noqa: E704

    async def get(self, key, default=None):
        return await super().get(key, default)

    async def values(self) -> List[Message]:
        return await super().values()

    async def items(self) -> List[Tuple[int, Message]]:
        return await super().items()

    async def pop(self, key: int, default=None) -> Message:
        return await super().pop(key, default)

    async def setdefault(self, key: int, default=None) -> Message:
        return await super().setdefault(key, default)
