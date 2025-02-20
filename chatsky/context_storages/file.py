"""
File
----
The File module provides simple file-based versions of the :py:class:`.DBContextStorage` class
such as json, pickle and shelve.
"""

from abc import ABC, abstractmethod
from pickle import loads, dumps
from shelve import DbfilenameShelf
from typing import Any, List, Set, Tuple, Dict, Optional

from pydantic import BaseModel, Field

from .database import DBContextStorage, _SUBSCRIPT_DICT

try:
    from aiofiles import open
    from aiofiles.os import stat, makedirs
    from aiofiles.ospath import isfile

    json_available = True
    pickle_available = True
except ImportError:
    json_available = False
    pickle_available = False


class SerializableStorage(BaseModel):
    """
    A special serializable database implementation.
    One element of this class will be used to store all the contexts, read and written to file on every turn.
    """

    main: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    turns: List[Tuple[str, str, int, Optional[bytes]]] = Field(default_factory=list)


class FileContextStorage(DBContextStorage, ABC):
    """
    Implements :py:class:`.DBContextStorage` with any file-based storage format.

    :param path: Target file URI.
    :param rewrite_existing: Whether `TURNS` modified locally should be updated in database or not.
    :param partial_read_config: Dictionary of subscripts for all possible turn items.
    """

    is_concurrent: bool = False

    def __init__(
        self,
        path: str = "",
        rewrite_existing: bool = False,
        partial_read_config: Optional[_SUBSCRIPT_DICT] = None,
    ):
        DBContextStorage.__init__(self, path, rewrite_existing, partial_read_config)

    @abstractmethod
    async def _save(self, data: SerializableStorage) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _load(self) -> SerializableStorage:
        raise NotImplementedError

    async def _connect(self):
        await self._load()

    async def _load_main_info(self, ctx_id: str) -> Optional[Dict[str, Any]]:
        return (await self._load()).main.get(ctx_id, None)

    async def _update_context(
        self,
        ctx_id: str,
        ctx_info: Optional[Dict[str, Any]],
        field_info: List[Tuple[str, List[Tuple[int, Optional[bytes]]]]],
    ) -> None:
        storage = await self._load()
        if ctx_info is not None:
            storage.main[ctx_id] = ctx_info
        for field_name, items in field_info:
            for k, v in items:
                upd = (ctx_id, field_name, k, v)
                for i in range(len(storage.turns)):
                    if storage.turns[i][:-1] == upd[:-1]:
                        storage.turns[i] = upd
                        break
                else:
                    storage.turns += [upd]
        await self._save(storage)

    async def _delete_context(self, ctx_id: str) -> None:
        storage = await self._load()
        storage.main.pop(ctx_id, None)
        storage.turns = [(c, f, k, v) for c, f, k, v in storage.turns if c != ctx_id]
        await self._save(storage)

    async def _load_field_latest(self, ctx_id: str, field_name: str) -> List[Tuple[int, bytes]]:
        storage = await self._load()
        select = sorted(
            [(k, v) for c, f, k, v in storage.turns if c == ctx_id and f == field_name and v is not None],
            key=lambda e: e[0],
            reverse=True,
        )
        if isinstance(self._subscripts[field_name], int):
            select = select[: self._subscripts[field_name]]
        elif isinstance(self._subscripts[field_name], Set):
            select = [(k, v) for k, v in select if k in self._subscripts[field_name]]
        return select

    async def _load_field_keys(self, ctx_id: str, field_name: str) -> List[int]:
        return [k for c, f, k, v in (await self._load()).turns if c == ctx_id and f == field_name and v is not None]

    async def _load_field_items(self, ctx_id: str, field_name: str, keys: List[int]) -> List[Tuple[int, bytes]]:
        return [
            (k, v)
            for c, f, k, v in (await self._load()).turns
            if c == ctx_id and f == field_name and k in keys and v is not None
        ]

    async def _clear_all(self) -> None:
        await self._save(SerializableStorage())


class JSONContextStorage(FileContextStorage):
    """
    Implements :py:class:`.DBContextStorage` with `json` as the storage format.

    :param path: Target file URI. Example: `json://file.json`.
    :param rewrite_existing: Whether `TURNS` modified locally should be updated in database or not.
    :param partial_read_config: Dictionary of subscripts for all possible turn items.
    """

    async def _save(self, data: SerializableStorage) -> None:
        if not await isfile(self.path) or (await stat(self.path)).st_size == 0:
            await makedirs(self.path.parent, exist_ok=True)
        async with open(self.path, "w", encoding="utf-8") as file_stream:
            await file_stream.write(data.model_dump_json())

    async def _load(self) -> SerializableStorage:
        if not await isfile(self.path) or (await stat(self.path)).st_size == 0:
            storage = SerializableStorage()
            await self._save(storage)
        else:
            async with open(self.path, "r", encoding="utf-8") as file_stream:
                storage = SerializableStorage.model_validate_json(await file_stream.read())
        return storage


class PickleContextStorage(FileContextStorage):
    """
    Implements :py:class:`.DBContextStorage` with `pickle` as the storage format.

    :param path: Target file URI. Example: `pickle://file.pkl`.
    :param rewrite_existing: Whether `TURNS` modified locally should be updated in database or not.
    :param partial_read_config: Dictionary of subscripts for all possible turn items.
    """

    async def _save(self, data: SerializableStorage) -> None:
        if not await isfile(self.path) or (await stat(self.path)).st_size == 0:
            await makedirs(self.path.parent, exist_ok=True)
        async with open(self.path, "wb") as file_stream:
            await file_stream.write(dumps(data.model_dump()))

    async def _load(self) -> SerializableStorage:
        if not await isfile(self.path) or (await stat(self.path)).st_size == 0:
            storage = SerializableStorage()
            await self._save(storage)
        else:
            async with open(self.path, "rb") as file_stream:
                storage = SerializableStorage.model_validate(loads(await file_stream.read()))
        return storage


class ShelveContextStorage(FileContextStorage):
    """
    Implements :py:class:`.DBContextStorage` with `shelve` as the storage format.

    :param path: Target file URI. Example: `shelve://file.shlv`.
    :param rewrite_existing: Whether `TURNS` modified locally should be updated in database or not.
    :param partial_read_config: Dictionary of subscripts for all possible turn items.
    """

    _SHELVE_ROOT = "root"

    def __init__(
        self,
        path: str = "",
        rewrite_existing: bool = False,
        partial_read_config: Optional[_SUBSCRIPT_DICT] = None,
    ):
        self._storage = None
        FileContextStorage.__init__(self, path, rewrite_existing, partial_read_config)

    async def _save(self, data: SerializableStorage) -> None:
        self._storage[self._SHELVE_ROOT] = data.model_dump()

    async def _load(self) -> SerializableStorage:
        if self._storage is None:
            content = SerializableStorage()
            self._storage = DbfilenameShelf(str(self.path.absolute()), writeback=True)
            await self._save(content)
        else:
            content = SerializableStorage.model_validate(self._storage[self._SHELVE_ROOT])
        return content
