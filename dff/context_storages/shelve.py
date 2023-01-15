"""
shelve
---------------------------
Provides the shelve-based version of the :py:class:`.DBContextStorage`.
"""
import pickle
from shelve import DbfilenameShelf
from typing import Hashable

from dff.script import Context

from .database import DBContextStorage


class ShelveContextStorage(DBContextStorage):
    """
    Implements :py:class:`.DBContextStorage` with `shelve` as the driver.

    Parameters
    -----------

    path: str
        Target file URI. Example: `shelve://file.db`
    """

    def __init__(self, path: str):
        DBContextStorage.__init__(self, path)
        self.shelve_db = DbfilenameShelf(filename=self.path, protocol=pickle.HIGHEST_PROTOCOL)

    async def getitem_async(self, key: Hashable) -> Context:
        return self.shelve_db.get(str(key))

    async def setitem_async(self, key: Hashable, value: Context):
        return self.shelve_db.__setitem__(str(key), value)

    async def delitem_async(self, key: Hashable):
        return self.shelve_db.__delitem__(str(key))

    async def contains_async(self, key: Hashable) -> bool:
        return self.shelve_db.__contains__(str(key))

    async def len_async(self) -> int:
        return self.shelve_db.__len__()

    async def clear_async(self):
        return self.shelve_db.clear()
