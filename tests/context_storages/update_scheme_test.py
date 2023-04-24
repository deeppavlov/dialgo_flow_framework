from typing import List, Dict, Hashable, Any, Union
from uuid import UUID

import pytest

from dff.context_storages import UpdateScheme, default_update_scheme, full_update_scheme
from dff.script import Context


@pytest.mark.asyncio
async def default_scheme_creation(context_id, testing_context):
    context_storage = dict()

    async def fields_reader(field_name: str, _: Union[UUID, int, str], ext_id: Union[UUID, int, str]):
        container = context_storage.get(ext_id, list())
        return list(container[-1].dict().get(field_name, dict()).keys()) if len(container) > 0 else list()

    async def read_sequence(
        field_name: str, outlook: List[Hashable], _: Union[UUID, int, str], ext_id: Union[UUID, int, str]
    ) -> Dict[Hashable, Any]:
        container = context_storage.get(ext_id, list())
        return (
            {item: container[-1].dict().get(field_name, dict()).get(item, None) for item in outlook}
            if len(container) > 0
            else dict()
        )

    async def read_value(field_name: str, _: Union[UUID, int, str], ext_id: Union[UUID, int, str]) -> Any:
        container = context_storage.get(ext_id, list())
        return container[-1].dict().get(field_name, None) if len(container) > 0 else None

    async def write_anything(field_name: str, data: Any, _: Union[UUID, int, str], ext_id: Union[UUID, int, str]):
        container = context_storage.setdefault(ext_id, list())
        if len(container) > 0:
            container[-1] = Context.cast({**container[-1].dict(), field_name: data})
        else:
            container.append(Context.cast({field_name: data}))

    default_scheme = UpdateScheme.from_dict_schema(default_update_scheme)
    print(default_scheme.__dict__)

    full_scheme = UpdateScheme.from_dict_schema(full_update_scheme)
    print(full_scheme.__dict__)

    out_ctx = testing_context
    print(out_ctx.dict())

    mid_ctx = await default_scheme.write_context(out_ctx, None, fields_reader, write_anything, context_id)
    print(mid_ctx)

    context, hashes = await default_scheme.read_context(fields_reader, read_value, out_ctx.id, context_id)
    print(context.dict())
