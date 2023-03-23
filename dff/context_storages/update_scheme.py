from hashlib import sha256
from re import compile
from enum import Enum, auto, unique
from typing import Dict, List, Optional, Tuple, Iterable, Callable, Any, Union, Awaitable, Hashable

from dff.script import Context


@unique
class FieldType(Enum):
    LIST = auto()
    DICT = auto()
    VALUE = auto()


_ReadListFunction = Callable[[str, Optional[List], Optional[List], Any], Awaitable[Any]]
_ReadDictFunction = Callable[[str, Optional[List], Any], Awaitable[Any]]
_ReadValueFunction = Callable[[str, Any], Awaitable[Any]]
_ReadFunction = Union[_ReadListFunction, _ReadDictFunction, _ReadValueFunction]

_WriteListFunction = Callable[[str, Dict[int, Any], Optional[List], Optional[List], Any], Awaitable]
_WriteDictFunction = Callable[[str, Dict[Hashable, Any], Optional[List], Any], Awaitable]
_WriteValueFunction = Callable[[str, Any, Any], Awaitable]
_WriteFunction = Union[_WriteListFunction, _WriteDictFunction, _WriteValueFunction]


@unique
class FieldRule(Enum):
    READ = auto()
    IGNORE = auto()
    UPDATE = auto()
    HASH_UPDATE = auto()
    APPEND = auto()


class UpdateScheme:
    ALL_ITEMS = "__all__"
    _FIELD_NAME_PATTERN = compile(r"^(.+?)(\[.+\])?$")
    _LIST_FIELD_NAME_PATTERN = compile(r"^.+?(\[([^\[\]]+)\])$")
    _DICT_FIELD_NAME_PATTERN = compile(r"^.+?\[(\[.+\])\]$")

    def __init__(self, dict_scheme: Dict[str, List[str]]):
        self.fields = dict()
        for name, rules in dict_scheme.items():
            field_type = self._get_type_from_name(name)
            if field_type is None:
                raise Exception(f"Field '{name}' not included in Context!")
            field, field_name = self._init_update_field(field_type, name, rules)
            self.fields[field_name] = field

    @classmethod
    def _get_type_from_name(cls, field_name: str) -> Optional[FieldType]:
        if field_name.startswith("requests") or field_name.startswith("responses") or field_name.startswith("labels"):
            return FieldType.LIST
        elif field_name.startswith("misc") or field_name.startswith("framework_states"):
            return FieldType.DICT
        elif field_name.startswith("validation") or field_name.startswith("id"):
            return FieldType.VALUE
        else:
            return None

    @classmethod
    def _init_update_field(cls, field_type: FieldType, field_name: str, rules: List[str]) -> Tuple[Dict, str]:
        field = dict()

        if len(rules) == 0:
            raise Exception(f"For field '{field_name}' the read rule should be defined!")
        elif len(rules) > 2:
            raise Exception(f"For field '{field_name}' more then two (read, write) rules are defined!")
        elif len(rules) == 1:
            rules.append("ignore")

        if rules[0] == "ignore":
            read_rule = FieldRule.IGNORE
        elif rules[0] == "read":
            read_rule = FieldRule.READ
        else:
            raise Exception(f"For field '{field_name}' unknown read rule: '{rules[0]}'!")
        field["read"] = read_rule

        if rules[1] == "ignore":
            write_rule = FieldRule.IGNORE
        elif rules[1] == "update":
            write_rule = FieldRule.UPDATE
        elif rules[1] == "hash_update":
            write_rule = FieldRule.HASH_UPDATE
        elif rules[1] == "append":
            write_rule = FieldRule.APPEND
        else:
            raise Exception(f"For field '{field_name}' unknown write rule: '{rules[1]}'!")
        field["write"] = write_rule

        list_write_wrong_rule = field_type == FieldType.LIST and (write_rule == FieldRule.UPDATE or write_rule == FieldRule.HASH_UPDATE)
        field_write_wrong_rule = field_type != FieldType.LIST and write_rule == FieldRule.APPEND
        if list_write_wrong_rule or field_write_wrong_rule:
            raise Exception(f"Write rule '{write_rule}' not defined for field '{field_name}' of type '{field_type}'!")

        split = cls._FIELD_NAME_PATTERN.match(field_name)
        if field_type == FieldType.VALUE:
            if split.group(2) is not None:
                raise Exception(f"Field '{field_name}' shouldn't have an outlook value - it is of type '{field_type}'!")
            field_name_pure = field_name
        else:
            if split.group(2) is None:
                field_name += "[:]" if field_type == FieldType.LIST else "[[:]]"
            field_name_pure = split.group(1)

        if field_type == FieldType.LIST:
            outlook_match = cls._LIST_FIELD_NAME_PATTERN.match(field_name)
            if outlook_match is None:
                raise Exception(f"Outlook for field '{field_name}' isn't formatted correctly!")

            outlook = outlook_match.group(2).split(":")
            if len(outlook) == 1:
                if outlook == "":
                    raise Exception(f"Outlook array empty for field '{field_name}'!")
                else:
                    try:
                        outlook = eval(outlook_match.group(1), {}, {})
                    except Exception as e:
                        raise Exception(f"While parsing outlook of field '{field_name}' exception happened: {e}")
                    if not isinstance(outlook, List):
                        raise Exception(f"Outlook of field '{field_name}' exception isn't a list - it is of type '{field_type}'!")
                    if not all([isinstance(item, int) for item in outlook]):
                        raise Exception(f"Outlook of field '{field_name}' contains non-integer values!")
                    field["outlook_list"] = outlook
            else:
                if len(outlook) > 3:
                    raise Exception(f"Outlook for field '{field_name}' isn't formatted correctly: '{outlook_match.group(2)}'!")
                elif len(outlook) == 2:
                    outlook.append("1")

                if outlook[0] == "":
                    outlook[0] = "0"
                if outlook[1] == "":
                    outlook[1] = "-1"
                if outlook[2] == "":
                    outlook[2] = "1"
                field["outlook_slice"] = [int(index) for index in outlook]

        elif field_type == FieldType.DICT:
            outlook_match = cls._DICT_FIELD_NAME_PATTERN.match(field_name)
            if outlook_match is None:
                raise Exception(f"Outlook for field '{field_name}' isn't formatted correctly!")

            try:
                outlook = eval(outlook_match.group(1), {}, {"all": cls.ALL_ITEMS})
            except Exception as e:
                raise Exception(f"While parsing outlook of field '{field_name}' exception happened: {e}")
            if not isinstance(outlook, List):
                raise Exception(f"Outlook of field '{field_name}' exception isn't a list - it is of type '{field_type}'!")
            if cls.ALL_ITEMS in outlook and len(outlook) > 1:
                raise Exception(f"Element 'all' should be the only element of the outlook of the field '{field_name}'!")
            field["outlook"] = outlook

        return field, field_name_pure

    @staticmethod
    def get_outlook_slice(dictionary_keys: Iterable, update_field: List) -> List:
        list_keys = sorted(list(dictionary_keys))
        update_field[1] = min(update_field[1], len(list_keys))
        return list_keys[update_field[0]:update_field[1]:update_field[2]] if len(list_keys) > 0 else list()

    @staticmethod
    def get_outlook_list(dictionary_keys: Iterable, update_field: List) -> List:
        list_keys = sorted(list(dictionary_keys))
        return [list_keys[key] for key in update_field] if len(list_keys) > 0 else list()

    async def process_context_read(self, initial: Dict) -> Context:
        context_dict = initial.copy()
        context_hash = dict()
        for field in self.fields.keys():
            if self.fields[field]["read"] == FieldRule.IGNORE:
                del context_dict[field]
                continue
            field_type = self._get_type_from_name(field)
            if field_type is FieldType.LIST:
                if "outlook_slice" in self.fields[field]:
                    update_field = self.get_outlook_slice(context_dict[field].keys(), self.fields[field]["outlook_slice"])
                else:
                    update_field = self.get_outlook_list(context_dict[field].keys(), self.fields[field]["outlook_list"])
                context_dict[field] = {item: context_dict[field][item] for item in update_field}
            elif field_type is FieldType.DICT:
                update_field = self.fields[field].get("outlook", list())
                if self.ALL_ITEMS not in update_field:
                    context_dict[field] = {item: context_dict[field][item] for item in update_field}
            context_hash[field] = sha256(str(context_dict[field]).encode("utf-8"))
        context = Context.cast(context_dict)
        context.framework_states["LAST_STORAGE_HASH"] = context_hash
        return context

    async def process_context_write(self, ctx: Context, initial: Optional[Dict] = None) -> Dict:
        initial = dict() if initial is None else initial
        context_dict = ctx.dict()
        output_dict = dict()
        for field in self.fields.keys():
            if self.fields[field]["write"] == FieldRule.IGNORE:
                if field in initial:
                    output_dict[field] = initial[field]
                elif field in context_dict:
                    output_dict[field] = context_dict[field]
                continue
            field_type = self._get_type_from_name(field)
            initial_field = initial.get(field, dict())

            if field_type is FieldType.LIST:
                if "outlook_slice" in self.fields[field]:
                    update_field = self.get_outlook_slice(context_dict[field].keys(), self.fields[field]["outlook_slice"])
                else:
                    update_field = self.get_outlook_list(context_dict[field].keys(), self.fields[field]["outlook_list"])
                output_dict[field] = initial_field.copy()
                if self.fields[field]["write"] == FieldRule.APPEND:
                    patch = {item: context_dict[field][item] for item in update_field - initial_field.keys()}
                else:
                    patch = {item: context_dict[field][item] for item in update_field}
                output_dict[field].update(patch)
            elif field_type is FieldType.DICT:
                output_dict[field] = dict()
                update_field = self.fields[field].get("outlook", list())
                update_keys_all = set(list(initial_field.keys()) + list(context_dict[field].keys()))
                update_keys = update_keys_all if self.ALL_ITEMS in update_field else update_field
                for item in update_keys:
                    if field == "framework_states" and item == "LAST_STORAGE_HASH":
                        continue
                    if item in initial_field:
                        output_dict[field][item] = initial_field[item]
                    if item in context_dict[field]:
                        output_dict[field][item] = context_dict[field][item]
            else:
                output_dict[field] = context_dict[field]
        return output_dict

    async def process_fields_read(self, processors: Dict[FieldType, _ReadFunction], **kwargs) -> Context:
        result = dict()
        hashes = dict()
        for field in self.fields.keys():
            if self.fields[field]["read"] == FieldRule.IGNORE:
                continue
            field_type = self._get_type_from_name(field)
            if field_type in processors.keys():
                if field_type == FieldType.LIST:
                    outlook_list = self.fields[field].get("outlook_list", None)
                    outlook_slice = self.fields[field].get("outlook_slice", None)
                    result[field] = await processors[field_type](field, outlook_list, outlook_slice, **kwargs)
                elif field_type == FieldType.DICT:
                    outlook = self.fields[field].get("outlook", None)
                    result[field] = await processors[field_type](field, outlook, **kwargs)
                else:
                    result[field] = await processors[field_type](field, **kwargs)
                hashes[field] = sha256(str(result[field]).encode("utf-8"))
        context = Context.cast(result)
        context.framework_states["LAST_STORAGE_HASH"] = hashes
        return context

    async def process_fields_write(self, ctx: Context, processors: Dict[FieldType, _WriteFunction], **kwargs) -> Dict:
        context_dict = ctx.dict()
        for field in self.fields.keys():
            if self.fields[field]["write"] == FieldRule.IGNORE:
                continue
            field_type = self._get_type_from_name(field)
            if field_type in processors.keys():
                if field_type == FieldType.LIST:
                    outlook_list = self.fields[field].get("outlook_list", None)
                    outlook_slice = self.fields[field].get("outlook_slice", None)
                    patch = await processors[field_type](context_dict[field], field, outlook_list, outlook_slice, **kwargs)
                elif field_type == FieldType.DICT:
                    outlook = self.fields[field].get("outlook", None)
                    patch = await processors[field_type](context_dict[field], field, outlook, **kwargs)
                else:
                    patch = await processors[field_type](context_dict[field], field, **kwargs)
                # hashes[field] = sha256(str(result[field]).encode("utf-8"))
        return context_dict


default_update_scheme = UpdateScheme({
    "id": ["read"],
    "requests[-1]": ["read", "append"],
    "responses[-1]": ["read", "append"],
    "labels[-1]": ["read", "append"],
    "misc[[all]]": ["read", "hash_update"],
    "framework_states[[all]]": ["read", "hash_update"],
})

full_update_scheme = UpdateScheme({
    "id": ["read"],
    "requests[:]": ["read", "append"],
    "responses[:]": ["read", "append"],
    "labels[:]": ["read", "append"],
    "misc[[all]]": ["read", "update"],
    "framework_states[[all]]": ["read", "update"],
})
