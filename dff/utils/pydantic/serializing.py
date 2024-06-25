"""
Serialization
-------------
Tools that provide JSON serialization via Pickle for unserializable objects.

- :py:const:`~.SerializableValue`:
    A field annotated with this will be pickled/unpickled during JSON-serialization/validation.
- :py:const:`~.JSONSerializableDict`:
    A dictionary field annotated with this will make all its items smart-serializable:
    If an item is serializable -- nothing would change.
    Otherwise -- it will be serialized via pickle.
- :py:class:`~.JSONSerializableExtras`:
    A pydantic base class that makes its extra fields a `JSONSerializableDict`.
"""

from base64 import decodebytes, encodebytes
from copy import deepcopy
from pickle import dumps, loads
from typing import Any, Callable, Dict, List, Union
from typing_extensions import Annotated, TypeAlias
from pydantic import (
    AfterValidator,
    JsonValue,
    PlainSerializer,
    PlainValidator,
    RootModel,
    WrapSerializer,
    BaseModel,
    model_validator,
    model_serializer,
)
from pydantic_core import PydanticSerializationError

_JSON_EXTRA_FIELDS_KEYS = "__pickled_extra_fields__"
"""
This key is used in :py:const:`~.JSONSerializableDict` to remember pickled items.
"""

Serializable: TypeAlias = Dict[str, Union[JsonValue, Any, List[Any], Dict[str, Any]]]


class _WrapperModel(RootModel):
    """
    Wrapper model for testing whether an object is serializable to JSON.
    """

    root: Any


def pickle_serializer(value: Any) -> JsonValue:
    """
    Serializer function that serializes any pickle-serializable value into JSON-serializable.
    Serializes value with pickle and encodes bytes as base64 string.

    :param value: Pickle-serializable object.
    :return: String-encoded object.
    """

    return encodebytes(dumps(value)).decode()


def pickle_validator(value: JsonValue) -> Any:
    """
    Validator function that validates base64 string encoded bytes as a pickle-serializable value.
    Decodes base64 string and validates value with pickle.

    :param value: String-encoded string.
    :return: Pickle-serializable object.
    """

    return loads(decodebytes(value.encode()))


def json_pickle_serializer(
    model: Serializable, original_serializer: Callable[[Serializable], Serializable]
) -> Serializable:
    """
    Serializer function that serializes a dictionary or Pydantic object to JSON.
    For every object field, it checks whether the field is JSON serializable,
    and if it's not, serializes it using pickle.
    It also keeps track of pickle-serializable field names in a special list.

    :param model: Pydantic model object or a dictionary.
    :original_serializer: Original serializer function for model.
    :return: model with all the fields serialized to JSON.
    """

    extra_fields = list()
    model_copy = deepcopy(model)

    for field_name, field_value in model_copy.items():
        try:
            if isinstance(field_value, bytes):
                raise PydanticSerializationError("")
            else:
                _WrapperModel(root=field_value).model_dump_json()
        except PydanticSerializationError:
            model_copy[field_name] = pickle_serializer(field_value)
            extra_fields += [field_name]

    original_dump = original_serializer(model_copy)
    if len(extra_fields) > 0:
        original_dump[_JSON_EXTRA_FIELDS_KEYS] = extra_fields
    return original_dump


def json_pickle_validator(model: Serializable) -> Serializable:
    """
    Validator function that validates a JSON dictionary to a python dictionary.
    For every object field, it checks if the field is pickle-serialized,
    and if it is, validates it using pickle.

    :param model: Pydantic model object or a dictionary.
    :original_serializer: Original serializer function for model.
    :return: model with all the fields serialized to JSON.
    """

    model_copy = deepcopy(model)

    if _JSON_EXTRA_FIELDS_KEYS in model.keys():
        for extra_key in model[_JSON_EXTRA_FIELDS_KEYS]:
            extra_value = model[extra_key]
            model_copy[extra_key] = pickle_validator(extra_value)
        del model_copy[_JSON_EXTRA_FIELDS_KEYS]

    return model_copy


PickleSerializer = PlainSerializer(pickle_serializer, when_used="json")
PickleValidator = PlainValidator(pickle_validator)
SerializableValue = Annotated[Any, PickleSerializer, PickleValidator]
"""Annotation for field that makes it JSON serializable"""

JSONPickleSerializer = WrapSerializer(json_pickle_serializer, when_used="json")
JSONPickleValidator = AfterValidator(json_pickle_validator)
JSONSerializableDict = Annotated[Serializable, JSONPickleSerializer, JSONPickleValidator]
"""
Annotation for dictionary or Pydantic model that makes all its fields JSON serializable.

This uses a reserved dictionary key :py:const:`~._JSON_EXTRA_FIELDS_KEYS` to store
fields serialized that way.
"""


class JSONSerializableExtras(BaseModel, extra="allow"):
    """
    This model makes extra fields pickle-serializable.
    Do not use :py:const:`~._JSON_EXTRA_FIELDS_KEYS` as an extra field name.
    """

    def __init__(self, **kwargs):  # supress unknown arg warnings
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def extra_validator(self):
        """
        Validate model along with the `extras` field: i.e. all the fields not listed in the model.

        :return: Validated model.
        """
        self.__pydantic_extra__ = json_pickle_validator(self.__pydantic_extra__)
        return self

    @model_serializer(mode="wrap", when_used="json")
    def extra_serializer(self, original_serializer) -> Dict[str, Any]:
        """
        Serialize model along with the `extras` field: i.e. all the fields not listed in the model.

        :param original_serializer: Function originally used for serialization by Pydantic.
        :return: Serialized model.
        """
        model_copy = self.model_copy(deep=True)
        for extra_name in self.model_extra.keys():
            delattr(model_copy, extra_name)
        model_dict = original_serializer(model_copy)
        model_dict.update(json_pickle_serializer(self.model_extra, original_serializer))
        return model_dict
