"""
This module defines parser objects -- nodes that form a tree.
"""
import typing as tp
from abc import ABC, abstractmethod
import ast
import logging

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # todo: remove this when python3.7 support is dropped

try:
    from ast import unparse
except ImportError:
    from astunparse import unparse  # todo: remove this when python3.8 support is dropped

try:
    remove_suffix = str.removesuffix
    remove_prefix = str.removeprefix
except AttributeError:
    from .utils import remove_prefix, remove_suffix  # todo: remove this when python3.8 support is dropped


if tp.TYPE_CHECKING:
    from .namespace import Namespace
    from .dff_project import DFFProject
from .exceptions import StarError
from .utils import is_instance


logger = logging.getLogger(__name__)

KeywordDict = tp.Dict[str, tp.Union['BaseParserObject', 'KeywordDict']]


class BaseParserObject(ABC):
    """
    An interface for the other parser objects.

    :param parent: Parent node
    :type parent: :py:class:`.BaseParserObject`
    :param child_paths: Mapping from child `id`s to their path relative to `self`
    :type child_paths: dict[int, list[str]]
    :param children: Mapping from
    """
    def __init__(self):
        self.parent: tp.Optional[BaseParserObject] = None
        self.append_path: tp.List[str] = []
        self.children: KeywordDict = {}

    def resolve_path(self, path: tp.List[str]) -> 'BaseParserObject':
        if len(path) == 0:
            return self
        current_dict = self.children
        for index, key in enumerate(path, start=1):
            item = current_dict.get(key)
            if item is None:
                raise KeyError(f"Not found key {key} in {current_dict}\nObject: {repr(self)}")
            if isinstance(item, BaseParserObject):
                return item.resolve_path(path[index:])
            else:
                current_dict = item
        raise KeyError(f"Not found {path} in {self.children}\nObject: {repr(self)}")

    @cached_property
    def path(self) -> tp.List[str]:
        if self.parent is None:
            raise RuntimeError(f"Parent is not set: {repr(self)}")
        return self.parent.path + self.append_path

    @cached_property
    def namespace(self) -> 'Namespace':
        if self.parent is None:
            raise RuntimeError(f"Parent is not set: {repr(self)}")
        return self.parent.namespace

    @cached_property
    def dff_project(self) -> 'DFFProject':
        return self.namespace.dff_project

    @abstractmethod  # todo: add dump function, repr calls it with certain params
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, BaseParserObject):
            return repr(self) == repr(other)
        return False

    @classmethod
    @abstractmethod
    def from_ast(cls, node, **kwargs):
        ...


class Statement(BaseParserObject, ABC):
    """
    This class is for nodes that represent [statements](https://docs.python.org/3.10/library/ast.html#statements)
    """
    @classmethod
    @abstractmethod
    def from_ast(cls, node, **kwargs) -> tp.Dict[str, 'Statement']:
        if isinstance(node, ast.Import):
            return Import.from_ast(node)
        if isinstance(node, ast.ImportFrom):
            return ImportFrom.from_ast(node)
        if isinstance(node, ast.Assign):
            return Assignment.from_ast(node)
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                return Assignment.from_ast(node)
        return {}


class Expression(BaseParserObject, ABC):
    """
    This class is for nodes that represent [expressions](https://docs.python.org/3.10/library/ast.html#expressions)
    """
    @classmethod
    @abstractmethod
    def from_ast(cls, node, **kwargs) -> 'Expression':
        if isinstance(node, ast.Name):
            return Name.from_ast(node)
        if isinstance(node, ast.Attribute):
            return Attribute.from_ast(node)
        if isinstance(node, ast.Dict):
            return Dict.from_ast(node)
        # todo: replace this with isinstance when python3.7 support is dropped
        if is_instance(node, ("_ast.Constant", "ast.Constant")):
            if isinstance(node.value, str):
                return String.from_ast(node)
        if is_instance(node, "_ast.Str"):  # todo: remove this when python3.7 support is dropped
            return String.from_ast(node)
        return Python.from_ast(node)


class ReferenceObject(BaseParserObject, ABC):
    def __init__(self):
        BaseParserObject.__init__(self)

    @cached_property
    @abstractmethod
    def resolve_self(self) -> tp.Optional[BaseParserObject]:
        """

        :return: None, if can't resolve
        """
        ...

    @cached_property
    def absolute(self) -> tp.Optional[BaseParserObject]:  # todo: handle recursion
        """
        Returns an absolute object --  if the current object is a reference to another reference that reference will
        be resolved as well.
        """
        resolved = self.resolve_self
        if isinstance(resolved, ReferenceObject):
            return resolved.absolute
        return resolved

    def __hash__(self):
        return BaseParserObject.__hash__(self.resolve_self or self)

    def __eq__(self, other):
        if isinstance(other, ReferenceObject):
            return BaseParserObject.__eq__(self.resolve_self or self, other.resolve_self or other)
        return BaseParserObject.__eq__(self.resolve_self or self, other)


class Import(Statement, ReferenceObject):
    def __init__(self, module: str, alias: tp.Optional[str] = None):
        Statement.__init__(self)
        ReferenceObject.__init__(self)
        self.module = module
        self.alias = alias

    def __str__(self):
        return f"import {self.module}" + (f" as {self.alias}" if self.alias else "")

    def __repr__(self):
        return f"Import(module={self.module}, alias={self.alias})"

    @cached_property
    def resolve_self(self) -> tp.Optional[BaseParserObject]:
        try:
            return self.dff_project[".".join(self.namespace.resolve_relative_import(self.module))]
        except KeyError as error:
            logger.warning(f"{self.__class__.__name__} did not resolve: {repr(self)}\nKeyError: {error}")
            return None

    @classmethod
    def from_ast(cls, node: ast.Import, **kwargs) -> tp.Dict[str, 'Import']:
        result = {}
        for name in node.names:
            result[name.asname or name.name] = cls(name.name, name.asname)
        return result


class ImportFrom(Statement, ReferenceObject):
    def __init__(self, module: str, level: int, obj: str, alias: tp.Optional[str] = None):
        Statement.__init__(self)
        ReferenceObject.__init__(self)
        self.module = module
        self.level = level
        self.obj = obj
        self.alias = alias

    def __str__(self):
        return f"from {self.level * '.' + self.module} import {self.obj}" + (f" as {self.alias}" if self.alias else "")

    def __repr__(self):
        return f"ImportFrom(module={self.module}, level={self.level}, obj={self.obj}, alias={self.alias})"

    @cached_property
    def resolve_self(self) -> tp.Optional[BaseParserObject]:
        try:
            return self.dff_project[self.namespace.resolve_relative_import(self.module, self.level)][self.obj]
        except KeyError as error:
            logger.warning(f"{self.__class__.__name__} did not resolve: {repr(self)}\nKeyError: {error}")
            return None

    @classmethod
    def from_ast(cls, node: ast.ImportFrom, **kwargs) -> tp.Dict[str, 'ImportFrom']:
        result = {}
        for name in node.names:
            result[name.asname or name.name] = cls(node.module or "", node.level, name.name, name.asname)
        return result


class Assignment(Statement):
    def __init__(self, target: Expression, value: Expression):
        super().__init__()
        target.parent = self
        self.children["target"] = target
        value.parent = self
        self.children["value"] = value

    def __str__(self):
        return f"{str(self.children['target'])} = {str(self.children['value'])}"

    def __repr__(self):
        return f"Assignment(target={repr(self.children['target'])}; value={repr(self.children['value'])}"

    @classmethod
    def from_ast(cls, node, **kwargs) -> tp.Dict[str, 'Assignment']:
        result = {}
        if isinstance(node, ast.Assign):
            target = Expression.from_ast(node.targets[-1])
            result[str(target)] = cls(target=target, value=Expression.from_ast(node.value))
            for target in node.targets[:-1]:
                target = Expression.from_ast(target)
                result[str(target)] = cls(target=target, value=Expression.from_ast(node.targets[-1]))
        if isinstance(node, ast.AnnAssign):
            if node.value is None:
                raise RuntimeError(f"Assignment has no value: {node}")
            target = Expression.from_ast(node.target)
            result[str(target)] = cls(target=target, value=Expression.from_ast(node.value))
        return result


class String(Expression):
    def __init__(self, string: str):
        super().__init__()
        self.string = string

    def __str__(self):
        return repr(self.string)

    def __repr__(self):
        return f"String({self.string})"

    @classmethod
    def from_ast(cls, node: tp.Union['ast.Str', 'ast.Constant'], **kwargs) -> 'String':
        if is_instance(node, "_ast.Str"):  # todo: remove this when python3.7 support is dropped
            return cls(node.s)
        # todo: replace this with isinstance when python3.7 support is dropped
        elif is_instance(node, ("_ast.Constant", "ast.Constant")):
            return cls(node.value)
        raise RuntimeError(f"Node {node} is not str")


class Python(Expression):
    def __init__(self, string: str):
        super().__init__()
        self.string = string

    def __str__(self):
        return self.string

    def __repr__(self):
        return f"Python({self.string})"

    @classmethod
    def from_ast(cls, node: ast.AST, **kwargs) -> 'Python':
        return cls(remove_suffix(unparse(node), "\n"))


class Dict(Expression):
    def __init__(self, dictionary: tp.Dict[Expression, Expression]):
        super().__init__()
        self.keys: tp.Dict[Expression, str] = {}
        for key, value in dictionary.items():
            key.parent = self
            value.parent = self
            key.append_path = [repr(key), "key"]
            value.append_path = [repr(key), "value"]
            self.keys[key] = repr(key)
            self.children[repr(key)] = {}
            self.children[repr(key)]["key"] = key
            self.children[repr(key)]["value"] = value

    def __str__(self):
        return "{" + ", ".join(
            [f"{str(value['key'])}: {str(value['value'])}" for value in self.children.values()]
        ) + "}"

    def __repr__(self):
        return "Dict(" + ", ".join(
            [f"{repr(value['key'])}: {repr(value['value'])}" for value in self.children.values()]
        ) + ")"

    def __getitem__(self, item: tp.Union[Expression, str]):
        if isinstance(item, Expression):
            key = self.keys[item]
            return self.children[key]["value"]
        elif isinstance(item, str):
            dict_item = self.children[item]
            return dict_item["value"]
        else:
            raise TypeError(f"Item {repr(item)} is not `BaseParserObject` nor `str")

    @classmethod
    def from_ast(cls, node: ast.Dict, **kwargs) -> 'Dict':
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise StarError(f"Dict comprehensions are not supported: {unparse(node)}")
            result[Expression.from_ast(key)] = Expression.from_ast(value)
        return cls(result)


class Name(Expression, ReferenceObject):
    def __init__(self, name: str):
        Expression.__init__(self)
        ReferenceObject.__init__(self)
        self.name = name

    @cached_property
    def resolve_self(self) -> tp.Optional[BaseParserObject]:
        try:
            return self.namespace[self.name]
        except KeyError as error:
            logger.warning(f"{self.__class__.__name__} did not resolve: {repr(self)}\nKeyError: {error}")
            return None

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Name({self.name})"

    @classmethod
    def from_ast(cls, node: ast.Name, **kwargs) -> 'Expression':
        return cls(node.id)


class Attribute(Expression, ReferenceObject):
    def __init__(self, value: Expression, attr: str):
        Expression.__init__(self)
        ReferenceObject.__init__(self)
        value.parent = self
        value.append_path = ["value"]
        self.children["value"] = value
        self.attr = attr

    @cached_property
    def resolve_self(self) -> tp.Optional[BaseParserObject]:
        try:
            value = self.children["value"]
            if isinstance(value, ReferenceObject):
                value = value.absolute
            if is_instance(value, "dff.script.parser.namespace.Namespace"):
                return value[self.attr]
            return None
        except KeyError as error:
            logger.warning(f"{self.__class__.__name__} did not resolve: {repr(self)}\nKeyError: {error}")
            return None

    def __str__(self):
        return str(self.children["value"]) + "." + self.attr

    def __repr__(self):
        return f"Attribute(value={repr(self.children['value'])}; attr={self.attr})"

    @classmethod
    def from_ast(cls, node: ast.Attribute, **kwargs) -> 'Expression':
        return cls(Expression.from_ast(node.value), node.attr)
