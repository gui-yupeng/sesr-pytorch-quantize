from typing import Any, Optional
from typing import Callable, Type, Dict, Tuple
from typing import TypeVar
from torch import nn

import torch


class FunctionPackage:
    def __init__(self, the_function: Callable, parameter_dict: Optional[Dict[str, Any]] = None):
        self.function = the_function
        self.parameter_dict = parameter_dict


class NodeInsertConfig:
    def __init__(self, should_insert: bool, function_package: FunctionPackage = None):
        self.should_insert = should_insert
        self.function_package = function_package


class NodeInsertMappingElement:
    def __init__(self, insert_type , the_function: FunctionPackage):
        self.insert_mapping_config = (insert_type, the_function)

    def get_config(self) -> Tuple[Type, FunctionPackage]:
        return self.insert_mapping_config


class NodeInsertMapping:
    def __init__(self):
        self.insert_mapping = {}

    def add_config(self, insert_mapping_config: NodeInsertMappingElement):
        config = insert_mapping_config.get_config()
        self.insert_mapping.update(dict([config]))

    def get_mapping(self):
        return self.insert_mapping

