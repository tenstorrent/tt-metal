# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect

import ttnn._ttnn
import ttnn
import types

from ttnn.decorators import create_module_if_not_exists


def register_tt_lib_operations_as_ttnn_operations(module):
    module_name = module.__name__
    if not (
        module_name.startswith("ttnn._ttnn.deprecated.tensor")
        or module_name.startswith("ttnn._ttnn.deprecated.operations")
        or module_name.startswith("ttnn._ttnn.deprecated.profiler")
    ):
        return
    ttnn_module_name = module_name.replace("ttnn._ttnn.deprecated", "ttnn.experimental")
    for attribute_name in dir(module):
        if attribute_name.startswith("__"):
            continue
        attribute = getattr(module, attribute_name)
        probably_c_function = inspect.isbuiltin(attribute) or (
            hasattr(attribute, "profiler_wrapped_function") and inspect.isbuiltin(attribute.profiler_wrapped_function)
        )
        if probably_c_function and (
            "ttnn._ttnn.deprecated.tensor.Tensor" in attribute.__doc__ or "tt::tt_metal::Tensor" in attribute.__doc__
        ):
            attribute = ttnn.decorators.register_ttl_operation_as_ttnn_operation(
                python_fully_qualified_name=f"{ttnn_module_name}.{attribute_name}", function=attribute
            )
        elif isinstance(attribute, types.ModuleType):
            register_tt_lib_operations_as_ttnn_operations(attribute)
        else:
            target_module = create_module_if_not_exists(ttnn_module_name)
            setattr(target_module, attribute_name, attribute)


register_tt_lib_operations_as_ttnn_operations(ttnn._ttnn.deprecated.tensor)
register_tt_lib_operations_as_ttnn_operations(ttnn._ttnn.deprecated.operations)
register_tt_lib_operations_as_ttnn_operations(ttnn._ttnn.deprecated.profiler)
