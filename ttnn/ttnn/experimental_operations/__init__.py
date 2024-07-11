# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect

import ttnn._ttnn
import ttnn
import types


def register_tt_lib_operations_as_ttnn_operations(module):
    module_name = module.__name__
    if not module_name.startswith("ttnn._ttnn.deprecated.tensor") and not module_name.startswith(
        "ttnn._ttnn.deprecated.operations"
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
            ttnn_module = ttnn
            ttnn_module_path = ttnn_module_name.split(".")[1:]
            while ttnn_module_path:
                ttnn_submodule_name = ttnn_module_path.pop(0)
                ttnn_submodule = getattr(ttnn_module, ttnn_submodule_name, types.ModuleType(ttnn_submodule_name))
                setattr(ttnn_module, ttnn_submodule_name, ttnn_submodule)
                ttnn_module = ttnn_submodule
            setattr(ttnn_module, attribute_name, attribute)


register_tt_lib_operations_as_ttnn_operations(ttnn._ttnn.deprecated.tensor)
register_tt_lib_operations_as_ttnn_operations(ttnn._ttnn.deprecated.operations)
