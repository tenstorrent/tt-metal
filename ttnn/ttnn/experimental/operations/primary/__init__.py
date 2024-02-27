import inspect
import sys
import types

import tt_lib as ttl
import ttnn

THIS_MODULE = sys.modules[__name__]


def create_or_get_module(module_path):
    """Create a new module if it does not exist, or return the existing one."""
    parts = module_path.split(".")
    current_path = []
    module = None
    for part in parts:
        current_path.append(part)
        full_path = ".".join(current_path)
        if full_path not in sys.modules:
            new_module = types.ModuleType(full_path)
            sys.modules[full_path] = new_module
            if module is not None:
                # Set the new module as an attribute of its parent
                setattr(module, part, new_module)
        module = sys.modules[full_path]
    return module


def add_ttl_operations(root, ttnn_prefix):
    target_module = create_or_get_module(ttnn_prefix)

    for attribute_name in dir(root):
        if attribute_name.startswith("__"):
            continue

        attribute = getattr(root, attribute_name)
        if inspect.ismodule(attribute):
            # Recursive call for submodules with updated ttnn_prefix
            new_ttnn_prefix = f"{ttnn_prefix}.{attribute_name}"
            add_ttl_operations(root=attribute, ttnn_prefix=new_ttnn_prefix)
            continue

        if inspect.isbuiltin(attribute) and (
            "tt_lib.tensor.Tensor" in attribute.__doc__ or "tt::tt_metal::Tensor" in attribute.__doc__
        ):
            attribute = ttnn.decorators.register_ttl_operation_as_ttnn_operation(
                name=f"{ttnn_prefix}.{attribute_name}", function=attribute
            )
        setattr(target_module, attribute_name, attribute)


add_ttl_operations(root=ttl.operations.primary, ttnn_prefix="ttnn.experimental.operations.primary")
