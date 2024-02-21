# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
import sys

import tt_lib as ttl
import ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []

for attribute_name in dir(ttl.tensor):
    if attribute_name.startswith("__"):
        continue
    attribute = getattr(ttl.tensor, attribute_name)
    probably_c_function = inspect.isbuiltin(attribute) or (
        hasattr(attribute, "profiler_wrapped_function") and inspect.isbuiltin(attribute.profiler_wrapped_function)
    )
    if probably_c_function and (
        "tt_lib.tensor.Tensor" in attribute.__doc__ or "tt::tt_metal::Tensor" in attribute.__doc__
    ):
        attribute = ttnn.decorators.register_ttl_operation_as_ttnn_operation(
            name=f"ttnn.experimental.tensor.{attribute_name}", function=attribute
        )
    setattr(THIS_MODULE, attribute_name, attribute)
    __all__.append(attribute_name)
