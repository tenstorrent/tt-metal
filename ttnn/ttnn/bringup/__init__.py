# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup: derived ops of model bring-ups (INDEX.md).

The C++ forks are bound from ttnn._ttnn.operations.bringup and attached here by ttnn's auto-registration. The Python
forks (a ProgramDescriptor op in a subfolder) are listed in PYTHON_OPS: ``ttnn.bringup.<name>`` loads the subfolder on
first use and registers the function as a ttnn operation named ``ttnn.bringup.<name>``, so the profiler and the
fork-call capture see it like any other op. Nothing here imports at ``import ttnn`` time.
"""

# python name -> (fork folder, function in its package).  Empty: `rms_norm` (rms_norm_ttnn/) has a C++ host side
# now and binds as ttnn.bringup.rms_norm; its Python implementation stays importable as
# ttnn.bringup.rms_norm_ttnn.rms_norm_ttnn for A/B comparison.
PYTHON_OPS = {}


def __getattr__(name):
    if name not in PYTHON_OPS:
        raise AttributeError(f"module 'ttnn.bringup' has no attribute {name!r}")
    import importlib

    import ttnn

    folder, fn = PYTHON_OPS[name]
    function = getattr(importlib.import_module(f"{__name__}.{folder}"), fn)
    op = ttnn.register_python_operation(name=f"ttnn.bringup.{name}")(function)
    globals()[name] = op
    return op
