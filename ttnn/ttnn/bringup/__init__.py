# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup: derived ops of model bring-ups (INDEX.md).

The C++ forks are bound from ttnn._ttnn.operations.bringup and attached here by ttnn's auto-registration. The Python
forks (a ProgramDescriptor op in a subfolder) are listed in PYTHON_OPS: ``ttnn.bringup.<name>`` loads the subfolder on
first use and registers the function as a ttnn operation named ``ttnn.bringup.<name>``, so the profiler and the
fork-call capture see it like any other op. Nothing here imports at ``import ttnn`` time.
"""

# python name -> (fork folder, function in its package). `rms_norm` (rms_norm_ttnn/) has a C++ host side now and binds
# as ttnn.bringup.rms_norm; its Python implementation stays importable as ttnn.bringup.rms_norm_ttnn.rms_norm_ttnn for
# A/B comparison. mhc_pre / mhc_post (mhc_pre_ttnn/, mhc_post_ttnn/) likewise:
# C++ bindings, their Python implementations importable as ttnn.bringup.mhc_pre_ttnn.mhc_pre / .mhc_post_ttnn.mhc_post.
PYTHON_OPS = {}


def _stable_arg_reprs():
    """Value reprs for the non-tensor handles that fork ops take, so the fork-call capture
    (models/demos/common/bringup/testing/fork_capture.py records such arguments by str()) gives the same call signature
    on every run. The default "<... object at 0x...>" changes from run to run (hy4_preview_d_p O.1: the global
    semaphore pair of ring_indexer_score_dsa). Only the repr is set: the type is an opaque handle and its behaviour is
    unchanged. The fork's own IndexerScoreProgramConfig binds its repr in C++ (indexer_score_nanobind.cpp)."""
    import sys

    core = sys.modules.get("ttnn._ttnn")
    sem = getattr(getattr(core, "global_semaphore", None), "global_semaphore", None)
    if sem is not None and sem.__repr__ is object.__repr__:
        sem.__repr__ = lambda self: "global_semaphore"


_stable_arg_reprs()


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
