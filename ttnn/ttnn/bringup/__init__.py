# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup: derived ops of model bring-ups (INDEX.md).

The C++ forks are bound from ttnn._ttnn.operations.bringup and attached here by ttnn's auto-registration. The Python
forks (a ProgramDescriptor op in a subfolder) are listed in PYTHON_OPS: ``ttnn.bringup.<name>`` loads the subfolder on
first use and registers the function as a ttnn operation named ``ttnn.bringup.<name>``, so the profiler and the
fork-call capture see it like any other op. Nothing here imports at ``import ttnn`` time.
"""

# python name -> (fork folder, function in its package): the Python (ProgramDescriptor) forks, no C++ build.
PYTHON_OPS = {}


def _stable_arg_reprs():
    """Value reprs for the non-tensor handles that fork ops take, so the fork-call capture (bringup_fw's
    testing/fork_capture.py records such arguments by str()) gives the same call signature on every run. The default
    "<... object at 0x...>" changes from run to run (the hackathon's hy4_preview_d_p O.1: the global semaphore pair of
    ring_indexer_score_dsa). Only the repr is set: the type is an opaque handle and its behaviour is unchanged."""

    import sys

    core = sys.modules.get("ttnn._ttnn")
    sem = getattr(getattr(core, "global_semaphore", None), "global_semaphore", None)
    if sem is not None and sem.__repr__ is object.__repr__:
        sem.__repr__ = lambda self: "global_semaphore"

    # SDPAProgramConfig's C++ __repr__ raises TypeError (transformer_nanobind.cpp returns a std::string without
    # nanobind/stl/string.h), so the capture's str() of the sdpa fork's program_config failed (kimi_k27_code O.1).
    # A value repr from its fields, set only while the C++ one is broken.
    cfg = getattr(getattr(getattr(core, "operations", None), "transformer", None), "SDPAProgramConfig", None)
    if cfg is not None:
        try:
            repr(cfg())
        except TypeError:
            cfg.__repr__ = cfg.__str__ = _sdpa_program_config_repr


def _sdpa_program_config_repr(c):
    g = c.compute_with_storage_grid_size
    return (
        f"SDPAProgramConfig(compute_with_storage_grid_size=({g.x}, {g.y}), sub_core_grids={c.sub_core_grids}, "
        f"q_chunk_size={c.q_chunk_size}, k_chunk_size={c.k_chunk_size}, exp_approx_mode={c.exp_approx_mode}, "
        f"max_cores_per_head_batch={c.max_cores_per_head_batch}, max_k_splits={c.max_k_splits}, "
        f"matmul_math_fidelity={c.matmul_math_fidelity}, segmented_accumulation={c.segmented_accumulation})"
    )


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
