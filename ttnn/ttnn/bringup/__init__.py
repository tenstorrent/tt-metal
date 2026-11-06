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
PYTHON_OPS = {
    # fabric reduce-scatter (Python ProgramDescriptor op, mstaletovic/mimo-v2-dp); its planning code is the Python
    # fabric_all_gather in fabric_reduce_scatter_ttnn/fabric_all_gather_py.py (not a registered op)
    "fabric_reduce_scatter": ("fabric_reduce_scatter_ttnn", "fabric_reduce_scatter"),
    # bf16 TILE rows -> 320 B bfloat4_b / 576 B bfloat8_b row records, SFPU-rounded to nearest even (deepseek_v41_flash_d_p
    # compressed / window KV)
    "ckv_bfp4": ("ckv_bfp4_ttnn", "ckv_bfp4"),
    # bf16 TILE -> bfloat4_b / bfloat8_b TILE rounded to nearest even (deepseek_v41_flash_d_p index keys / query)
    "bfp_round": ("ckv_bfp4_ttnn", "bfp_round"),
    # [kv | gate] row pairs -> softmax-gated pair pool, fp32 (deepseek_v41_flash_d_p compressor, P.24 TINY_KERNELS)
    "pair_pool": ("pair_pool_ttnn", "pair_pool"),
}

# Retired forks (INDEX.md, "Retired forks"): python name -> what replaced it. The fork's folder and binding are gone; a
# model's committed fork-call capture (bringup/results/fork_calls.json) may still list calls to it until that model is
# captured again, and models/demos/common/bringup/testing/fork_cases.py reports such calls as retired instead of
# stopping on an op no fork binds.
RETIRED_OPS = {
    # deepseek_v41_flash_d_p's sqrtsoftplus router (fp32_selection + fp32_activation): upstream #60550 made the stock op
    # select in fp32 with its own fp32-accurate sqrt(softplus) (owner decision 2026-11-06)
    "moe_grouped_topk": "ttnn.experimental.deepseek_prefill.moe_grouped_topk (upstream #60550; retired 2026-11-06)",
    # glm53_flash_d_p / deepseek_v41_flash_d_p's fabric all-gather (mstaletovic/mimo-v2-dp): upstream #61387 rewrote the
    # stock op on the same program; same signature and validation, outputs bit-identical (owner decision 2026-11-06)
    "fabric_all_gather": "ttnn.experimental.high_bw_all_gather (upstream #61387; retired 2026-11-06)",
}


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
    # deepseek_v41_flash_d_p O.1: flat_routed_expert_with_program(extra=<the fused combine program>) gave a new
    # signature per call and per run (one per layer, keyed on the object's address). The value: how many programs
    # and the kernel count of each (the same for every layer's combine; an empty descriptor on a cache hit).
    mpd = getattr(getattr(core, "program_descriptor", None), "MeshProgramDescriptor", None)
    if mpd is not None and mpd.__repr__ is object.__repr__:
        mpd.__repr__ = lambda self: (
            "MeshProgramDescriptor(kernels=" + str([len(pd.kernels) for _, pd in iter(self)]) + ")"
        )


_stable_arg_reprs()


def __getattr__(name):
    if name in RETIRED_OPS:
        raise AttributeError(f"ttnn.bringup.{name} is a retired fork: use {RETIRED_OPS[name]}")
    if name not in PYTHON_OPS:
        raise AttributeError(f"module 'ttnn.bringup' has no attribute {name!r}")
    import importlib

    import ttnn

    folder, fn = PYTHON_OPS[name]
    function = getattr(importlib.import_module(f"{__name__}.{folder}"), fn)
    op = ttnn.register_python_operation(name=f"ttnn.bringup.{name}")(function)
    globals()[name] = op
    return op
