# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""paged_scaled_dot_product_attention_decode: one decode step of attention over a paged K/V cache.

Registry-model op file (eval/op_template.py): INPUT_TAGGERS (the specification's AXES), SUPPORTED,
EXCLUSIONS, validate(), and the public entry point, which dispatches exactly one `ttnn.generic_op`
(a Metal 2.0 ProgramSpec, the only host program Quasar runs). The program itself is built in
paged_scaled_dot_product_attention_decode_program_artifacts.py; the design is op_design.md.
"""

from __future__ import annotations

import math

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory as _memory
from eval.specs.paged_scaled_dot_product_attention_decode.spec import AXES, canonical_memory

from .paged_scaled_dot_product_attention_decode_program_artifacts import create_program_artifacts

OP_NAME = "paged_scaled_dot_product_attention_decode"

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (the specification's PHASE0).
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True, False],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["L1"],
    "sharding":         ["height"],
    "output_memory":    ["DRAM"],
    "kv_heads_mode":    ["grouped"],
    "batch":            ["single"],
    "alignment":        ["tile_aligned"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# A user's heads are the rows of one tile (rejection X8); a block of the cache is whole tiles (X9).
MAX_HEADS = 32
TILE = 32


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R8): the one place the default lives."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_compute_config(compute_kernel_config):
    """(math_fidelity, fp32_dest_acc_en, math_approx_mode) read as attributes (rule R10): a
    ttnn.ComputeConfigDescriptor or tt-metal's device config (WormholeComputeKernelConfig) alike;
    the fields a descriptor lacks (packer_l1_acc, dst_full_sync_en, throttle_level) are ignored."""
    default = default_compute_kernel_config()  # the only place the default's values live (rule R8)
    config = compute_kernel_config if compute_kernel_config is not None else default
    return (
        getattr(config, "math_fidelity", default.math_fidelity),
        bool(getattr(config, "fp32_dest_acc_en", default.fp32_dest_acc_en)),
        bool(getattr(config, "math_approx_mode", default.math_approx_mode)),
    )


def _inputs_tuple(input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, cur_pos_tensor, scale):
    """The INPUTS-entry form the specification's axis functions read."""
    return (
        tuple(input_tensor_q.shape),
        tuple(input_tensor_k.shape),
        tuple(input_tensor_v.shape),
        tuple(page_table_tensor.shape),
        tuple(cur_pos_tensor.shape),
        scale,
    )


def _check_well_formed(q, k, v, page_table, cur_pos):
    """REJECTIONS X2-X9: malformed calls, refused with ValueError before the support gate."""
    qs, ks, vs, ps, cs = (tuple(t.shape) for t in (q, k, v, page_table, cur_pos))
    if len(qs) != 4 or qs[0] != 1:
        raise ValueError(f"{OP_NAME}: input_tensor_q must be [1, B, NH, D], got {list(qs)}")
    users, heads, head_dim = qs[1], qs[2], qs[3]
    if len(ks) != 4:
        raise ValueError(f"{OP_NAME}: input_tensor_k must be [blocks, NKV, block_size, D], got {list(ks)}")
    if ks != vs:
        raise ValueError(f"{OP_NAME}: the key cache {list(ks)} and the value cache {list(vs)} differ in shape")
    if ks[3] != head_dim:
        raise ValueError(f"{OP_NAME}: the caches' head dimension {ks[3]} differs from the queries' {head_dim}")
    if len(ps) != 2 or ps[0] != users:
        raise ValueError(f"{OP_NAME}: page_table_tensor must be [B={users}, P], got {list(ps)}")
    if len(cs) != 1 or cs[0] != users:
        raise ValueError(f"{OP_NAME}: cur_pos_tensor must be [B={users}], got {list(cs)}")
    kv_heads = ks[1]
    if kv_heads == 0 or heads % kv_heads != 0:
        raise ValueError(f"{OP_NAME}: NH={heads} query heads is not a multiple of NKV={kv_heads} key/value heads")
    if heads > MAX_HEADS:
        raise ValueError(f"{OP_NAME}: NH={heads} query heads; a user's heads are the rows of one tile (<= {MAX_HEADS})")
    if ks[2] % TILE != 0:
        raise ValueError(f"{OP_NAME}: block_size={ks[2]} is not a multiple of {TILE}: a block is whole tiles")


def _check_placements(q, k, v, page_table, cur_pos):
    """Rule R13: the caches DRAM-interleaved TILE of q's dtype; page table and positions DRAM-interleaved
    int32 ROW_MAJOR. Anything else is refused by name with NotImplementedError."""
    dram = {"memory_config": "DRAM", "sharding": "interleaved"}
    for name, t in (("input_tensor_k", k), ("input_tensor_v", v)):
        placed = _memory.memory_axes(t)
        if placed != dram or t.layout != ttnn.TILE_LAYOUT or t.dtype != q.dtype:
            raise NotImplementedError(
                f"{OP_NAME}: {name} must be DRAM-interleaved, TILE layout, dtype {q.dtype} "
                f"(got {placed}, {t.layout}, {t.dtype})")
    for name, t in (("page_table_tensor", page_table), ("cur_pos_tensor", cur_pos)):
        placed = _memory.memory_axes(t)
        if placed != dram or t.layout != ttnn.ROW_MAJOR_LAYOUT or t.dtype != ttnn.int32:
            raise NotImplementedError(
                f"{OP_NAME}: {name} must be DRAM-interleaved, ROW_MAJOR, int32 "
                f"(got {placed}, {t.layout}, {t.dtype})")


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    page_table_tensor,
    *,
    cur_pos_tensor,
    scale=None,
    sliding_window_size=None,
    memory_config=None,
    program_config=None,
    compute_kernel_config=None,
):
    """The runtime gate: the rejections (X1-X10, R13), then SUPPORTED per axis, then EXCLUSIONS.
    `program_config` is a hint and is not read (rule R11)."""
    if sliding_window_size is not None:
        raise NotImplementedError(f"{OP_NAME}: sliding_window_size={sliding_window_size!r}: only None (rule R15)")
    output_memory = canonical_memory(memory_config, None)  # TypeError for a non-config (X10)
    _check_well_formed(input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, cur_pos_tensor)

    _, fp32_dest_acc_en, _ = _resolve_compute_config(compute_kernel_config)
    if input_tensor_q.dtype == ttnn.float32 and not fp32_dest_acc_en:
        raise UnsupportedAxisValue(f"{OP_NAME}: float32 with fp32_dest_acc_en=False is lossy and refused (rule R7)")

    placed = _memory.memory_axes(input_tensor_q)
    axes = {
        "dtype": input_tensor_q.dtype,
        "fp32_dest_acc_en": fp32_dest_acc_en,
        "layout": input_tensor_q.layout,
        "memory_config": placed["memory_config"],
        "sharding": placed["sharding"],
        "output_memory": output_memory,
    }
    inputs = _inputs_tuple(
        input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, cur_pos_tensor, scale)
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(key) == value for key, value in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")

    _check_placements(input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, cur_pos_tensor)
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def paged_scaled_dot_product_attention_decode(
    input_tensor_q: ttnn.Tensor,
    input_tensor_k: ttnn.Tensor,
    input_tensor_v: ttnn.Tensor,
    page_table_tensor: ttnn.Tensor,
    *,
    cur_pos_tensor: ttnn.Tensor,
    scale: float = None,
    sliding_window_size: int = None,
    memory_config: "ttnn.MemoryConfig" = None,
    program_config=None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        page_table_tensor,
        cur_pos_tensor=cur_pos_tensor,
        scale=scale,
        sliding_window_size=sliding_window_size,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    math_fidelity, fp32_dest_acc_en, math_approx_mode = _resolve_compute_config(compute_kernel_config)
    head_dim = input_tensor_q.shape[-1]
    scale_value = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)

    device = input_tensor_q.device()
    output = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(input_tensor_q.shape, input_tensor_q.dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        device,
    )
    io_tensors, spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        page_table_tensor,
        cur_pos_tensor,
        output,
        scale=scale_value,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=math_approx_mode,
    )
    ttnn.generic_op(io_tensors, spec, run_args, tensor_indices)
    return output
