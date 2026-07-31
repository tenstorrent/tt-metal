# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""moe_fused_swiglu — one MoE routed-expert block as ONE device program.

    h   = SiLU(x @ W_gate) * (x @ W_up)      # [count, 2048]  INTERNAL, never reaches DRAM
    out = h @ W_down                          # [capacity, emb]

`count` is DEVICE-resident: the kernels read `counts[idx[local_expert_id]]` themselves; there is
no host readback and no host branch on the counts' contents.

Registry model (eval/op_template.py): INPUT_TAGGERS / SUPPORTED / EXCLUSIONS declared here,
validate() is the entry point's first line. INVALID is NOT declared here — it lives in
eval/golden_tests/moe_fused_swiglu/feature_spec.py.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
    HIDDEN,
    TILE,
    create_program_descriptor,
    make_mailbox,
)


# ---------------------------------------------------------------------------
# Precision default — ONE exported definition; `None` resolves through it.
# ---------------------------------------------------------------------------
def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """LoFi + approx SFPU + fp16 DEST.

    bfp4 weights carry ~4 mantissa bits, so higher fidelity buys nothing and costs FPU passes;
    fp32 DEST would halve DEST capacity (DEST_AUTO_LIMIT 8 -> 4) for no accuracy gain.
    `dst_full_sync_en=False` is load-bearing: the Blackhole fast tilize path requires half sync.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.math_approx_mode = True
    cfg.fp32_dest_acc_en = False
    cfg.dst_full_sync_en = False
    cfg.bfp8_pack_precise = True
    return cfg


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — over (x_shape, w_gate_shape, w_up_shape, w_down_shape, count)
# ---------------------------------------------------------------------------
def tag_emb(inputs, axes):
    return int(inputs[0][-1])


def tag_capacity(inputs, axes):
    return int(inputs[0][-2])


def tag_fill(inputs, axes):
    """VERBATIM the bucket rule of feature_spec.classify_fill."""
    count, capacity = inputs[4], inputs[0][-2]
    if count == 0:
        return "empty"
    if count == capacity:
        return "full"
    if count <= capacity // 16:
        return "balanced"
    return "partial"


INPUT_TAGGERS = {
    "emb": tag_emb,
    "capacity": tag_capacity,
    "fill": tag_fill,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
SUPPORTED = {
    # The activation's dtype x layout cross, collapsed to the two real combinations.
    "input_format": ["bf16_rm", "bfp8_tile"],
    "weight_dtype": [ttnn.bfloat4_b],
    "emb": [6144, 7168],
    "capacity": [1024, 2048, 5120],
    "fill": ["balanced", "partial", "full", "empty"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------
PROPERTIES = {
    "multi_core": {"value": True, "source": "verified"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi"], "source": "declared"},
}


def _input_format(input_tensor):
    if input_tensor.dtype == ttnn.bfloat16 and input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        return "bf16_rm"
    if input_tensor.dtype == ttnn.bfloat8_b and input_tensor.layout == ttnn.TILE_LAYOUT:
        return "bfp8_tile"
    return f"{input_tensor.dtype}/{input_tensor.layout}"


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
def validate(
    input_tensor,
    w_gate,
    w_up,
    w_down,
    counts,
    global_expert_idx_table,
    local_expert_id,
):
    # ---- structural (ValueError) --------------------------------------------------
    if len(input_tensor.shape) != 4:
        raise ValueError(f"moe_fused_swiglu: input_tensor must have rank 4, got rank {len(input_tensor.shape)}")
    if int(input_tensor.shape[0]) != 1 or int(input_tensor.shape[1]) != 1:
        raise ValueError(
            "moe_fused_swiglu: input_tensor leading dims must be (1, 1), got "
            f"({int(input_tensor.shape[0])}, {int(input_tensor.shape[1])})"
        )
    for name, w in (("w_gate", w_gate), ("w_up", w_up), ("w_down", w_down)):
        if len(w.shape) != 2:
            raise ValueError(f"moe_fused_swiglu: {name} must have rank 2, got rank {len(w.shape)}")

    emb = int(input_tensor.shape[-1])
    if int(w_gate.shape[-2]) != emb:
        raise ValueError(
            f"moe_fused_swiglu: w_gate contraction dim {int(w_gate.shape[-2])} does not match "
            f"input_tensor emb {emb}"
        )
    if list(w_gate.shape) != list(w_up.shape):
        raise ValueError(
            f"moe_fused_swiglu: w_gate shape {list(w_gate.shape)} and w_up shape "
            f"{list(w_up.shape)} must be identical"
        )
    if int(w_down.shape[-2]) != int(w_gate.shape[-1]):
        raise ValueError(
            f"moe_fused_swiglu: w_down contraction dim {int(w_down.shape[-2])} does not match the "
            f"hidden dim {int(w_gate.shape[-1])} produced by w_gate"
        )
    if int(w_down.shape[-1]) != emb:
        raise ValueError(f"moe_fused_swiglu: w_down output dim {int(w_down.shape[-1])} must equal emb {emb}")
    hidden = int(w_gate.shape[-1])
    if hidden % TILE != 0 or emb % TILE != 0:
        raise ValueError(f"moe_fused_swiglu: emb {emb} and hidden {hidden} must be tile-aligned")
    if int(input_tensor.shape[-2]) % TILE != 0:
        raise ValueError(f"moe_fused_swiglu: capacity {int(input_tensor.shape[-2])} must be tile-aligned")

    for name, t in (("counts", counts), ("global_expert_idx_table", global_expert_idx_table)):
        if t.dtype != ttnn.uint32:
            raise ValueError(f"moe_fused_swiglu: {name} must be uint32 ROW_MAJOR, got dtype {t.dtype}")
        if t.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"moe_fused_swiglu: {name} must be uint32 ROW_MAJOR, got layout {t.layout}")

    num_local = int(global_expert_idx_table.shape[-1])
    if not isinstance(local_expert_id, int) or local_expert_id < 0 or local_expert_id >= num_local:
        raise ValueError(
            f"moe_fused_swiglu: local_expert_id {local_expert_id} out of range for the idx table "
            f"of length {num_local}"
        )

    # ---- registry axes ------------------------------------------------------------
    # `fill` derives from a DEVICE-resident value and validate() is host-side and forbidden from
    # reading `counts`, so it is observed-but-uncheckable: SUPPORTED lists all four buckets and
    # the gate below covers the host-visible axes only.
    axes = {
        "input_format": _input_format(input_tensor),
        "weight_dtype": w_gate.dtype,
        "emb": emb,
        "capacity": int(input_tensor.shape[-2]),
    }
    for axis, value in axes.items():
        if value not in SUPPORTED[axis]:
            raise UnsupportedAxisValue(f"moe_fused_swiglu: {axis}={value!r} not in SUPPORTED {SUPPORTED[axis]}")
    for name, w in (("w_up", w_up), ("w_down", w_down)):
        if w.dtype not in SUPPORTED["weight_dtype"]:
            raise UnsupportedAxisValue(
                f"moe_fused_swiglu: weight_dtype={w.dtype!r} ({name}) not in SUPPORTED " f"{SUPPORTED['weight_dtype']}"
            )

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"moe_fused_swiglu: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def moe_fused_swiglu(
    input_tensor: ttnn.Tensor,
    w_gate: ttnn.Tensor,
    w_up: ttnn.Tensor,
    w_down: ttnn.Tensor,
    counts: ttnn.Tensor,
    global_expert_idx_table: ttnn.Tensor,
    local_expert_id: int,
    *,
    input_m_tiles: int = None,
    dtype: ttnn.DataType = None,
    memory_config: ttnn.MemoryConfig = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(input_tensor, w_gate, w_up, w_down, counts, global_expert_idx_table, local_expert_id)

    device = input_tensor.device()
    capacity = int(input_tensor.shape[-2])
    emb = int(input_tensor.shape[-1])

    out_dtype = dtype if dtype is not None else ttnn.bfloat8_b
    out_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    m_t_max = input_m_tiles if input_m_tiles is not None else capacity // TILE
    if m_t_max < 1 or m_t_max > capacity // TILE:
        raise ValueError(f"moe_fused_swiglu: input_m_tiles {m_t_max} out of range [1, {capacity // TILE}]")

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, capacity, emb]),
        out_dtype,
        ttnn.TILE_LAYOUT,
        device,
        out_memory_config,
    )

    grid = device.compute_with_storage_grid_size()
    mailbox = make_mailbox(device, int(grid.x) * int(grid.y))

    descriptor = create_program_descriptor(
        input_tensor,
        w_gate,
        w_up,
        w_down,
        counts,
        global_expert_idx_table,
        output_tensor,
        mailbox,
        local_expert_id=local_expert_id,
        input_m_tiles=m_t_max,
        compute_kernel_config=cfg,
    )

    ttnn.generic_op(
        [
            input_tensor,
            w_gate,
            w_up,
            w_down,
            counts,
            global_expert_idx_table,
            mailbox,
            output_tensor,
        ],
        descriptor,
    )
    return output_tensor
