# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""moe_fused_swiglu — one MoE routed-expert block as ONE device program.

    h   = SiLU(x @ W_gate) * (x @ W_up)      # [count, hidden]  INTERNAL, never reaches DRAM
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

from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import (
    TILE,
    WEIGHT_DTYPES,
    weight_memory_configs,
    worker_grid,
)


# ---------------------------------------------------------------------------
# Precision default — ONE exported definition; `None` resolves through it.
# ---------------------------------------------------------------------------
def default_compute_kernel_config() -> ttnn.DeviceComputeKernelConfig:
    """LoFi + approx SFPU + fp16 DEST.

    bfp4 weights carry ~4 mantissa bits, so higher fidelity buys nothing and costs FPU passes;
    fp32 DEST would halve DEST capacity (DEST_AUTO_LIMIT 8 -> 4) for no accuracy gain.
    `dst_full_sync_en=False` is load-bearing: the Blackhole fast tilize path requires half sync.
    """
    # DeviceComputeKernelConfig is the abstract Python base; this concrete
    # configuration type is also used on Blackhole.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — over (x_shape, w_gate_shape, w_up_shape, w_down_shape, count)
# ---------------------------------------------------------------------------
def tag_emb(inputs, axes):
    return int(inputs[0][-1])


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
    "fill": tag_fill,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
SUPPORTED = {
    # The activation's dtype x layout cross, collapsed to the two real combinations.
    "input_format": ["bf16_rm", "bfp8_tile"],
    "weight_dtype": list(WEIGHT_DTYPES),
    "emb": [6144, 7168],
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
    *,
    output=None,
    expert_region_offsets=None,
    read_x_at_offset=False,
    input_m_tiles=None,
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
        # The kernels address weights as a 2-D array of TILE pages at a W_TILE stride; a row-major
        # weight would be read as tiles and silently give wrong numbers.
        if w.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"moe_fused_swiglu: {name} must be TILE_LAYOUT, got {w.layout}")

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
        shape = list(t.shape)
        if not (len(shape) == 1 or (len(shape) == 2 and int(shape[0]) == 1)):
            raise ValueError(f"moe_fused_swiglu: {name} must be 1-D or (1, N), got shape {shape}")
        if int(shape[-1]) == 0:
            raise ValueError(f"moe_fused_swiglu: {name} must not be empty")

    num_local = int(global_expert_idx_table.shape[-1])
    if not isinstance(local_expert_id, int) or local_expert_id < 0 or local_expert_id >= num_local:
        raise ValueError(
            f"moe_fused_swiglu: local_expert_id {local_expert_id} out of range for the idx table "
            f"of length {num_local}"
        )

    # ---- shared-buffer region mode (fused extract / insert) ------------------------
    # Same two knobs as `unified_routed_expert_ffn`: an offsets tensor turns the writer into a
    # direct placement into `output` (fusing ttnn::insert), and `read_x_at_offset` opts the reader
    # into the mirror rebase of its x reads (fusing ttnn::extract).
    if read_x_at_offset and expert_region_offsets is None:
        raise ValueError("moe_fused_swiglu: read_x_at_offset requires expert_region_offsets")
    if expert_region_offsets is not None:
        if output is None:
            raise ValueError(
                "moe_fused_swiglu: direct-write mode (expert_region_offsets set) requires `output` — "
                "the shared destination buffer this expert's rows are placed into"
            )
        s = expert_region_offsets
        if s.dtype != ttnn.uint32 or s.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(
                f"moe_fused_swiglu: expert_region_offsets must be uint32 ROW_MAJOR, got dtype "
                f"{s.dtype} layout {s.layout}"
            )
        shape = list(s.shape)
        if not (len(shape) == 1 or (len(shape) == 2 and int(shape[0]) == 1)):
            raise ValueError(f"moe_fused_swiglu: expert_region_offsets must be 1-D or (1, N), got shape {shape}")
        # The kernel indexes start[global_id] and counts[global_id] out of the SAME global-expert
        # index space, and it reads `start` over the counts scratch page — so equal lengths are both
        # a correctness requirement and what makes that page reuse sound. Mirrors ttnn::insert.
        if int(shape[-1]) != int(counts.shape[-1]):
            raise ValueError(
                f"moe_fused_swiglu: expert_region_offsets length {int(shape[-1])} must equal counts "
                f"length {int(counts.shape[-1])}"
            )

    if output is not None:
        if output.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"moe_fused_swiglu: output must be TILE_LAYOUT, got {output.layout}")
        if output.dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
            raise ValueError(f"moe_fused_swiglu: output dtype {output.dtype!r} must be bfloat8_b or bfloat16")
        if int(output.shape[-1]) != emb:
            raise ValueError(
                f"moe_fused_swiglu: output last dim {int(output.shape[-1])} must equal emb {emb} — "
                f"the writer's tile-row stride IS that number"
            )
        out_m = int(output.shape[-2])
        if out_m % TILE != 0:
            raise ValueError(f"moe_fused_swiglu: output rows {out_m} must be tile-aligned")
        if expert_region_offsets is not None:
            # Direct write targets the larger shared buffer; the device bounds every row against
            # its real tile-row count, so only "at least one region fits" is checkable here.
            if out_m < int(input_tensor.shape[-2]):
                raise ValueError(
                    f"moe_fused_swiglu: output rows {out_m} must be >= input rows "
                    f"{int(input_tensor.shape[-2])} in direct-write mode"
                )
        elif out_m != int(input_tensor.shape[-2]):
            raise ValueError(
                f"moe_fused_swiglu: output rows {out_m} must equal input rows "
                f"{int(input_tensor.shape[-2])} (pass expert_region_offsets for a shared destination)"
            )

    # ---- registry axes ------------------------------------------------------------
    # `fill` derives from a DEVICE-resident value and validate() is host-side and forbidden from
    # reading `counts`, so it is observed-but-uncheckable: SUPPORTED lists all four buckets and
    # the gate below covers the host-visible axes only.
    #
    axes = {
        "input_format": _input_format(input_tensor),
        "weight_dtype": w_gate.dtype,
        "emb": emb,
    }
    for axis, value in axes.items():
        if value not in SUPPORTED[axis]:
            raise UnsupportedAxisValue(f"moe_fused_swiglu: {axis}={value!r} not in SUPPORTED {SUPPORTED[axis]}")
    for name, w in (("w_up", w_up), ("w_down", w_down)):
        if w.dtype != w_gate.dtype:
            raise ValueError(
                f"moe_fused_swiglu: all three weights must share one dtype (the CB format and the "
                f"tile stride are one number); got w_gate={w_gate.dtype!r}, {name}={w.dtype!r}"
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
    compute_kernel_config: ttnn.DeviceComputeKernelConfig = None,
    core_grid=None,
    output: ttnn.Tensor = None,
    expert_region_offsets: ttnn.Tensor = None,
    read_x_at_offset: bool = False,
) -> ttnn.Tensor:
    """`output` / `expert_region_offsets` / `read_x_at_offset` are the three fusion knobs, with the
    same meanings as `ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn`:

      * `output` — a pre-allocated DRAM TILE tensor to write into (no per-call allocation). Rows past
        ceil_tile(count) are NEVER written, so whatever was there stays there.
      * `expert_region_offsets` — the per-global-expert region starts (the `start` vector
        ttnn::insert consumes). Set it and the writer places this expert's rows directly into
        `output` (the shared destination) at start[global_id], fusing the ttnn::insert step. Requires
        `output`.
      * `read_x_at_offset` — x is that same kind of shared buffer, so the reader rebases its x reads
        by start[global_id], fusing ttnn::extract. Requires `expert_region_offsets`, and pair it with
        `input_m_tiles` = ONE region's tile-rows so the op still sizes its grid and chunks to a
        single expert rather than to the whole buffer.

    `output` must not alias `input_tensor`: this op's cross-M-block x prefetch is not ordered against
    the write-back of an earlier block, so in-place would be a read-after-write race.
    """
    validate(
        input_tensor,
        w_gate,
        w_up,
        w_down,
        counts,
        global_expert_idx_table,
        local_expert_id,
        output=output,
        expert_region_offsets=expert_region_offsets,
        read_x_at_offset=read_x_at_offset,
        input_m_tiles=input_m_tiles,
    )

    capacity = int(input_tensor.shape[-2])

    out_dtype = (
        output.dtype if output is not None and dtype is None else (dtype if dtype is not None else ttnn.bfloat8_b)
    )
    if out_dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
        raise ValueError(f"moe_fused_swiglu: output dtype {out_dtype!r} must be bfloat8_b or bfloat16")
    out_memory_config = (
        output.memory_config()
        if output is not None and memory_config is None
        else (memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG)
    )
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    # Compatibility for callers of the former ProgramDescriptor implementation.
    # The standard C++ operation takes DeviceComputeKernelConfig; copy the
    # hardware-relevant fields from the old descriptor when one is supplied.
    if isinstance(cfg, ttnn.ComputeConfigDescriptor):
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=cfg.math_fidelity,
            math_approx_mode=cfg.math_approx_mode,
            fp32_dest_acc_en=cfg.fp32_dest_acc_en,
            packer_l1_acc=False,
            dst_full_sync_en=cfg.dst_full_sync_en,
        )
    m_t_max = input_m_tiles if input_m_tiles is not None else capacity // TILE
    if m_t_max < 1 or m_t_max > capacity // TILE:
        raise ValueError(f"moe_fused_swiglu: input_m_tiles {m_t_max} out of range [1, {capacity // TILE}]")

    if output is not None:
        # `dtype` / `memory_config` describe an allocation this call is not making. Refuse a
        # disagreement rather than silently honouring one of the two.
        if dtype is not None and dtype != output.dtype:
            raise ValueError(
                f"moe_fused_swiglu: dtype={dtype!r} contradicts the supplied output's dtype "
                f"{output.dtype!r} — pass one or the other"
            )
        if memory_config is not None and memory_config != output.memory_config():
            raise ValueError(
                "moe_fused_swiglu: memory_config contradicts the supplied output's memory config — "
                "pass one or the other"
            )
        if output.buffer_address() == input_tensor.buffer_address():
            raise ValueError(
                "moe_fused_swiglu: output must not alias input_tensor — the reader prefetches the "
                "NEXT M-block's x rows with no ordering against this block's write-back"
            )
    if core_grid is not None and not hasattr(core_grid, "x"):
        core_grid = ttnn.CoreCoord(int(core_grid[0]), int(core_grid[1]))

    return ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
        input_tensor,
        w_gate,
        w_up,
        w_down,
        counts,
        global_expert_idx_table,
        local_expert_id,
        input_m_tiles=m_t_max,
        dtype=out_dtype,
        memory_config=out_memory_config,
        compute_kernel_config=cfg,
        core_grid=core_grid,
        output=output,
        expert_region_offsets=expert_region_offsets,
        read_x_at_offset=read_x_at_offset,
    )
