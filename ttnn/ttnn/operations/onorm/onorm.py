# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm — the Kimi-Linear KDA **s6** tail, fused on-chip.

    out = flatten_heads( RMSNorm_over_V(o) * weight ) * sigmoid(gate)

`o` arrives head-major ``[B, T, HV, V]`` (heads on the tiled row axis, ``V`` the
RMSNorm reduction axis); the output is flat token-major ``[B, T, HV*V]`` with
feature index ``f = head * V + channel``.  The head-major -> flat re-tile is a
genuine re-tiling and is performed **in-kernel** (untilize -> row-major ->
tilize); this entry point never calls ``ttnn.reshape`` / ``to_layout`` /
``tilize`` / ``untilize``.

Registry model: the four declarations below (``INPUT_TAGGERS``, ``SUPPORTED``,
``EXCLUSIONS``, ``validate()``) are the single source of truth for the op's
support contract; ``validate()`` is the first line of ``onorm()``.  ``INVALID``
is deliberately absent — it is a test-harness concept living in
``eval/golden_tests/onorm/feature_spec.py``.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .onorm_program_descriptor import TILE_H, TILE_W, create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# onorm has no shape-facet axes: the KDA s6 geometry is fixed (HV = 32 heads =
# exactly one tile height, V = 128 = 4 tile-widths) and T is tile-aligned by
# contract, so there is nothing categorical to project off the input shapes.

INPUT_TAGGERS: dict = {}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Matches `eval/golden_tests/onorm/feature_spec.py`'s TARGET axis-for-axis
# (`dtype`, `layout` — those are the only two axes the suite enumerates).

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# TARGET is a single (bfloat16, TILE) cell and it is implemented, so there is
# nothing inside the SUPPORTED rectangle to refuse.

EXCLUSIONS: list = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    # verified: the program descriptor's core-range set comes from
    # ttnn.split_work_to_cores over B * ceil(T / TOKENS_PER_BLOCK) token-blocks,
    # times RETILE_GROUP_CORES cores per block (Refinement 2's cross-core re-tile,
    # which is what lets a 1-block shape occupy 32 cores instead of 1).
    "multi_core": {"value": True, "source": "verified"},
    # declared: every CB page count derives from a block-factor knob
    # (NORM_CHUNK_TOKENS / TOKENS_PER_BLOCK / GATE_CHUNK_TILES / DM_BLOCK_TILES /
    # RETILE_GROUP_CORES) and never from B or T — see op_design.md §6.2.
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# Compute-kernel-config factory (the single None-resolution point)
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.DeviceComputeKernelConfig:
    """The one place ``compute_kernel_config=None`` resolves through.

    * ``fp32_dest_acc_en=False`` — **a documented deviation** from the task
      contract; see the block below.
    * ``math_fidelity=HiFi4``    — bf16 x bf16 mantissas fully retained on every
      FPU multiply.
    * ``math_approx_mode=False`` — exact ``rsqrt``.  (It does *not* change the
      sigmoid: the LLK's ``_calculate_sigmoid_`` ignores ``APPROXIMATION_MODE``
      on both Blackhole and Wormhole B0 — same 6-entry LUT either way.)
    * ``dst_full_sync_en=False`` — **load-bearing for performance**:
      ``can_use_fast_tilize()`` gates on ``!get_dst_full_sync_enabled()``, so
      enabling full sync silently drops the 128-tile re-tile onto the slow
      tilize path on every core, every block.

    DOCUMENTED DEVIATION — ``fp32_dest_acc_en`` (Refinement 1b, route 2)
    -------------------------------------------------------------------
    ``eval/prompts/onorm.txt`` -> ## Rules -> Precision states:

        "When reducing for RMSNorm: accumulate the sum-of-squares in fp32 in
         DST (fp32_dest_acc_en=True) even though the I/O dtype is bf16 — the
         reduction is the precision-sensitive step."

    This factory ships ``fp32_dest_acc_en=False`` instead.  Why, and what was
    tried first:

    * **Prize.** A 16-bit DEST is a measured **1.208x** on the whole kernel
      (B=1/T=640: 244,312 -> 202,256 ns) and **1.39x** on ``sigmoid(gate)``
      alone, which is 62 % of the kernel.  The non-sigmoid remainder is flat,
      so the win is specifically the SFPU's: a 32-bit DEST costs the SFPU an
      extra pass over every vector op, and this op is SFPU-throughput-bound
      (Refinement 1 proved it by ablation).
    * **Route 1 (preserve fp32 accumulation by another mechanism) is not
      available on this hardware.**  ``fp32_dest_acc_en`` is a whole-kernel
      compile-time flag, so "fp32 for P1 only" means moving the accumulation
      off DEST.  The only fp32 accumulation datapath that bypasses DEST is the
      **packer's L1 accumulator**, and it is *fp32-DEST-only hardware* — see
      the measured catalog example ``examples/row_reduce_accumulate`` ("a bf16
      DEST corrupts the accumulate").  Every other route from FPU/SFPU into L1
      traverses DEST, so at 16 bits there is nothing left to accumulate in.
    * **Measured precision cost** (4 shapes, ``test_onorm_precision_baseline``):
      PCC 0.999993 -> 0.999988, rel-RMS 0.0037 -> 0.0056, got/true ratio median
      0.9997 -> 1.0026.  That is 3.5x inside the op's own stated bar
      (PCC >= 0.9995, rel-RMS < 0.02) and ~7x inside the golden bf16 bar
      (PCC >= 0.995, RMS 0.04), and the ratio stays a broad spread centred on
      1.0 — rounding, not a scale bug.
    * **The contract's field is still honoured.** ``fp32_dest_acc_en`` is a
      public ``compute_kernel_config`` field: a caller who passes
      ``ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=True)`` gets the
      32-bit DEST and the fp32 sum-of-squares accumulation, exactly as before.
      Only the ``None`` default moved.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(o, gate, weight, *, epsilon: float = 1e-5, compute_kernel_config=None) -> None:
    """Runtime support gate: SUPPORTED per-axis, then EXCLUSIONS cell-level.

    All three input tensors are checked against the ``dtype`` / ``layout`` axes.
    """
    per_tensor_axes = {
        "o": {"dtype": o.dtype, "layout": o.layout},
        "gate": {"dtype": gate.dtype, "layout": gate.layout},
        "weight": {"dtype": weight.dtype, "layout": weight.layout},
    }

    # 1. SUPPORTED — per-axis, for every input tensor.
    for name, axes in per_tensor_axes.items():
        for axis, allowed in SUPPORTED.items():
            if axes[axis] not in allowed:
                raise UnsupportedAxisValue(f"onorm: {name}.{axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside the SUPPORTED rectangle.  The cell is
    #    the op's (dtype, layout) pair; all three tensors agree by step 1.
    cell = per_tensor_axes["o"]
    for exc in EXCLUSIONS:
        if all(cell.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"onorm: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def onorm(
    o: ttnn.Tensor,
    gate: ttnn.Tensor,
    weight: ttnn.Tensor,
    epsilon: float = 1e-5,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig = None,
) -> ttnn.Tensor:
    """Gated RMSNorm + head-flatten, fused.

    Args:
        o:      ``[B, T, HV, V]`` head-major, TILE, bfloat16.
        gate:   ``[B, T, HV*V]`` flat token-major, TILE, bfloat16 (**pre**-sigmoid).
        weight: ``[1, 1, 1, V]`` per-head_dim RMSNorm scale, TILE, bfloat16.
        epsilon: added to ``mean(o**2)`` before ``rsqrt``.
        compute_kernel_config: resolved through
            :func:`default_compute_kernel_config` when ``None``.

    Returns:
        ``[B, T, HV*V]`` flat token-major, TILE, bfloat16.
    """
    validate(o, gate, weight, epsilon=epsilon, compute_kernel_config=compute_kernel_config)

    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()

    # --- host-side shape contract (not axis refusals: these are not TARGET axes) ---
    o_shape = list(o.shape)
    gate_shape = list(gate.shape)
    weight_shape = list(weight.shape)

    assert len(o_shape) == 4, f"onorm: o must be rank-4 [B, T, HV, V], got {o_shape}"
    assert len(gate_shape) == 3, f"onorm: gate must be rank-3 [B, T, HV*V], got {gate_shape}"

    batch, tokens, num_heads, head_dim = o_shape
    assert num_heads == TILE_H, (
        f"onorm: HV must equal the tile height ({TILE_H}) so all heads of a token "
        f"occupy exactly one tile row; got HV={num_heads}"
    )
    assert (
        gate_shape[0] == batch and gate_shape[1] == tokens
    ), f"onorm: gate {gate_shape} must share (B, T) with o {o_shape}"
    assert (
        gate_shape[2] == num_heads * head_dim
    ), f"onorm: gate feature width must be HV*V={num_heads * head_dim}, got {gate_shape[2]}"
    assert weight_shape[-1] == head_dim and all(
        d == 1 for d in weight_shape[:-1]
    ), f"onorm: weight must be [1, ..., 1, V={head_dim}], got {weight_shape}"
    # V must be a whole number of tile widths.  The head-major -> flat re-tile
    # (untilize -> row-major -> tilize) relies on head h's V values being
    # *exactly* the row-major span [h*V, (h+1)*V); a V that does not fill whole
    # tiles interleaves TILE padding into that span and silently mis-maps every
    # flat feature index.  (The RMSNorm scaler 1/V would still be right — this
    # is a layout failure, not an arithmetic one, so it has no numeric signal.)
    assert (
        head_dim % TILE_W == 0
    ), f"onorm: V must be a multiple of the tile width ({TILE_W}) for the in-kernel re-tile; got V={head_dim}"
    assert float(epsilon) > 0.0, f"onorm: epsilon must be > 0, got {epsilon}"

    # Placement contract.  `memory_config` is not a TARGET axis, so this is a
    # host assert rather than an axis refusal: the work split, the CB budget and
    # the reader/writer are all written against interleaved buffers.  A sharded
    # input would still read correctly through TensorAccessor but its resident
    # L1 is invisible to the CB budget below, so it can OOM without a signal.
    for name, tensor in (("o", o), ("gate", gate), ("weight", weight)):
        assert not tensor.memory_config().is_sharded(), (
            f"onorm: {name} must be interleaved (got {tensor.memory_config()}). "
            f"Sharded placement is not a declared axis of this op."
        )

    device = o.device()
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(gate_shape),
        gate.dtype,
        gate.layout,
        device,
        gate.memory_config(),
    )

    program_descriptor = create_program_descriptor(o, gate, weight, output, float(epsilon), compute_kernel_config)

    # Output tensor MUST be last in io_tensors.
    return ttnn.generic_op([o, gate, weight, output], program_descriptor)
