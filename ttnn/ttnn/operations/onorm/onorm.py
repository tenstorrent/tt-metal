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
    # ttnn.split_work_to_cores over B * ceil(T / TOKENS_PER_BLOCK) work units.
    "multi_core": {"value": True, "source": "verified"},
    # declared: every CB page count derives from a block-factor knob
    # (NORM_CHUNK_TOKENS / TOKENS_PER_BLOCK / GATE_CHUNK_TILES / DM_BLOCK_TILES)
    # and never from B or T — see op_design.md §6.2.
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# Compute-kernel-config factory (the single None-resolution point)
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.DeviceComputeKernelConfig:
    """The one place ``compute_kernel_config=None`` resolves through.

    * ``fp32_dest_acc_en=True``  — the sum-of-squares accumulates in fp32 DEST;
      that is the precision-sensitive step of the op.
    * ``math_fidelity=HiFi4``    — bf16 x bf16 mantissas fully retained on every
      FPU multiply.  The op is DRAM-bound, so the extra FPU passes are free.
    * ``math_approx_mode=False`` — exact ``rsqrt`` / ``sigmoid``.
    * ``dst_full_sync_en=False`` — **load-bearing for performance**:
      ``can_use_fast_tilize()`` gates on ``!get_dst_full_sync_enabled()``, so
      enabling full sync silently drops the 128-tile re-tile onto the slow
      tilize path on every core, every block.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
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
