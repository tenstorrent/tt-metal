# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""permute — reorder tensor dimensions (torch.permute semantics).

Phase 0 realizes the `whole_tile_relocation` regime of `op_design.md`: TILE
layout, fp32, tile-aligned, rank 4, interleaved DRAM, and a `dims` that keeps
the innermost two dims in place, so every 32x32 tile moves intact. One native
dispatch: reader (NoC0) -> cb_tiles -> writer (NoC1).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .permute_program_descriptor import create_program_descriptor

TILE_DIM = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (shape-derived only)
# ---------------------------------------------------------------------------
def tag_alignment(inputs, axes):
    shape = inputs[0]
    w_ok = shape[-1] % TILE_DIM == 0
    h_ok = shape[-2] % TILE_DIM == 0 if len(shape) >= 2 else True
    if w_ok and h_ok:
        return "tile_aligned"
    if not w_ok:
        return "w_non_aligned"
    return "h_non_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED  (Phase 0 rectangle — one entry per TARGET axis)
# ---------------------------------------------------------------------------
SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [4],
    "swap_hw": [False],
    "mem": ["dram_interleaved"],
}

# Validate-only gate (deliberately NOT an entry in SUPPORTED — the harness never
# generates it, and an extra SUPPORTED axis would make every generated cell read
# as "unsupported"). A permutation that moves a non-innermost axis into the tiled
# H position is a *retile*, not whole-tile relocation, so it must be refused
# rather than silently produce wrongly re-tiled data.
SUPPORTED_INNER_PAIR = ["preserved"]

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
EXCLUSIONS = []

PROPERTIES = {
    "multi_core": {"value": True, "source": "verified"},
    "bounded_cb": {"value": True, "source": "declared"},
}


def _is_sharded(memory_config):
    if memory_config is None:
        return False
    is_sharded = getattr(memory_config, "is_sharded", None)
    try:
        return bool(is_sharded()) if callable(is_sharded) else bool(is_sharded)
    except Exception:
        return False


def _canonical_dims(dims, rank):
    """Canonicalize integer-index axes to non-negative form (dim -1 == rank-1)."""
    canon = tuple(int(d) % rank for d in dims)
    if sorted(canon) != list(range(rank)):
        raise ValueError(f"permute: dims={dims!r} is not a permutation of range({rank})")
    return canon


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
def validate(input_tensor, dims, *, memory_config=None):
    shape = list(input_tensor.shape)
    rank = len(shape)
    canon = _canonical_dims(dims, rank)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "swap_hw": bool(canon[-1] != rank - 1),
        "mem": "l1_sharded" if _is_sharded(memory_config) else "dram_interleaved",
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"permute: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    inner_pair = "preserved" if (rank >= 2 and canon[-2] == rank - 2) else "moved"
    if inner_pair not in SUPPORTED_INNER_PAIR:
        raise UnsupportedAxisValue(
            f"permute: inner_pair={inner_pair!r} not in SUPPORTED {SUPPORTED_INNER_PAIR} "
            "(dims moves a non-innermost axis into the tiled H position — retile regime)"
        )

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"permute: unsupported combination (refinement candidate): {exc}")

    return canon


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def permute(
    input_tensor: ttnn.Tensor,
    dims,
    *,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    canon = validate(input_tensor, dims, memory_config=memory_config)

    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    in_shape = list(input_tensor.shape)
    out_shape = [in_shape[d] for d in canon]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        out_mem,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, canon)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
