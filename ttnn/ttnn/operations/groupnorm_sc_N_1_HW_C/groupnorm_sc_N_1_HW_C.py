# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""groupnorm_sc_N_1_HW_C — single-core GroupNorm over (N, 1, H*W, C).

y[n,0,s,c] = (x[n,0,s,c] - mean(n,g)) * rstd(n,g) * gamma[c] + beta[c]
with g = c // (C / num_groups), mean/rstd over the H*W x (C/G) group slab.

Registry model (see eval/op_template.py): INPUT_TAGGERS, SUPPORTED,
EXCLUSIONS, validate(). INVALID lives test-side in feature_spec.py.
"""

from __future__ import annotations

import ttnn

from .groupnorm_sc_N_1_HW_C_program_descriptor import create_program_descriptor

TILE_DIM = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """tile_aligned | hw_non_aligned | c_non_aligned for (N, 1, HW, C)."""
    shape = inputs[0]
    HW, C = shape[-2], shape[-1]
    if C % TILE_DIM != 0:
        return "c_non_aligned"
    if HW % TILE_DIM != 0:
        return "hw_non_aligned"
    return "tile_aligned"


def tag_groups_alignment(inputs, axes):
    """aligned iff per-group channel count (C / num_groups) is a tile multiple.

    Reads the sibling axis `num_groups` (per-shape extra). For G that does
    not divide C the cell is rejected with ValueError before the registry
    gate fires; report non_aligned defensively here.
    """
    C = inputs[0][-1]
    G = axes["num_groups"]
    if G < 1 or C % G != 0:
        return "non_aligned"
    return "aligned" if (C // G) % TILE_DIM == 0 else "non_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "groups_alignment": tag_groups_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned"],
    "groups_alignment": ["aligned"],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    "affine_dtype": [ttnn.bfloat16],
    # TILE-given gamma/beta flow through the same host-side to_layout path
    # as ROW_MAJOR (kernel always sees TILE) — both layouts supported.
    "affine_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, num_groups, *, gamma=None, beta=None, eps=1e-5):
    # --- argument validation: ValueError, before registry gates -----------
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: input must be rank 4 (N, 1, HW, C), got rank {len(shape)}")
    N, one, HW, C = shape
    if one != 1:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: dim 1 must be 1, got {one}")
    if num_groups < 1:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: num_groups must be >= 1, got {num_groups}")
    if C % num_groups != 0:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: C={C} not divisible by num_groups={num_groups}")
    if eps <= 0:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: eps must be > 0, got {eps}")
    for name, t in (("gamma", gamma), ("beta", beta)):
        if t is not None and list(t.shape) != [1, 1, 1, C]:
            raise ValueError(f"groupnorm_sc_N_1_HW_C: {name} shape must be (1, 1, 1, {C}), got {list(t.shape)}")
    if beta is not None and gamma is None:
        raise NotImplementedError("groupnorm_sc_N_1_HW_C: beta without gamma is not supported")

    # --- registry gates: NotImplementedError ------------------------------
    if gamma is not None and beta is not None:
        affine = "gamma_beta"
    elif gamma is not None:
        affine = "gamma_only"
    else:
        affine = "no_affine"

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "num_groups": num_groups,
        "affine": affine,
        # no_affine canonical cell: (bfloat16, ROW_MAJOR_LAYOUT)
        "affine_dtype": gamma.dtype if gamma is not None else ttnn.bfloat16,
        "affine_layout": gamma.layout if gamma is not None else ttnn.ROW_MAJOR_LAYOUT,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise NotImplementedError(f"groupnorm_sc_N_1_HW_C: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise NotImplementedError(f"groupnorm_sc_N_1_HW_C: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def groupnorm_sc_N_1_HW_C(
    input_tensor: ttnn.Tensor,
    num_groups: int,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Single-core GroupNorm over (N, 1, H*W, C). Output is always TILE_LAYOUT."""
    validate(input_tensor, num_groups, gamma=gamma, beta=beta, eps=eps)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Kernel always sees TILE — RM input and RM gamma/beta convert host-side.
    x = input_tensor
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    g = gamma
    if g is not None and g.layout != ttnn.TILE_LAYOUT:
        g = ttnn.to_layout(g, ttnn.TILE_LAYOUT)
    b = beta
    if b is not None and b.layout != ttnn.TILE_LAYOUT:
        b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(x, g, b, output_tensor, num_groups, eps)

    io_tensors = [x]
    if g is not None:
        io_tensors.append(g)
    if b is not None:
        io_tensors.append(b)
    io_tensors.append(output_tensor)  # output MUST be last
    return ttnn.generic_op(io_tensors, program_descriptor)
