# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""groupnorm_sc_N_1_HW_C — registry-model op file (see eval/op_template.py).

GroupNorm over an ``(N, 1, HW, C)`` channel-last tensor with per-(image, group)
statistics over ``HW x C/G``, centered two-pass variance and optional
per-channel affine. Groups may straddle 32-channel tile boundaries from
Phase 0: the kernel never reduces over channel lanes with a tile reduce — it
forms per-channel column sums and aggregates them by group with a 0/1
membership matmul built on-device from runtime args.

Four registry declarations (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate)
plus the public entry point. INVALID lives in
eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py, not here.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import config
from .groupnorm_sc_N_1_HW_C_program_descriptor import create_program_descriptor, default_compute_kernel_config

_OP = "groupnorm_sc_N_1_HW_C"


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """(N, 1, HW, C): HW is dim -2, C is dim -1. C wins when both are off."""
    shape = inputs[0]
    HW, C = shape[-2], shape[-1]
    if HW % 32 == 0 and C % 32 == 0:
        return "tile_aligned"
    if C % 32 != 0:
        return "c_non_aligned"
    return "hw_non_aligned"


def tag_groups_alignment(inputs, axes):
    """Whole-tile groups vs groups straddling 32-channel tile boundaries.

    Both values are SUPPORTED from Phase 0 — this axis never gates the
    partial-channel case; it exists so the golden harness can bucket cells.
    """
    C = inputs[0][-1]
    G = axes["num_groups"]
    return "group_aligned" if (C // G) % 32 == 0 else "group_straddling"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "groups_alignment": tag_groups_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    # Refinement 4: fp32 and bf8b activations. The input / output CBs take the tensor dtype
    # (ttnn.tile_size(dtype) is the page-size source of truth); every intermediate stays fp32.
    # bf8b is TILE-only (bf8b + ROW_MAJOR is INVALID in feature_spec.py).
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    # BLOCK_SHARDED (Refinement 1): the L1 shard is the per-core block — consumed in place through
    # a CB placed on the shard buffer (TILE) or staged stick-by-stick from it (ROW_MAJOR); the
    # output is the output shard, or the input shard itself when in_place.
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.BLOCK_SHARDED],
    "in_place": [False, True],
    # HW/C tile alignment (Refinement 2): a ragged last tile-row is masked in pass 2 through the
    # expansion matmul (masked group-mean tiles), a ragged last channel block is handled by the
    # per-core (c0, c_valid) valid-lane pair (zero membership rows / affine lanes, RM sticks read
    # and written at their valid byte count). Orthogonal to per-group channel alignment.
    "alignment": ["tile_aligned", "hw_non_aligned", "c_non_aligned"],
    # Partial channels are NOT gated: both values supported from Phase 0.
    "groups_alignment": ["group_aligned", "group_straddling"],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    # "none" = no weight tensor (always legal — the canonical no_affine cell). Refinement 4: fp32 and
    # bf8b weights (bf8b decoded lane by lane into a bf16 rows CB by the reader) and TILE-layout
    # weights (the reader lane-gathers row 0 of weight tile c/32 instead of slicing the stick).
    "affine_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, "none"],
    "affine_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, "none"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # in_place writes the result into the input's L1 shard; an interleaved DRAM input has no
    # resident block to write over (also INVALID in feature_spec.py — the harness never emits it).
    {"memory_layout": ttnn.TensorMemoryLayout.INTERLEAVED, "in_place": True},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _build_axes(input_tensor, num_groups, gamma, beta, in_place):
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_layout": input_tensor.memory_config().memory_layout,
        "in_place": bool(in_place),
        "num_groups": num_groups,
    }
    if gamma is not None and beta is not None:
        axes["affine"] = "gamma_beta"
    elif gamma is not None:
        axes["affine"] = "gamma_only"
    else:
        axes["affine"] = "no_affine"
    weight = gamma if gamma is not None else beta
    axes["affine_dtype"] = weight.dtype if weight is not None else "none"
    axes["affine_layout"] = weight.layout if weight is not None else "none"
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)
    return axes


def validate(input_tensor, num_groups, *, gamma=None, beta=None, in_place=False):
    """Registry gate: SUPPORTED per-axis, then EXCLUSIONS cell-level."""
    axes = _build_axes(input_tensor, num_groups, gamma, beta, in_place)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{_OP}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{_OP}: unsupported combination (refinement candidate): {exc}")


def _validate_arguments(input_tensor, num_groups, gamma, beta, eps, compute_kernel_config):
    """Argument validation (ValueError) — runs after the registry gate."""
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"{_OP}: input must be 4D (N, 1, H*W, C), got rank {len(shape)}")
    if shape[1] != 1:
        raise ValueError(f"{_OP}: input dim[1] must be 1 for the (N, 1, H*W, C) layout, got {shape[1]}")
    C = shape[-1]
    if not isinstance(num_groups, int) or num_groups < 1:
        raise ValueError(f"{_OP}: num_groups must be a positive int, got {num_groups!r}")
    if C % num_groups != 0:
        raise ValueError(f"{_OP}: C={C} is not divisible by num_groups={num_groups}")
    for name, w in (("gamma", gamma), ("beta", beta)):
        if w is None:
            continue
        wshape = list(w.shape)
        if wshape != [1, 1, 1, C]:
            raise ValueError(f"{_OP}: {name} must have shape (1, 1, 1, {C}), got {tuple(wshape)}")
    if gamma is not None and beta is not None:
        if gamma.dtype != beta.dtype or gamma.layout != beta.layout:
            raise ValueError(f"{_OP}: gamma and beta must share dtype and layout")
    if eps <= 0:
        raise ValueError(f"{_OP}: eps must be > 0, got {eps}")
    # Refinement 4: compute_kernel_config. The statistics path (column sums, membership / combine /
    # expansion matmuls, centered squares) accumulates in fp32 DEST into fp32 CBs — with a 16-bit
    # DEST the group mean / variance would round at 2^-8 (op_design.md "Never store stat CBs as
    # bf16") and the matmul_block fidelity rule (#38306) would no longer be documented-correct, so
    # fp32_dest_acc_en=False is refused rather than silently degraded. packer_l1_acc does not compose
    # with fp32 DEST (#28800). math_fidelity / math_approx_mode / dst_full_sync_en pass through.
    for name in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en"):
        if not hasattr(compute_kernel_config, name):
            raise ValueError(
                f"{_OP}: compute_kernel_config must be a ttnn.WormholeComputeKernelConfig (missing {name})"
            )
    if not compute_kernel_config.fp32_dest_acc_en:
        raise ValueError(f"{_OP}: compute_kernel_config.fp32_dest_acc_en must be True (fp32 statistics path)")
    if compute_kernel_config.packer_l1_acc:
        raise ValueError(f"{_OP}: compute_kernel_config.packer_l1_acc is not supported (fp32 DEST accumulation)")


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
    in_place: bool = False,
    compute_kernel_config=None,
    activation: str = None,
    channel_rounds: int = None,
) -> ttnn.Tensor:
    """GroupNorm over an (N, 1, H*W, C) tensor; one native device dispatch.

    `compute_kernel_config` (ttnn.WormholeComputeKernelConfig / BlackholeComputeKernelConfig, optional)
    drives math_fidelity / math_approx_mode / dst_full_sync_en; the default reproduces the Phase-0
    configuration (HiFi4, fp32 DEST, exact SFPU). fp32_dest_acc_en must stay True (fp32 statistics).
    `activation="silu"` fuses SiLU into the apply pass (output = silu(groupnorm(x))).
    """
    validate(input_tensor, num_groups, gamma=gamma, beta=beta, in_place=in_place)
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    _validate_arguments(input_tensor, num_groups, gamma, beta, eps, compute_kernel_config)
    if activation not in (None, "silu"):
        raise ValueError(f"{_OP}: activation must be None or 'silu', got {activation!r}")

    device = input_tensor.device()
    if in_place:
        # The result lands in the input's own L1 shard (pass 3 reads each input tile for the last
        # time exactly once, then packs the output over it); the returned tensor IS the input.
        output_tensor = input_tensor
    else:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(input_tensor.shape)),
            input_tensor.dtype,
            input_tensor.layout,
            device,
            input_tensor.memory_config(),
        )

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)  # output MUST be last

    rounds = plan_channel_rounds(input_tensor, num_groups) if channel_rounds is None else int(channel_rounds)
    Ct = -(-int(input_tensor.shape[3]) // 32)
    if rounds <= 1:
        program_descriptor = create_program_descriptor(
            input_tensor,
            output_tensor,
            num_groups,
            gamma=gamma,
            beta=beta,
            eps=eps,
            compute_kernel_config=compute_kernel_config,
            activation=activation,
        )
        return ttnn.generic_op(io_tensors, program_descriptor)
    # "temporal" rounds: one L1-resident program per contiguous slice of whole groups (see config.TEMPORAL_ROUNDS)
    Kt = Ct // rounds
    out = None
    for r in range(rounds):
        program_descriptor = create_program_descriptor(
            input_tensor,
            output_tensor,
            num_groups,
            gamma=gamma,
            beta=beta,
            eps=eps,
            compute_kernel_config=compute_kernel_config,
            activation=activation,
            channel_tile_offset=r * Kt,
            channel_tiles=Kt,
        )
        out = ttnn.generic_op(io_tensors, program_descriptor)
    return out


def plan_channel_rounds(input_tensor, num_groups):
    """Number of channel rounds (1 = whole tensor in one program). Picks the fewest rounds whose per-core block
    (all cores splitting HW, the slice's Kt tiles wide) fits the L1 budget; slices must be whole tiles and
    whole groups. Interleaved inputs only; 1 when the whole tensor is resident or nothing fits."""
    if not config.TEMPORAL_ROUNDS or input_tensor.memory_config().is_sharded():
        return 1
    N, _, HW, C = [int(v) for v in input_tensor.shape]
    G = int(num_groups)
    Cg = C // G
    Ct = -(-C // 32)
    if C % 32 != 0:
        return 1  # ragged last channel tile: keep the single-program path
    HWt = -(-HW // 32)
    grid = input_tensor.device().compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    Hmax = -(-HWt // min(HWt, num_cores))
    tile_bytes = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}.get(input_tensor.dtype, 2048)
    budget = config.L1_CB_BUDGET_BYTES - config.TEMPORAL_FIXED_RESERVE_BYTES
    for rounds in range(1, Ct + 1):
        if Ct % rounds != 0:
            continue
        Kt = Ct // rounds
        if (Kt * 32) % Cg != 0:
            continue  # slice boundary would split a group
        if Kt > config.MAX_CORE_C_TILES and rounds > 1:
            continue  # per-core block wider than the kernel cap: a c_split would still be needed
        if Hmax * Kt * tile_bytes <= budget:
            return rounds
    return 1
