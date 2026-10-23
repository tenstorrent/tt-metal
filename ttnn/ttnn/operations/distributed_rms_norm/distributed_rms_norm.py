# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm — RMSNorm over a hidden dim sharded across one mesh axis, as ONE fused device program.

Every device holds a (..., M, W_local) slice of each row; the G devices along `cluster_axis` together hold the full
row W = G * W_local. One ttnn.generic_op per call: local sum of squares on a row x W-chunk worker grid, a Fabric
exchange of the fp32 per-row partials between port cores, the cross-device combine on one leader per row-group,
the inv_rms fan-out and the normalize (see op_design.md).
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from .distributed_rms_norm_program_descriptor import (
    build_groups,
    discovered_links,
    run_distributed_rms_norm,
)


# ---------------------------------------------------------------------------
# Compute-config default (single source of truth; the golden axis tagger reads it)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """Phase 0 default: the maxed-out precision corner (HiFi4, fp32 DEST accumulation, no approximations)."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Per-device shape: H (-2) and W_local (-1) tile alignment."""
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    # bfloat8_b: x / out CB formats derive from the tensor dtype; intermediates follow fp32_dest_acc_en.
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # False: 16-bit DEST; the exchanged partials / inv vectors keep Float32 pages (wire format), cb_xg drops to bf16,
    # UnpackToDestFp32 tags are dropped (they need a 32-bit DEST). float32 x stays refused (EXCLUSIONS).
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    # h_non_aligned: the padded rows of the last tile row stay row-independent end to end (sum, exchange, combine,
    # normalize are per row), so they never touch a valid row; the output's pad rows are don't-care.
    # w_non_aligned: the reader zeroes the tile padding of the device's last tile column (NoC zeros + the partial
    # face rows) before compute sees it, so it never enters the sum of squares; 1/W_global uses the true width.
    "alignment": ["tile_aligned", "h_non_aligned", "w_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, "none"],
    # ROW_MAJOR gamma: (1, 1, ceil(W_local/32), 32) sticks; the reader drops stick c into cb_gamma page c as row 0 of
    # face 0 / face 1 (whole-stick DRAM read, second half relocated locally) — what the Row broadcast reads.
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    # HEIGHT / WIDTH_SHARDED (L1): consumed natively — the shard backs cb_x_block (zero-copy), the shard grid is the
    # worker set (HEIGHT: one row-slot per shard, num_w_chunks = 1; WIDTH: num_w_chunks = shards), the ports go on
    # free cores past the shard grid, and a sharded output is packed straight into its shard.
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
    "cluster_axis": [0, 1],
    "topology": [Topology.Linear, Topology.Ring],
    # None = "every usable link" (the signature default). validate() canonicalizes it to the discovered count before
    # this gate, so the op never sees None here; it is listed because the harness's observe-only classify_call records
    # the raw kwarg, and a default-num_links call (every translated test) must classify as the supported cell it is.
    "num_links": [1, 2, None],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # float32 input requires fp32 accumulation (precision convention).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_caller_errors(input_tensor, cluster_axis, gamma, topology, num_links):
    """Caller errors (ValueError), checked before the support gate: they are not support refusals."""
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"distributed_rms_norm: input rank must be >= 2, got {len(shape)}")
    if cluster_axis not in (0, 1):
        raise ValueError(f"distributed_rms_norm: cluster_axis must be 0 or 1, got {cluster_axis!r}")
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"distributed_rms_norm: cluster_axis={cluster_axis} has mesh size {G}; nothing to reduce")
    w_local = shape[-1]
    if gamma is not None:
        gshape = list(gamma.shape)
        if gamma.layout == ttnn.TILE_LAYOUT:
            want = [1, 1, 1, w_local]
        else:
            want = [1, 1, -(-w_local // 32), 32]
        if gshape != want:
            raise ValueError(f"distributed_rms_norm: gamma shape {gshape} does not match {want} for W_local={w_local}")
    ring = topology == Topology.Ring and G > 2
    if ring:
        cfg = ttnn.get_fabric_config()
        wraps = {
            ttnn.FabricConfig.FABRIC_2D_TORUS_XY: (0, 1),
            ttnn.FabricConfig.FABRIC_2D_TORUS_X: (1,),
            ttnn.FabricConfig.FABRIC_2D_TORUS_Y: (0,),
        }.get(cfg, ())
        if cluster_axis not in wraps:
            raise ValueError(
                f"distributed_rms_norm: Ring on cluster_axis={cluster_axis} needs a fabric config that wraps it"
            )
    links = discovered_links(mesh_device, cluster_axis, ring)  # raises ValueError if a ring cannot close
    if num_links is not None and (num_links <= 0 or num_links > links):
        raise ValueError(f"distributed_rms_norm: num_links={num_links} invalid; {links} usable link(s) per hop")
    return links


def validate(
    input_tensor,
    *,
    cluster_axis,
    gamma=None,
    topology=Topology.Linear,
    num_links=None,
    compute_kernel_config=None,
    memory_config=None,
):
    links = _check_caller_errors(input_tensor, cluster_axis, gamma, topology, num_links)
    cfg = compute_kernel_config or default_compute_kernel_config()
    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": gamma.dtype if has_gamma else "none",
        "gamma_layout": gamma.layout if has_gamma else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        # canonicalization: None resolves to the discovered link count
        "num_links": num_links if num_links is not None else links,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"distributed_rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"distributed_rms_norm: unsupported combination (refinement candidate): {exc}")
    in_mc = input_tensor.memory_config()
    if in_mc.is_sharded():
        if in_mc.buffer_type != ttnn.BufferType.L1:
            raise UnsupportedAxisValue("distributed_rms_norm: sharded input must live in L1")
        if in_mc.shard_spec is None:
            raise UnsupportedAxisValue("distributed_rms_norm: sharded input without a shard spec (ND sharding)")
        if in_mc.shard_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
            raise UnsupportedAxisValue("distributed_rms_norm: only ROW_MAJOR shard orientation is supported")
    # output placement: interleaved (DRAM or L1), or sharded exactly like the input (packed in place into its shard)
    out_mc = output_memory_config(input_tensor, memory_config)
    if out_mc.is_sharded() and out_mc != in_mc:
        raise UnsupportedAxisValue(
            "distributed_rms_norm: a sharded output must have the input's memory config (layout + shard spec)"
        )
    return axes


def output_memory_config(input_tensor, memory_config):
    """The output's placement: the caller's, else a sharded input's own (the output follows the shards), else DRAM."""
    if memory_config is not None:
        return memory_config
    in_mc = input_tensor.memory_config()
    return in_mc if in_mc.is_sharded() else ttnn.DRAM_MEMORY_CONFIG


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def distributed_rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    cluster_axis: int,
    gamma: ttnn.Tensor | None = None,
    epsilon: float = 1e-6,
    topology=Topology.Linear,
    num_links: int | None = None,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    axes = validate(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    cfg = compute_kernel_config or default_compute_kernel_config()
    return run_distributed_rms_norm(
        input_tensor,
        gamma=gamma,
        cluster_axis=cluster_axis,
        epsilon=float(epsilon),
        ring=(topology == Topology.Ring),
        num_links=axes["num_links"],
        compute_kernel_config=cfg,
        memory_config=output_memory_config(input_tensor, memory_config),
    )
