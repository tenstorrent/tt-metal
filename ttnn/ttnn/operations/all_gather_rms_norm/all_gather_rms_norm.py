# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm — fused Fabric all-gather of the hidden-dim slices + RMSNorm / LayerNorm over the full
gathered row, as ONE ttnn.generic_op dispatch over a MeshProgramDescriptor (op_design.md, regime R1
mirror_resident_gather).

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below, then the public entry point.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from ttnn.operations.ccl import Topology
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .all_gather_rms_norm_fabric_plan import build_groups, resolve_num_links
from .all_gather_rms_norm_program_descriptor import OpConfig, create_mesh_program_descriptor


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Fresh Phase 0 config per call: HiFi4, fp32 DEST accumulation, exact SFPU math."""
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = True
    config.math_approx_mode = False
    return config


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (per-device INPUT shape inputs[0])
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    "norm_type": ["rms_norm", "layer_norm"],
    "beta_mode": ["beta", "no_beta"],
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
    "cluster_axis": [0, 1],
    "topology": [Topology.Linear, Topology.Ring],
    "num_links": [1, 2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Precision convention: float32 input requires fp32 DEST accumulation.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # bfloat8_b is block-quantized: the slice-tail zero fill is element-addressed (base op exclusion).
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _resolved_links(input_tensor, cluster_axis, topology, num_links):
    if num_links is not None:
        return num_links
    try:
        return resolve_num_links(input_tensor.device(), cluster_axis, topology == Topology.Ring)
    except Exception:  # noqa: BLE001
        return None


def validate(
    input_tensor,
    *,
    cluster_axis,
    gamma=None,
    beta=None,
    norm_type="rms_norm",
    epsilon=1e-6,
    topology=Topology.Linear,
    num_links=None,
    compute_kernel_config=None,
    memory_config=None,
    output_tensor=None,
):
    shape = list(input_tensor.shape)
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": gamma.dtype if gamma is not None else "none",
        "gamma_layout": gamma.layout if gamma is not None else "none",
        "norm_type": norm_type,
        "beta_mode": "beta" if beta is not None else "no_beta",
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": _resolved_links(input_tensor, cluster_axis, topology, num_links),
    }
    if len(shape) >= 2:
        for axis_name, tagger in INPUT_TAGGERS.items():
            axes[axis_name] = tagger((tuple(shape),), axes)
    else:
        axes["rank"] = len(shape)
    for axis, allowed in SUPPORTED.items():
        if axis not in axes:
            continue
        if axis == "num_links" and axes[axis] is not None and axes[axis] > 2:
            continue  # more discovered links than the gate lists: the caller-error check below handles it
        if axis == "rank" and len(shape) < 2:
            continue  # caller error below
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_gather_rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_gather_rms_norm: unsupported combination (refinement candidate): {exc}")
    return axes


def _check_caller(input_tensor, *, cluster_axis, gamma, beta, norm_type, epsilon, topology, num_links, output_tensor):
    shape = [int(d) for d in input_tensor.shape]
    if len(shape) < 2:
        raise ValueError(f"all_gather_rms_norm: input rank must be >= 2, got {len(shape)}")
    if norm_type not in ("rms_norm", "layer_norm"):
        raise ValueError(f"all_gather_rms_norm: norm_type must be 'rms_norm' or 'layer_norm', got {norm_type!r}")
    if beta is not None and gamma is None:
        raise ValueError("all_gather_rms_norm: beta requires gamma")
    if beta is not None and norm_type == "rms_norm":
        raise ValueError("all_gather_rms_norm: beta is only valid with norm_type='layer_norm'")
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    if cluster_axis not in (0, 1):
        raise ValueError(f"all_gather_rms_norm: cluster_axis must be 0 or 1, got {cluster_axis}")
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"all_gather_rms_norm: cluster_axis {cluster_axis} has size {G} (need >= 2)")
    W = G * shape[-1]

    def check_weight(t, name):
        ws = [int(d) for d in t.shape]
        if t.layout == ttnn.TILE_LAYOUT:
            want = [1, 1, 1, W]
        else:
            want = [1, 1, -(-W // 32), 32]
        if ws != want:
            raise ValueError(f"all_gather_rms_norm: {name} shape {ws} != {want} for layout {t.layout}")

    if gamma is not None:
        check_weight(gamma, "gamma")
    if beta is not None:
        check_weight(beta, "beta")
        if beta.dtype != gamma.dtype or beta.layout != gamma.layout:
            raise ValueError("all_gather_rms_norm: beta must match gamma's dtype and layout")
    if epsilon is None or not (float(epsilon) >= 0.0) or float(epsilon) == float("inf"):
        raise ValueError(f"all_gather_rms_norm: epsilon must be finite and >= 0, got {epsilon}")
    if output_tensor is not None:
        want = shape[:-1] + [W]
        if [int(d) for d in output_tensor.shape] != want:
            raise ValueError(f"all_gather_rms_norm: output_tensor shape {list(output_tensor.shape)} != {want}")
        if output_tensor.dtype != input_tensor.dtype or output_tensor.layout != input_tensor.layout:
            raise ValueError("all_gather_rms_norm: output_tensor dtype / layout must match the input's")
    # live cluster
    node = lambda coord: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    groups = build_groups(mesh_shape, cluster_axis)
    ring = topology == Topology.Ring and G > 2
    if ring:
        fc = ttnn.get_fabric_config()
        wraps = {
            ttnn.FabricConfig.FABRIC_2D_TORUS_X: {1},
            ttnn.FabricConfig.FABRIC_2D_TORUS_Y: {0},
            ttnn.FabricConfig.FABRIC_2D_TORUS_XY: {0, 1},
        }
        ring_1d = {getattr(ttnn.FabricConfig, n) for n in ("FABRIC_1D_RING",) if hasattr(ttnn.FabricConfig, n)}
        if fc not in ring_1d and cluster_axis not in wraps.get(fc, set()):
            raise ValueError(f"all_gather_rms_norm: fabric config {fc} does not wrap cluster_axis {cluster_axis}")
        for grp in groups:
            if not ttnn.get_forwarding_link_indices(node(grp[-1]), node(grp[0])):
                raise ValueError("all_gather_rms_norm: Ring requested but the axis has no wrap link")
    avail = resolve_num_links(mesh, cluster_axis, topology == Topology.Ring)
    if num_links is not None and (num_links <= 0 or num_links > avail):
        raise ValueError(f"all_gather_rms_norm: num_links={num_links} outside [1, {avail}] discovered")
    return G, (num_links if num_links is not None else avail)


# ---------------------------------------------------------------------------
# Persistent cross-chip state (global semaphores, cached per mesh + port-core set)
# ---------------------------------------------------------------------------

_SEM_CACHE = {}


def _global_sems(mesh, ports_union):
    key = (id(mesh), tuple(ports_union))
    if key not in _SEM_CACHE:
        cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(ports_union)]
        )
        sems = tuple(ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2))  # (arrival, ready)
        ttnn.synchronize_device(mesh)  # every chip's counters are 0 before any chip's first call
        _SEM_CACHE[key] = (mesh, sems)
    return _SEM_CACHE[key][1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def all_gather_rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    cluster_axis: int,
    gamma: Optional[ttnn.Tensor] = None,
    beta: Optional[ttnn.Tensor] = None,
    norm_type: str = "rms_norm",
    epsilon: float = 1e-6,
    topology=Topology.Linear,
    num_links: Optional[int] = None,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    output_tensor: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        beta=beta,
        norm_type=norm_type,
        epsilon=epsilon,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        output_tensor=output_tensor,
    )
    G, L = _check_caller(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        beta=beta,
        norm_type=norm_type,
        epsilon=epsilon,
        topology=topology,
        num_links=num_links,
        output_tensor=output_tensor,
    )
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    mesh = input_tensor.device()
    shape = [int(d) for d in input_tensor.shape]
    out_shape = shape[:-1] + [G * shape[-1]]
    if output_tensor is None:
        out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(out_shape), input_tensor.dtype, ttnn.TILE_LAYOUT, mesh, out_mem
        )
    cfg = OpConfig(
        cluster_axis=cluster_axis,
        ring=topology == Topology.Ring,
        num_links=L,
        is_ln=norm_type == "layer_norm",
        epsilon=float(epsilon),
        compute_kernel_config=compute_kernel_config,
    )
    from .all_gather_rms_norm_fabric_plan import plan_fabric

    fabric = plan_fabric(mesh, cluster_axis=cluster_axis, ring_requested=cfg.ring, num_links=L)
    sems = _global_sems(mesh, fabric[2])
    mesh_desc, _ = create_mesh_program_descriptor(input_tensor, gamma, beta, output_tensor, cfg, sems, fabric)
    io = [input_tensor] + ([gamma] if gamma is not None else []) + ([beta] if beta is not None else [])
    return ttnn.generic_op(io + [output_tensor], mesh_desc)
