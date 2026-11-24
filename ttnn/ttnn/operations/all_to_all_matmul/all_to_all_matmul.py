# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_to_all_matmul — fused all-to-all (head-sharded -> sequence-sharded) + per-head matmul over a ring.

Within each group of G devices along `cluster_axis` (a mesh row), device q ends with rows [q*S/G, (q+1)*S/G) of
every source's chunk for all H = G*Hl heads, each head multiplied by its own (K, N) matrix of a replicated weight:

    chunks[p] = chunk(x[p], G, dim=2);  a2a[q] = cat_p(chunks[p][q], dim=1);  out[q] = a2a[q] @ w   (per head)

One ttnn.generic_op on a MeshProgramDescriptor per call (see all_to_all_matmul_program_descriptor.py and
op_design.md): port cores send every remote quarter over the fabric straight into the L1 of the compute core that
multiplies it; compute cores read their weights from DRAM while the rows are in flight.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology


# ---------------------------------------------------------------------------
# Compute configuration (the only place the default is spelled)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """HiFi4, fp32 DEST accumulation, no approximation (the golden default)."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """inputs = ((B, Hl, S, K), (1, H, K, N), G): rows per device must be whole tiles."""
    (_, _, S, _), _, G = inputs
    if S % 32 or (S // G) % 32:
        return "m_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "weight_dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True],
    "alignment": ["tile_aligned"],
    "output_buffer": [ttnn.BufferType.DRAM],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "cluster_axis": [1],
    "topology": [Topology.Ring],
    "num_links": [2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _node(mesh_device, coord):
    return mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def _check_caller_errors(input_tensor, weight, cluster_axis, topology, num_links, memory_config):
    """Caller errors (ValueError / RuntimeError), raised before the support check."""
    x_shape = tuple(input_tensor.shape)
    w_shape = tuple(weight.shape)
    if len(x_shape) != 4:
        raise ValueError(f"all_to_all_matmul: input must be rank 4 (B, Hl, S, K), got {x_shape}")
    if len(w_shape) != 4:
        raise ValueError(f"all_to_all_matmul: weight must be rank 4 (1, H, K, N), got {w_shape}")
    if cluster_axis not in (0, 1):
        raise ValueError(f"all_to_all_matmul: cluster_axis must be 0 or 1, got {cluster_axis}")
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"all_to_all_matmul: cluster_axis={cluster_axis} has extent {G}; nothing to exchange")
    B, Hl, S, K = x_shape
    _, H, Kw, N = w_shape
    if w_shape[0] != 1:
        raise ValueError(f"all_to_all_matmul: weight leading dim must be 1, got {w_shape}")
    if H != G * Hl:
        raise ValueError(f"all_to_all_matmul: weight has {H} heads, expected G*Hl = {G}*{Hl} = {G * Hl}")
    if Kw != K:
        raise ValueError(f"all_to_all_matmul: weight K {Kw} != input K {K}")
    if S % G:
        raise ValueError(f"all_to_all_matmul: S={S} not divisible by G={G}")
    if K % 32 or N % 32:
        raise ValueError(f"all_to_all_matmul: K ({K}) and N ({N}) must be multiples of 32")
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("all_to_all_matmul: memory_config must be interleaved")
    if num_links < 1:
        raise ValueError(f"all_to_all_matmul: num_links must be >= 1, got {num_links}")
    # Live-cluster checks: every hop of every group (both directions) has num_links links; a Ring needs the wrap.
    rows, cols = mesh_shape
    groups = (
        [[(r, c) for c in range(cols)] for r in range(rows)]
        if cluster_axis == 1
        else [[(r, c) for r in range(rows)] for c in range(cols)]
    )
    ring = topology == Topology.Ring
    for grp in groups:
        hops = [(grp[i], grp[i + 1]) for i in range(G - 1)]
        if ring and G > 2:
            hops.append((grp[-1], grp[0]))
        for a, b in hops:
            for s, d in ((a, b), (b, a)):
                links = ttnn.get_forwarding_link_indices(_node(mesh_device, s), _node(mesh_device, d))
                if len(links) == 0 and ring and (s, d) in ((grp[-1], grp[0]), (grp[0], grp[-1])):
                    raise ValueError(
                        f"all_to_all_matmul: Ring along cluster_axis={cluster_axis} has no wrap link "
                        f"{s} -> {d} under {ttnn.get_fabric_config()}"
                    )
                if len(links) < num_links:
                    raise ValueError(
                        f"all_to_all_matmul: hop {s} -> {d} has {len(links)} usable link(s), {num_links} requested"
                    )
    # The weight must be TILE / interleaved and the input in DRAM (contract; no TARGET axis enumerates either).
    wmc = weight.memory_config()
    if weight.layout != ttnn.TILE_LAYOUT or wmc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("all_to_all_matmul: weight must be TILE, interleaved")
    if input_tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("all_to_all_matmul: input must be DRAM interleaved")
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    tile_bytes = int(input_tensor.buffer_page_size()) if input_tensor.layout == ttnn.TILE_LAYOUT else 2048
    if payload // tile_bytes < 1:
        raise RuntimeError(f"all_to_all_matmul: fabric max payload {payload} B is smaller than one tile")


def _axes(input_tensor, weight, cluster_axis, topology, num_links, memory_config, compute_kernel_config):
    cfg = compute_kernel_config or default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "output_buffer": (memory_config or ttnn.DRAM_MEMORY_CONFIG).buffer_type,
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": num_links,
    }
    G = tuple(input_tensor.device().shape)[cluster_axis]
    inputs = (tuple(input_tensor.shape), tuple(weight.shape), G)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes


def validate(
    input_tensor,
    weight,
    *,
    cluster_axis,
    topology=Topology.Ring,
    num_links=2,
    memory_config=None,
    compute_kernel_config=None,
):
    _check_caller_errors(input_tensor, weight, cluster_axis, topology, num_links, memory_config)
    axes = _axes(input_tensor, weight, cluster_axis, topology, num_links, memory_config, compute_kernel_config)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_to_all_matmul: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_to_all_matmul: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def all_to_all_matmul(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    cluster_axis: int,
    topology=Topology.Ring,
    num_links: int = 2,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        weight,
        cluster_axis=cluster_axis,
        topology=topology,
        num_links=num_links,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    from .all_to_all_matmul_program_descriptor import run

    return run(
        input_tensor,
        weight,
        cluster_axis=cluster_axis,
        num_links=num_links,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config or default_compute_kernel_config(),
    )
