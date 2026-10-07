# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_reduce_scatter — fused per-device matmul + fabric reduce-scatter (SUM), one generic_op dispatch.

Every device holds A (..., M, K) and W (K, N); the group of G devices along `cluster_axis` sums its partials
A[g] @ W[g] and the device at group position p keeps block p of the sum along `scatter_dim` (-2: rows, -1: columns).
The transport of a finished scatter block starts while later blocks are still being computed.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from .matmul_reduce_scatter_program_descriptor import (
    BF16_TILE_BYTES,
    BLOCKS_IN_FLIGHT,
    HANDOFF_DEPTH,
    L1_RESERVE,
    _plan_blocking,
    _plan_placement,
    _plan_transport,
    compute_core,
    create_mesh_program_descriptor,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    a_shape, w_shape = inputs[0], inputs[1]
    M, K, N = a_shape[-2], a_shape[-1], w_shape[-1]
    return "tile_aligned" if (M % 32 == 0 and K % 32 == 0 and N % 32 == 0) else "non_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "weight_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "cluster_axis": [0, 1],
    "scatter_dim": [-1, -2],
    "topology": [Topology.Linear, Topology.Ring],
    "num_links": [1, 2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _canonical_scatter_dim(scatter_dim, rank):
    return scatter_dim - rank if scatter_dim >= 0 else scatter_dim


def validate(input_tensor, weight, *, cluster_axis, scatter_dim=-2, topology=Topology.Linear, num_links=None):
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight.dtype,
        "layout": input_tensor.layout,
        "cluster_axis": cluster_axis,
        "scatter_dim": _canonical_scatter_dim(scatter_dim, len(input_tensor.shape)),
        "topology": topology,
        "num_links": num_links,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape), tuple(weight.shape)), axes)
    for axis, allowed in SUPPORTED.items():
        if axis == "num_links" and axes[axis] is None:
            continue  # None = every usable link (checked against the fabric below)
        if axis == "layout" and weight.layout not in allowed:
            raise UnsupportedAxisValue(f"matmul_reduce_scatter: weight layout {weight.layout} not in {allowed}")
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"matmul_reduce_scatter: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"matmul_reduce_scatter: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """The op's default matmul compute config: HiFi2 with fp32 DEST accumulation."""
    return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)


def _check_dram_interleaved(t, what):
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"matmul_reduce_scatter: {what} must be DRAM interleaved")


def _groups(mesh_shape, cluster_axis, ring=False):
    """{coord: (position p, previous coord or None, next coord or None)} for every device.

    ring: the line closes (position 0's previous is position G-1 and vice versa) -- the wrap hop."""
    rows, cols = mesh_shape
    out = {}
    if cluster_axis == 0:
        lines = [[(r, c) for r in range(rows)] for c in range(cols)]
    else:
        lines = [[(r, c) for c in range(cols)] for r in range(rows)]
    for line in lines:
        for p, coord in enumerate(line):
            n = len(line)
            if ring:
                out[coord] = (p, line[(p - 1) % n], line[(p + 1) % n])
            else:
                out[coord] = (p, line[p - 1] if p > 0 else None, line[p + 1] if p + 1 < n else None)
    return out


def _links(mesh_device, groups, num_links):
    """{(coord, peer): link ids} for every hop; resolves num_links=None to the min usable over hops."""
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    links = {}
    for coord, (_, prev, nxt) in groups.items():
        for peer in (prev, nxt):
            if peer is not None:
                links[(coord, peer)] = list(ttnn.get_forwarding_link_indices(node(coord), node(peer)))
    usable = min(len(v) for v in links.values()) if links else 0
    if num_links is None:
        num_links = min(usable, max(SUPPORTED["num_links"]))
    if num_links < 1 or usable < num_links:
        raise ValueError(
            f"matmul_reduce_scatter: {num_links} link(s) requested, but some hop has only {usable} usable link(s) "
            f"(fabric={ttnn.get_fabric_config()})"
        )
    return {k: v[:num_links] for k, v in links.items()}, num_links


def _check_ring_fabric(cluster_axis):
    """Ring needs a fabric config that wraps cluster_axis (TORUS_Y: axis 0, TORUS_X: axis 1, TORUS_XY / 1D_RING: both);
    the wrap hop's links are checked by _links like every other hop."""
    fc = ttnn.get_fabric_config()
    F = ttnn.FabricConfig
    wraps = {
        F.FABRIC_2D_TORUS_XY: (0, 1),
        F.FABRIC_1D_RING: (0, 1),
        F.FABRIC_2D_TORUS_X: (1,),
        F.FABRIC_2D_TORUS_Y: (0,),
    }.get(fc, ())
    if cluster_axis not in wraps:
        raise ValueError(
            f"matmul_reduce_scatter: topology=Ring needs a fabric config that wraps cluster_axis={cluster_axis} "
            f"(got {fc})"
        )


_PLAN_CACHE = {}
_SEM_CACHE = {}
NUM_SEMS = 7  # arrival fwd / bwd, ready fence, block ready, block ack x 3 (fwd ports, bwd ports, finals)


def _l1_cb_capacity(mesh_device):
    """Bytes of worker L1 the allocator manages per core (allocator base .. end of L1): the region shared by the static
    CBs (growing up from the base) and L1 buffers (growing down from the end). NOT get_max_worker_l1_unreserved_size(),
    which also counts the kernel-config ring buffer below the allocator base and so over-budgets the CBs."""
    return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_per_bank)


def _get_sems(mesh_device, cluster_axis, num_links):
    """Global semaphores, shared by every plan with the same (mesh, cluster_axis, num_links).

    One set per transport identity, over the whole worker grid: the port / final cores (placement depends on
    num_links only) then always receive ready-fence and arrival increments from the same neighbours (cluster_axis), so
    sharing across plans keeps the fence semantics of reusing one plan; every kernel re-arms what it consumed, so the
    counters are zero between calls. Per-plan semaphores would accumulate L1 allocations across the distinct shapes a
    model runs (bounded here at 4 sets per mesh)."""
    key = (id(mesh_device), cluster_axis, num_links)
    if key not in _SEM_CACHE:
        grid = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        sems = tuple(ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(NUM_SEMS))
        ttnn.synchronize_device(mesh_device)  # every counter zero before any chip's first call increments a neighbour's
        _SEM_CACHE[key] = (mesh_device, sems)
    return _SEM_CACHE[key][1]


def _get_plan(
    mesh_device, cluster_axis, num_links, M, K, N, scatter_dim, a_dtype, w_dtype, fp32_acc, fidelity, ring=False
):
    """Host plan + the persistent DRAM relay scratch, cached per plan key (global semaphores: `_get_sems`).

    The L1 hand-off buffer is NOT cached here: a persistent L1 allocation per plan key would accumulate across the
    distinct shapes a model (or a test session) runs on one mesh and eventually clash with a later plan's CB region
    (and permanently steal L1 from every other op). It is allocated per call (`_allocate_handoff`); the generic_op
    program cache patches its CB address on a hit (UpdateDynamicCircularBufferAddress)."""
    key = (
        id(mesh_device),
        cluster_axis,
        ring,
        num_links,
        M,
        K,
        N,
        scatter_dim,
        a_dtype,
        w_dtype,
        fp32_acc,
        str(fidelity),
    )
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key][1]
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    ring = ring and G >= 3  # a 2-device ring is the line (one link pair; both directions would share a channel)
    if ring:
        _check_ring_fabric(cluster_axis)
    groups = _groups(mesh_shape, cluster_axis, ring)
    links, L = _links(mesh_device, groups, num_links)
    pl = _plan_placement(mesh_device, L)
    num_banks = mesh_device.dram_grid_size().x * mesh_device.dram_grid_size().y
    if num_banks % (2 * L):
        raise ValueError(f"matmul_reduce_scatter: {num_banks} DRAM banks do not split into {2 * L} bank sets")
    comp_rows, comp_cols = pl.grid_y - pl.transport_rows, pl.grid_x
    # L1 a compute core can give to its CBs: the allocator's L1 region minus the hand-off shard and a fixed reserve.
    l1_free = _l1_cb_capacity(mesh_device) - L1_RESERVE
    plan_args = dict(
        comp_rows=comp_rows,
        comp_cols=comp_cols,
        Mt=M // 32,
        Kt=K // 32,
        Nt=N // 32,
        G=G,
        scatter_dim=scatter_dim,
        a_dtype=a_dtype,
        w_dtype=w_dtype,
        fp32_acc=fp32_acc,
    )
    # the hand-off shard depends on the per-core block; the factorization does not depend on L1, so solve once with
    # the full budget to get the block, then re-solve the K-block / regime with the shard taken out
    probe = _plan_blocking(**plan_args, l1_cb_budget=l1_free)
    handoff_bytes = HANDOFF_DEPTH * probe.core_m_tiles * probe.core_n_tiles * BF16_TILE_BYTES
    blk = _plan_blocking(**plan_args, l1_cb_budget=l1_free - handoff_bytes)
    assert BLOCKS_IN_FLIGHT == 1, "blocks_in_flight > 1 is not built"
    xp = _plan_transport(blk)

    # ---- persistent buffers (per plan, never per call) ----
    relay_scratch = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, (G + 1) * xp.segs_per_block, xp.seg_tiles * 1024]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    compute_cores = [compute_core(pl, blk, ml, nl) for ml in range(blk.m_lines) for nl in range(blk.n_lines)]
    compute_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute_cores])
    shard_tiles = HANDOFF_DEPTH * blk.core_m_tiles * blk.core_n_tiles
    handoff_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_set, (shard_tiles * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    sems = _get_sems(mesh_device, cluster_axis, L)
    plan = dict(
        G=G,
        groups=groups,
        links=links,
        num_links=L,
        pl=pl,
        blk=blk,
        xp=xp,
        relay_scratch=relay_scratch,
        handoff_shape=ttnn.Shape([len(compute_cores) * shard_tiles * 32, 32]),
        handoff_mem=handoff_mem,
        sems=sems,
        ring=ring,
    )
    _PLAN_CACHE[key] = (mesh_device, plan)
    return plan


def _allocate_handoff(mesh_device, plan):
    """Per-call L1 backing of cb_partial_handoff (one HEIGHT shard of handoff_depth slots per compute core)."""
    return ttnn.allocate_tensor_on_device(
        plan["handoff_shape"], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, plan["handoff_mem"]
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def matmul_reduce_scatter(
    input_tensor,
    weight,
    *,
    cluster_axis,
    scatter_dim=-2,
    topology=Topology.Linear,
    num_links=None,
    compute_kernel_config=None,
    memory_config=None,
):
    validate(
        input_tensor, weight, cluster_axis=cluster_axis, scatter_dim=scatter_dim, topology=topology, num_links=num_links
    )
    a_shape, w_shape = list(input_tensor.shape), list(weight.shape)
    rank = len(a_shape)
    scatter_dim = _canonical_scatter_dim(scatter_dim, rank)
    if not 2 <= rank <= 4 or any(d != 1 for d in a_shape[:-2]):
        raise ValueError(f"matmul_reduce_scatter: A must be rank 2-4 with leading dims 1, got {a_shape}")
    if len(w_shape) != 2:
        raise ValueError(f"matmul_reduce_scatter: W must be rank 2 (K, N), got {w_shape}")
    if a_shape[-1] != w_shape[0]:
        raise ValueError(f"matmul_reduce_scatter: A's K={a_shape[-1]} != W's K={w_shape[0]}")
    if input_tensor.layout != ttnn.TILE_LAYOUT or weight.layout != ttnn.TILE_LAYOUT:
        raise ValueError("matmul_reduce_scatter: A and W must be TILE layout")
    _check_dram_interleaved(input_tensor, "A")
    _check_dram_interleaved(weight, "W")
    if memory_config is not None and (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError("matmul_reduce_scatter: memory_config must be DRAM interleaved")
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"matmul_reduce_scatter: cluster_axis={cluster_axis} has {G} device(s); need >= 2")
    M, K, N = a_shape[-2], a_shape[-1], w_shape[-1]
    extent = M if scatter_dim == -2 else N
    if extent % (32 * G):
        raise ValueError(f"matmul_reduce_scatter: scattered extent {extent} does not split into {G} tile blocks")
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    plan = _get_plan(
        mesh_device,
        cluster_axis,
        num_links,
        M,
        K,
        N,
        scatter_dim,
        input_tensor.dtype,
        weight.dtype,
        bool(cfg.fp32_dest_acc_en),
        cfg.math_fidelity,
        ring=topology == Topology.Ring,
    )
    blk = plan["blk"]
    out_shape = a_shape[:-2] + [blk.blk_m_tiles * 32, blk.blk_n_tiles * 32]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    handoff_l1 = _allocate_handoff(mesh_device, plan)
    sems = tuple(int(ttnn.get_global_semaphore_address(s)) for s in plan["sems"])
    desc = create_mesh_program_descriptor(
        mesh_device,
        a=input_tensor,
        w=weight,
        scratch=plan["relay_scratch"],
        handoff=handoff_l1,
        output=output,
        sems=sems,
        blk=blk,
        xp=plan["xp"],
        pl=plan["pl"],
        groups=plan["groups"],
        cluster_axis=cluster_axis,
        links=plan["links"],
        num_links=plan["num_links"],
        compute_config=cfg,
        ring=plan["ring"],
    )
    return ttnn.generic_op([input_tensor, weight, plan["relay_scratch"], handoff_l1, output], desc)
