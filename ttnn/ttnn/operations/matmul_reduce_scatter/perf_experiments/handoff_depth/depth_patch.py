# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf experiment (idea B, hand-off depth): a pytest plugin that re-plans matmul_reduce_scatter's hand-off depth
WITHOUT touching the real op files. Load it with

    PYTHONPATH=<this dir> MMRS_HANDOFF_DEPTH=<2|3|...|G|auto> scripts/run_safe_pytest.sh <test> -p depth_patch

MMRS_HANDOFF_DEPTH:
  * an integer d  -> every plan uses d slots per compute core (d = 2 is the op's shipped baseline);
  * "auto"        -> the candidate rule: depth = G (one slot per scatter block of the compute order, no slot reuse) when
                     the plan at depth G keeps the depth-2 plan's regime and K-block; else the largest such depth >= 2
                     (never trades residency / K-block for depth -- unmeasured, see the report).
  * unset         -> no patch (the op as shipped).

Mechanism: the kernels are already depth-generic (the compute packs into a CB ring of handoff_slots * block_tiles
pages; the transport reader gets each entry's slot index `cidx mod handoff_slots` from the host), so the only change is
the planner: `Blocking.handoff_slots` reads a per-plan `handoff_depth` and `_get_plan` picks it (a copy of the real
`_get_plan` with the depth rule inserted after the blocking convergence loop; everything else verbatim).
"""

import os
import sys

import ttnn

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as pd

import ttnn.operations.matmul_reduce_scatter  # noqa: F401

mrs = sys.modules["ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter"]

POLICY = os.environ.get("MMRS_HANDOFF_DEPTH")


def _plan_converged(plan_args, l1_free, depth):
    """The real _get_plan's convergence loop (hand-off shard taken out of the CB budget) at a given depth."""
    hb = lambda b: depth * b.waves * b.core_m_tiles * b.core_n_tiles * pd.BF16_TILE_BYTES
    blk = pd._plan_blocking(**plan_args, l1_cb_budget=l1_free)
    for _ in range(3):
        nxt = pd._plan_blocking(**plan_args, l1_cb_budget=l1_free - hb(blk))
        if hb(nxt) <= hb(blk):
            return nxt
        blk = nxt
    raise ValueError("matmul_reduce_scatter: blocking does not converge under the L1 budget")


def _choose_depth(plan_args, l1_free, G):
    """Candidate rule: G when it fits without changing the depth-2 plan, else the largest fitting depth >= 2."""
    ref = _plan_converged(plan_args, l1_free, pd.HANDOFF_DEPTH)  # shipped depth (2): the plan to preserve
    sig = lambda b: (b.regime, b.k_block_tiles, b.core_m_tiles, b.core_n_tiles, b.waves)
    for d in range(G, pd.HANDOFF_DEPTH, -1):
        try:
            b = _plan_converged(plan_args, l1_free, d)
        except ValueError:
            continue
        if sig(b) == sig(ref):
            return d, b
    return pd.HANDOFF_DEPTH, ref


def _get_plan(
    mesh_device, cluster_axis, num_links, M, K, N, scatter_dim, a_dtype, w_dtype, fp32_acc, fidelity, ring=False
):
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
        pd.WAVES_PIN,
        pd.WAVES_MAX,
        pd.K_BLOCKS_MIN,
        POLICY,
    )
    if key in mrs._PLAN_CACHE:
        return mrs._PLAN_CACHE[key][1]
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    ring = ring and G >= 3
    if ring:
        mrs._check_ring_fabric(cluster_axis)
    groups = mrs._groups(mesh_shape, cluster_axis, ring)
    links, L = mrs._links(mesh_device, groups, num_links)
    pl = pd._plan_placement(mesh_device, L, groups, links)
    num_banks = mesh_device.dram_grid_size().x * mesh_device.dram_grid_size().y
    if num_banks % (2 * L):
        raise ValueError(f"matmul_reduce_scatter: {num_banks} DRAM banks do not split into {2 * L} bank sets")
    comp_rows, comp_cols = pl.grid_y - pl.transport_rows, pl.grid_x
    l1_free = mrs._l1_cb_capacity(mesh_device) - pd.L1_RESERVE
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
    # ---- the depth rule (the only change vs the real _get_plan) ----
    if POLICY == "auto":
        depth, blk = _choose_depth(plan_args, l1_free, G)
    else:
        depth = int(POLICY)
        blk = _plan_converged(plan_args, l1_free, depth)
    blk.handoff_depth = depth
    print(
        f"[handoff_depth] policy={POLICY} G={G} depth={depth} regime={blk.regime} kbt={blk.k_block_tiles} "
        f"core={blk.core_m_tiles}x{blk.core_n_tiles} M={M} K={K} N={N} sd={scatter_dim} ring={ring}"
    )
    assert pd.BLOCKS_IN_FLIGHT == 1
    xp = pd._plan_transport(blk)

    relay_scratch = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, (G + 1) * xp.segs_per_block, xp.seg_tiles * 1024]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    compute_cores = [pd.compute_core(pl, blk, ml, nl) for ml in range(blk.m_lines) for nl in range(blk.n_lines)]
    compute_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute_cores])
    shard_tiles = blk.handoff_slots * blk.core_m_tiles * blk.core_n_tiles
    handoff_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_set, (shard_tiles * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    sems = mrs._get_sems(mesh_device, cluster_axis, L, ring)
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
    mrs._PLAN_CACHE[key] = (mesh_device, plan)
    return plan


if POLICY:
    pd.Blocking.handoff_slots = property(lambda self: getattr(self, "handoff_depth", pd.HANDOFF_DEPTH) * self.waves)
    mrs._get_plan = _get_plan
    print(f"[handoff_depth] patch active: MMRS_HANDOFF_DEPTH={POLICY}")


# extra perf-harness geometries (L1-tight cells where depth G would change the plan, so the auto rule falls back):
# appended to the op's perf harness CASES at collection time (the harness reads CASES[case] at run time)
EXTRA_CASES = {
    "k3ffn": ((1, 1, 640, 8448), (8448, 7168), 1, -1, ttnn.bfloat8_b, False),  # auto depth 3 (G=4 shrinks the K-block)
    "k3ffn_fp32": ((1, 1, 640, 8448), (8448, 7168), 1, -1, ttnn.bfloat8_b, True),  # auto depth 2
}


def pytest_collection_modifyitems(items):
    for it in items:
        cases = getattr(it.module, "CASES", None)
        if isinstance(cases, dict) and "focus" in cases:
            for k, v in EXTRA_CASES.items():
                cases.setdefault(k, v)
