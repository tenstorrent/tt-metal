# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model cases of the sdpa fork (bring-up task O.1): each captured ttnn.bringup call on random inputs, against
reference.py, plus the A, B, A determinism check (FORK_TESTS.md).

    python -m bringup_fw.testing.fork_tests --fork sdpa

ring_joint cases: the metadata (slot_id / kv_actual_isl_tensor) chunked path of
ttnn.bringup.ring_joint_scaled_dot_product_attention over a served bfloat8_b KV cache (block-cyclic over SP, DRAM
ND-sharded one 32-row record per bank round-robin), as the model calls it. Every slot of the cache is random, so a read
of the wrong slot, or of keys past the causal / window limit, changes the output. The reference reads the cache's
bfloat8_b values (rounded on the host, then uploaded unchanged). Checked: every output element (the chunk is all real
rows) by PCC and rel L2; with a window, also the gather buffers' halo rows the model consumes (those before kv_actual,
from the cache), exactly. Not checked: the third output (log-sum-exp) and other buffer rows (the model ignores them).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import ttnn

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"sdpa_fork_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASES = _load("cases").CASES
ref = _load("reference")
RING_CASES = [c for c in CASES if c.get("op") == "ring_joint"]


def _device_params(case):
    # require_exact_physical_num_devices: run a case only on a box of its mesh's size (known issue "Other models'
    # 4-chip fork cases cannot run on an 8-chip box")
    return {
        "fabric_config": getattr(ttnn.FabricConfig, case["fabric_config"]),
        "require_exact_physical_num_devices": True,
    }


def _params(cases):
    return [pytest.param(tuple(c["mesh"]), _device_params(c), c, id=c["id"]) for c in cases]


def _bfp8_round(t: torch.Tensor) -> torch.Tensor:
    """The values a bfloat8_b TILE tensor holds (shared exponent per 16 values of a tile row), on the host."""
    return ttnn.to_torch(ttnn.from_torch(t.float(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


def _served_memory_config(mesh, width):
    banks = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(mesh.dram_grid_size().x)]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, ttnn.TILE_SIZE, width],
            grid=banks,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _replicated_topology(mesh, t):
    """The served cache's topology (the model allocates it on the device and marks it replicated)."""
    dist = ttnn.MeshShape(*tuple(mesh.shape))
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
    t.update_tensor_topology(ttnn.TensorTopology(dist, [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], coords))
    return t


def _scalar(mesh, value):
    import torch

    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _per_device(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]


class RingInputs:
    """Host and device inputs of one ring_joint case. The cache (both slots) is shared by A and B; q, slot and
    kv_actual are per run."""

    def __init__(self, case, mesh):
        import torch

        self.case, self.mesh = case, mesh
        sp, tp = case["mesh"]
        self.sp, self.tp = sp, tp
        g = torch.Generator().manual_seed(case["seed"])
        slots, kvh, L = case["slots"], case["kv_heads"], case["local_seq"]
        # cache[r, s, c, h, l, d]: SP row r, slot s, TP column c, local kv head h, local row l
        self.k = _bfp8_round(torch.randn(sp * slots, tp * kvh, L, case["k_dim"], generator=g)).view(
            sp, slots, tp, kvh, L, case["k_dim"]
        )
        self.v = _bfp8_round(torch.randn(sp * slots, tp * kvh, L, case["v_dim"], generator=g)).view(
            sp, slots, tp, kvh, L, case["v_dim"]
        )
        self.local_pos = ref.cache_local_index(sp, case["chunk"], L)  # [sp, L] global position of each local row
        self.tt_k, self.tt_v = (self._cache(t) for t in (self.k, self.v))
        self.sink = None
        self.tt_sink = None
        if case["attention_sink"]:
            # sink * scale is the sink logit; ~N(2.9, 1.4) sits inside the range of a 128-key row's log-sum-exp
            self.sink = (40 + 20 * torch.randn(tp * case["heads"], generator=g)).to(torch.bfloat16).float()
            self.tt_sink = ttnn.from_torch(
                self.sink.view(1, -1, 1, 1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(None, 1)),
            )

    def _cache(self, t):
        sp, slots, tp, kvh, L, d = t.shape
        tt = ttnn.from_torch(
            t.reshape(sp * slots, tp * kvh, L, d),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=_served_memory_config(self.mesh, d),
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(sp, tp), dims=(0, 1)),
        )
        return _replicated_topology(self.mesh, tt)

    def global_kv(self, t, slot, col):
        """[kv_heads, S, d]: column ``col``'s keys / values of ``slot`` in global position order."""
        import torch

        sp, _, _, kvh, L, d = t.shape
        out = torch.zeros(kvh, sp * L, d)
        for r in range(sp):
            out[:, self.local_pos[r]] = t[r, slot, col]
        return out

    def run_inputs(self, seed, slot, kv_actual):
        """q (host [sp, tp * heads, W, d] and device), the metadata scalars, and each SP row's q positions."""
        import torch

        c = self.case
        g = torch.Generator().manual_seed(seed)
        W = c["chunk"] // self.sp
        q = (torch.randn(self.sp, self.tp * c["heads"], W, c["k_dim"], generator=g) * c["q_scale"]).to(torch.bfloat16)
        tt_q = ttnn.from_torch(
            q,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(self.sp, self.tp), dims=(0, 1)),
        )
        pos = ref.chunk_positions(kv_actual, self.sp, c["chunk"])
        return dict(
            q=q.float(),
            tt_q=tt_q,
            slot=_scalar(self.mesh, slot),
            kv_actual=_scalar(self.mesh, kv_actual),
            pos=pos,
            slot_value=slot,
            kv_actual_value=kv_actual,
        )


def _ring_call(case, mesh, inp, run, bufs, sems):
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, case["math_fidelity"]),
        math_approx_mode=case["math_approx_mode"],
        fp32_dest_acc_en=case["fp32_dest_acc_en"],
        packer_l1_acc=case["packer_l1_acc"],
    )
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*case["sdpa_grid"]),
        q_chunk_size=case["q_chunk"],
        k_chunk_size=case["k_chunk"],
        exp_approx_mode=False,
        segmented_accumulation=case["segmented_accumulation"],
    )
    extra = {}
    if case["attention_sink"]:
        extra["attention_sink"] = inp.tt_sink
    if case["sliding_window_size"]:
        extra["sliding_window_size"] = case["sliding_window_size"]
    out, _, _ = ttnn.bringup.ring_joint_scaled_dot_product_attention(
        run["tt_q"],
        inp.tt_k,
        inp.tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=bufs[0],
        persistent_output_buffer_v=bufs[1],
        joint_strategy="rear",
        logical_n=case["logical_n"],
        program_config=prog,
        scale=case["scale"],
        compute_kernel_config=kcfg,
        dim=2,
        multi_device_global_semaphore=sems,
        num_links=case["num_links"],
        cluster_axis=case["cluster_axis"],
        mesh_device=mesh,
        topology=getattr(ttnn.Topology, case["topology"]),
        ccl_core_grid_offset=ttnn.CoreCoord(*case["ccl_core_grid_offset"]),
        use_column_major_ccl=case["use_column_major_ccl"],
        is_causal=True,
        is_balanced=False,
        slot_id=run["slot"],
        kv_actual_isl_tensor=run["kv_actual"],
        kv_cache_num_layers=case["kv_cache_num_layers"],
        kv_cache_layer_idx=case["kv_cache_layer_idx"],
        exact_exp=case["exact_exp"],
        **extra,
    )
    return out


def _halo_expect(case, inp, run):
    """Per device (row-major): (mask [2 * halo], k rows [kv, 2 * halo, d], v rows) of the halo rows the model takes
    from the gather buffers: positions in [0, kv_actual), from the selected slot's cache."""
    halo = case["gather_rows"] // 2
    out = []
    for r in range(inp.sp):
        hp = ref.halo_positions(run["pos"][r], halo, case["chunk"] // inp.sp)
        mask = (hp >= 0) & (hp < run["kv_actual_value"])
        for c in range(inp.tp):
            kg = inp.global_kv(inp.k, run["slot_value"], c)
            vg = inp.global_kv(inp.v, run["slot_value"], c)
            idx = hp.clamp(min=0)
            out.append((mask, kg[:, idx], vg[:, idx]))
    return out


def _where(rows, inp, slot, col, which):
    """Diagnostic: the global position of the selected slot whose kv-head-0 row equals each buffer row (-1: none)."""
    g = inp.global_kv(inp.k if which == "k" else inp.v, slot, col)[0]
    key = {tuple(r[:8].tolist()): p for p, r in enumerate(g)}
    return [key.get(tuple(r[:8].tolist()), -1) for r in rows[0]]


def _masked_halo(buf, expect):
    """The device's halo rows under each device's mask (others zeroed), for the determinism check."""
    return [b[0] * m.view(1, -1, 1) for b, (m, _, _) in zip(_per_device(buf), expect)]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case", _params(RING_CASES), indirect=["mesh_device", "device_params"]
)
def test_ring_joint(mesh_device, device_params, case):
    import torch
    from bringup_fw.testing import determinism

    mesh = mesh_device
    sp, tp = case["mesh"]
    assert tuple(mesh.shape) == (sp, tp)
    grid = mesh.compute_with_storage_grid_size()
    assert [grid.x - 1, grid.y] == case["sdpa_grid"], f"captured SDPA grid {case['sdpa_grid']}, box {grid}"

    cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
    sems = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
    ttnn.synchronize_device(mesh)
    bufs = [
        ttnn.zeros(
            [1, case["kv_heads"], case["gather_rows"], d],
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for d in (case["k_dim"], case["v_dim"])
    ]

    inp = RingInputs(case, mesh)
    run_a = inp.run_inputs(case["seed"], case["slot"], case["kv_actual"])
    run_b = inp.run_inputs(case["seed"] + 1, case["b_slot"], case["b_kv_actual"])

    out = _ring_call(case, mesh, inp, run_a, bufs, sems)
    dev = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(sp, tp), dims=(0, 1))).float()

    # reference, per chip (SP row r, TP column c): its q heads over its kv heads of the selected slot
    H = case["heads"]
    exp = torch.empty_like(dev)
    win = case["sliding_window_size"]
    for c in range(tp):
        kg = inp.global_kv(inp.k, case["slot"], c)
        vg = inp.global_kv(inp.v, case["slot"], c)
        sink = inp.sink[c * H : (c + 1) * H] if inp.sink is not None else None
        for r in range(sp):
            exp[r, c * H : (c + 1) * H] = ref.ring_sdpa(
                run_a["q"][r, c * H : (c + 1) * H], kg, vg, run_a["pos"][r], case["scale"], window=win, sink=sink
            )
    pcc = torch.corrcoef(torch.stack((dev.flatten(), exp.flatten())))[0, 1].item()
    rel = ((dev - exp).norm() / exp.norm()).item()
    ratio = (dev.norm() / exp.norm()).item()
    row_rel = ((dev - exp).norm(dim=-1) / exp.norm(dim=-1)).max().item()  # worst (head, q row)
    print(
        f"{case['id']}: pcc {pcc:.6f} rel {rel:.5f} norm ratio {ratio:.5f} worst row rel {row_rel:.4f} "
        f"max abs {(dev - exp).abs().max():.4f}"
    )
    assert (
        pcc >= case["pcc"] and rel <= case["rel"]
    ), f"{case['id']}: pcc {pcc} (>= {case['pcc']}), rel {rel} (<= {case['rel']})"
    lo, hi = case["norm_ratio"]
    assert lo <= ratio <= hi, f"{case['id']}: output norm ratio {ratio} outside [{lo}, {hi}]"
    assert row_rel <= case["row_rel"], f"{case['id']}: worst row rel {row_rel} > {case['row_rel']}"

    expect_a = None
    if case["check_halo"]:
        expect_a = _halo_expect(case, inp, run_a)
        checked = 0
        for which, buf, j in (("k", bufs[0], 1), ("v", bufs[1], 2)):
            for i, (got, e) in enumerate(zip(_per_device(buf), expect_a)):
                m = e[0]
                checked += int(m.sum())
                bad = (got[0][:, m] != e[j][:, m]).sum().item()
                assert bad == 0, (
                    f"{case['id']}: halo {which} device {i}: {bad} of {e[j][:, m].numel()} elements differ; "
                    f"buffer rows hold positions {_where(got[0], inp, case['slot'], i % tp, which)}, "
                    f"expected {ref.halo_positions(run_a['pos'][i // tp], case['gather_rows'] // 2, case['chunk'] // sp).tolist()}"
                )
        assert checked > 0, "no halo row before kv_actual: the case does not test the halo"
        print(f"{case['id']}: halo rows from the cache exact ({checked // 2} rows per buffer over all devices)")

    def a():
        o = _ring_call(case, mesh, inp, run_a, bufs, sems)
        if expect_a is None:
            return o
        return [o, _masked_halo(bufs[0], expect_a), _masked_halo(bufs[1], expect_a)]

    first = out if expect_a is None else [out, _masked_halo(bufs[0], expect_a), _masked_halo(bufs[1], expect_a)]
    determinism.assert_deterministic(
        a, lambda: _ring_call(case, mesh, inp, run_b, bufs, sems), first=first, label=case["id"]
    )
