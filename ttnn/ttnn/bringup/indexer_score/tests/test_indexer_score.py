# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The indexer_score fork's DSA scorers against their torch semantics (reference.py), one random-input case per
captured call (cases.py).

ring_indexer_score_dsa: random natural-order keys are laid out as the model's block-cyclic, 4-chip-striped cache would
leave them after its TP-inner gather (k_local per SP row, reference.k_local_positions), the ring's full-T k buffer
starts as random garbage, and each chip's scores [Sq, T] are checked against a float32 reference on the same bf16
q / k / gates:
- every future column (t > the row's position) is -inf and every causal column finite (exact);
- causal columns: PCC and the relative L2 error, per chip.
Don't-care: nothing (kv_len = T, so every column of every row is written)."""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.testing import determinism

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_indexer_score_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    return p


def _kernel_config(k):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, k["math_fidelity"]),
        math_approx_mode=k["math_approx_mode"],
        fp32_dest_acc_en=k["fp32_dest_acc_en"],
        packer_l1_acc=k["packer_l1_acc"],
        dst_full_sync_en=k["dst_full_sync_en"],
    )


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES if c["op"] == "ring_indexer_score_dsa"],
    ids=[c["id"] for c in CASES if c["op"] == "ring_indexer_score_dsa"],
    indirect=["mesh_device", "device_params"],
)
def test_ring_indexer_score_dsa(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    sp_axis, tp_axis = c["block_cyclic_sp_axis"], c["seq_subshard_axis"]
    assert c["cluster_axis"] == sp_axis and (sp_axis, tp_axis) == (0, 1), "the test lays out SP on rows, TP on cols"
    assert c["block_cyclic_cache_tp_sharded"]
    sp, tp = rows, cols
    H, D, Sq, T, cl = c["heads"], c["head_dim"], c["q_rows"], c["t"], c["block_cyclic_chunk_local"]
    assert cl == tp * Sq and c["k_local_rows"] == T // sp and c["kv_len"] == T
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(c["num_semaphores"])]
    host, dev = _inputs(mesh_device, c, c["seed"])
    q, w, k_nat = host
    out, tt_k = _call(mesh_device, c, dev, sems)
    outs = [ttnn.to_torch(t).float().reshape(Sq, T) for t in ttnn.get_device_tensors(out)]
    assert len(outs) == n_dev

    for d in range(n_dev):
        r, cc = divmod(d, cols)
        q0 = ref.query_start(c["chunk_start_idx"], r, cc, Sq, cl)
        blk = slice(d * Sq, (d + 1) * Sq)
        want = ref.dsa_score(q[0, :, blk], k_nat, w[0, 0, blk], q0)
        got = outs[d]
        future = torch.isneginf(want)
        n_mask = int((~torch.isneginf(got[future])).sum())
        n_nonfinite = int((~torch.isfinite(got[~future])).sum())
        assert n_mask == 0, f"dev {d}: {n_mask} future columns not -inf"
        assert n_nonfinite == 0, f"dev {d}: {n_nonfinite} causal columns not finite"
        a, b = got[~future], want[~future]
        pcc = _pcc(a, b)
        rel = float((a - b).norm() / b.norm())
        logger.info(f"dev {d}: rows from {q0}, {int((~future).sum())} causal scores, pcc {pcc:.7f}, rel {rel:.5f}")
        assert (
            pcc >= c["pcc"] and rel <= c["rel"]
        ), f"dev {d}: pcc {pcc:.7f} (min {c['pcc']}), rel {rel:.5f} (max {c['rel']})"

    # The op fills the ring's k buffer (an input): each call gets a fresh copy of its starting garbage (_call), and
    # the filled buffer is compared too.
    dev_b = _inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(
        lambda: _call(mesh_device, c, dev, sems),
        lambda: _call(mesh_device, c, dev_b, sems),
        first=(out, tt_k),
        label=c["id"],
    )
    ttnn.deallocate(out)


def _put(mesh_device, c, t, mapper):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=getattr(ttnn.DataType, c["dtype"]),
        layout=getattr(ttnn.Layout, c["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _inputs(mesh_device, c, seed):
    """Host (q, w, k_nat) for `seed` and the device inputs (q, w, k_local) plus the host garbage of the k buffer."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    sp, tp = rows, cols
    H, D, Sq, T, cl = c["heads"], c["head_dim"], c["q_rows"], c["t"], c["block_cyclic_chunk_local"]
    g = torch.Generator().manual_seed(seed)

    # Host inputs, bf16. Device d = r * cols + c (row-major) gets query block d; its rows start at
    # chunk_start + r * chunk_local + c * Sq = chunk_start + d * Sq.
    q = torch.randn(1, H, n_dev * Sq, D, generator=g).to(torch.bfloat16)
    w = (torch.randn(1, 1, n_dev * Sq, H, generator=g) * c["gate_scale"]).to(torch.bfloat16)
    k_nat = torch.randn(T, D, generator=g).to(torch.bfloat16)
    garbage = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    pos = ref.k_local_positions(T, sp, tp, cl)  # [sp, T / sp]
    k_local_host = k_nat[pos.reshape(-1)].reshape(1, 1, T, D)  # SP row r = rows [r * T / sp, (r + 1) * T / sp)

    per_dev = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # dim 2 split over all devices, row-major
    tt_q, tt_w = _put(mesh_device, c, q, per_dev), _put(mesh_device, c, w, per_dev)
    k_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None))
    tt_k_local = _put(mesh_device, c, k_local_host, k_map)
    assert list(tt_q.shape) == [1, H, Sq, D] and list(tt_w.shape) == [1, 1, Sq, H]
    assert list(tt_k_local.shape) == [1, 1, T // sp, D]
    return (q, w, k_nat), (tt_q, tt_w, tt_k_local, garbage)


def _call(mesh_device, c, dev, sems):
    """One call on a fresh copy of the k buffer's garbage; returns (scores, the filled k buffer)."""
    tt_q, tt_w, tt_k_local, garbage = dev
    tt_k = _put(mesh_device, c, garbage, ttnn.ReplicateTensorToMesh(mesh_device))
    assert list(tt_k.shape) == [1, 1, c["t"], c["head_dim"]]
    pc = c["program_config"]
    out = ttnn.bringup.ring_indexer_score_dsa(
        tt_q,
        tt_k,
        tt_w,
        tt_k_local,
        sems,
        cluster_axis=c["cluster_axis"],
        topology=getattr(ttnn.Topology, c["topology"]),
        num_links=c["num_links"],
        chunk_start_idx=c["chunk_start_idx"],
        program_config=ttnn.bringup.IndexerScoreProgramConfig(
            q_chunk_size=pc["q_chunk_size"], k_chunk_size=pc["k_chunk_size"], head_group_size=pc["head_group_size"]
        ),
        compute_kernel_config=_kernel_config(c["compute_kernel_config"]),
        kv_len=c["kv_len"],
        seq_subshard_axis=c["seq_subshard_axis"],
        block_cyclic_sp_axis=c["block_cyclic_sp_axis"],
        block_cyclic_chunk_local=c["block_cyclic_chunk_local"],
        block_cyclic_cache_tp_sharded=c["block_cyclic_cache_tp_sharded"],
    )
    return out, tt_k
