# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ring_indexer_score_dsa(natural_query_blocks=True, natural_query_rows=Wq) (fork option, DeepSeek-V4.1
P.31): queries of a SMALL chunk scoring index keys a LARGER chunk laid out (the decoder tail of DECODER_SKIP=replay128:
1024-token queries, keys written by 5120-token chunks), on the box's mesh (BRINGUP_TEST_MESH=RxC, default 2x2; SP = axis
0 ring, TP = axis 1 query sub-shard).

Keys (DeepSeek-V4.1's served index-key cache, SP-sharded, TP-replicated, ratio 1): Sk-token chunks, Wk = Sk / SP tokens
per SP row; key t on SP rank (t / Wk) % SP, block-cyclic (reference.k_local_positions with tp = 1); the call passes
block_cyclic_chunk_local = Wk. Queries: an Sq-token chunk at any 32-aligned start, Wq = Sq / SP tokens per SP row,
device (d, c) holds tokens start + d Wq + c (Wq / TP) + [0, Wq / TP) (natural_query_rows = Wq). The ring's full-T key
buffer starts as random garbage. Per device, against a float32 reference on the same bf16 inputs (reference.dsa_score)
over columns [0, kv_len), kv_len = ceil32(start + Sq): every future column -inf, every visible column finite, PCC and
relative L2 on the visible ones. Starts: 0, mid key chunk, across a key-chunk boundary, the cache's last query chunk.
Refusal: natural_query_rows without natural_query_blocks.

    BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/bringup/indexer_score/tests/unit/test_natural_query_rows.py
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_TESTS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_indexer_score_tests_nqr_{name}", _TESTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
_MESH = tuple(int(v) for v in os.environ.get("BRINGUP_TEST_MESH", "2x2").split("x"))
CASE = {
    "heads": 32,
    "head_dim": 128,
    "q_rows_chip": 32,  # Sq / (SP TP): the decoder tail's one tile row a chip
    "k_rows_chip": 160,  # Sk / (SP TP): DeepSeek-V4.1's 5120-token chunk on 8x4
    "chunks": 2,  # cache capacity in key chunks
    "gate_scale": 1.0 / 64,
    "seed": 11,
    "pcc": 0.9999,
    "rel": 0.004,
}

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")


def _geometry():
    sp, tp = _MESH
    wq, wk = CASE["q_rows_chip"] * tp, CASE["k_rows_chip"] * tp
    sq, sk = wq * sp, wk * sp
    return sp, tp, sq, wq, wq // tp, sk, wk, CASE["chunks"] * sk


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _put(mesh_device, t, mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t, device=mesh_device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, start, **extra):
    sp, tp, sq, wq, rq, sk, wk, t = _geometry()
    tt_k = _put(mesh_device, garbage, ttnn.ReplicateTensorToMesh(mesh_device))
    kv_len = min(t, -(-(start + sq) // 32) * 32)
    kw = dict(
        cluster_axis=0,
        topology=ttnn.Topology.Linear,
        num_links=1,
        chunk_start_idx=start,
        key_compression_ratio=1,
        program_config=ttnn.bringup.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=False,
        ),
        kv_len=kv_len,
        seq_subshard_axis=1 if tp > 1 else None,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=wk,
        natural_query_blocks=True,
        natural_query_rows=wq,
    )
    kw.update(extra)
    out = ttnn.bringup.ring_indexer_score_dsa(tt_q, tt_k, tt_w, tt_k_local, sems, **kw)
    ttnn.deallocate(tt_k)
    return out, kv_len


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [(_MESH, {"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    ids=[f"{_MESH[0]}x{_MESH[1]}-natural-query-rows"],
    indirect=True,
)
def test_natural_query_rows(mesh_device, device_params, expect_error):
    c = CASE
    sp, tp, sq, wq, rq, sk, wk, t = _geometry()
    H, D = c["heads"], c["head_dim"]
    g = torch.Generator().manual_seed(c["seed"])
    k_nat = torch.randn(t, D, generator=g).to(torch.bfloat16)
    garbage = torch.randn(1, 1, t, D, generator=g).to(torch.bfloat16)
    pos = ref.k_local_positions(t, sp, 1, wk)  # [sp, t / sp]
    k_local = k_nat[pos.reshape(-1)].reshape(sp, 1, t // sp, D)
    tt_k_local = _put(mesh_device, k_local, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, None)))
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, 1))
    starts = [0, wk + 96, sk - sq // 2 // 32 * 32, t - sq]
    for start in starts:
        q = torch.randn(1, H, sq, D, generator=g).to(torch.bfloat16)
        wt = (torch.randn(1, 1, sq, H, generator=g) * c["gate_scale"]).to(torch.bfloat16)
        qd = q.reshape(1, H, sp, tp, rq, D).permute(2, 3, 1, 4, 5, 0).reshape(sp, tp * H, rq, D)
        wd = wt.reshape(1, 1, sp, tp, rq, H).permute(2, 3, 0, 1, 4, 5).reshape(sp, tp, rq, H)
        tt_q, tt_w = _put(mesh_device, qd, mapper), _put(mesh_device, wd, mapper)
        out, kv_len = _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, start)
        outs = [ttnn.to_torch(x).float().reshape(rq, t) for x in ttnn.get_device_tensors(out)]
        for d in range(sp):
            for cc in range(tp):
                blk = slice(d * wq + cc * rq, d * wq + (cc + 1) * rq)
                q0 = start + d * wq + cc * rq
                want = ref.dsa_score(q[0, :, blk], k_nat, wt[0, 0, blk], q0)[:, :kv_len]
                got = outs[d * tp + cc][:, :kv_len]
                future = torch.isneginf(want)
                n_mask = int((~torch.isneginf(got[future])).sum())
                n_bad = int((~torch.isfinite(got[~future])).sum())
                assert n_mask == 0, f"start {start} dev ({d}, {cc}): {n_mask} future columns not -inf"
                assert n_bad == 0, f"start {start} dev ({d}, {cc}): {n_bad} visible columns not finite"
                a, b = got[~future], want[~future]
                pcc, rel = _pcc(a, b), float((a - b).norm() / b.norm())
                logger.info(
                    f"start {start} dev ({d}, {cc}): tokens from {q0}, kv_len {kv_len}, pcc {pcc:.7f} rel {rel:.5f}"
                )
                assert pcc >= c["pcc"] and rel <= c["rel"], f"start {start} dev ({d}, {cc}): {pcc}, {rel}"
        for x in (out, tt_q, tt_w):
            ttnn.deallocate(x)
    q = torch.zeros(1, H, sq, D).to(torch.bfloat16)
    qd = q.reshape(1, H, sp, tp, rq, D).permute(2, 3, 1, 4, 5, 0).reshape(sp, tp * H, rq, D)
    wd = torch.zeros(sp, tp, rq, H).to(torch.bfloat16)
    tt_q, tt_w = _put(mesh_device, qd, mapper), _put(mesh_device, wd, mapper)
    with expect_error(RuntimeError, "needs natural_query_blocks and a tile-aligned count"):
        _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, 0, natural_query_blocks=False)
