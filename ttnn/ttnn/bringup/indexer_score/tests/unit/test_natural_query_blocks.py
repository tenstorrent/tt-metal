# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ring_indexer_score_dsa with key_compression_ratio 2 and natural_query_blocks (fork options,
deepseek_v41_flash_d_p C.enc_source.indexer) on a 2x2 SP x TP mesh (SP = axis 0 ring, TP = axis 1 query sub-shard).

Layout (DeepSeek-V4.1's served index-key cache, 2_index_k_r2: SP-sharded, TP-replicated): S-token chunks, W = S / SP
tokens per SP row; key t (it pools tokens 2t, 2t + 1) lives on SP rank (2t / W) % SP at k_local row
(2t / S) (W / 2) + t mod (W / 2) -- block-cyclic with W / 2 keys per rank per chunk (reference.k_local_positions with
tp = 1). Queries are NATURAL blocks at any 32-aligned start: device (d, c) holds tokens start + d W + c Sq + [0, Sq),
Sq = W / TP. Key t is visible to token p iff t < (p + 1) // 2. The ring's full-T k buffer starts as random garbage.
Per device, against a float32 reference on the same bf16 inputs (reference.dsa_score, key_stride 2), over columns
[0, kv_len), kv_len = ceil32((start + S) / 2): every future column -inf, every visible column finite (exact), PCC and
relative L2 on the visible columns. Starts cover the slab-aligned case (where natural == the source's rotation-exact
geometry, also checked with the option off), a mid-W start, a mid-slab start and the last window of the cache.
Refusals: natural_query_blocks without a block-cyclic layout; with the metadata (chunk_start_idx_tensor) path.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_TESTS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_indexer_score_tests_nqb_{name}", _TESTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASE = {
    "mesh": [2, 2],
    "heads": 32,
    "head_dim": 128,
    "chunk": 2048,  # S (tokens); W = 1024, Sq = 512
    "chunks": 3,  # cache capacity in chunks: T = 3 * 2048 / 2 = 3072 keys
    "ratio": 2,
    "starts": [0, 32, 1088, 2048, 2592, 4064],
    "gate_scale": 1.0 / 64,
    "seed": 7,
    "pcc": 0.9999,
    "rel": 0.004,
}

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _put(mesh_device, t, mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t, device=mesh_device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def _geometry(mesh_device):
    c = CASE
    sp, tp = c["mesh"]
    s, r = c["chunk"], c["ratio"]
    w = s // sp
    return sp, tp, s, w, w // tp, c["chunks"] * s // r, r


def _setup(mesh_device):
    c = CASE
    sp, tp, s, w, sq, t, r = _geometry(mesh_device)
    g = torch.Generator().manual_seed(c["seed"])
    H, D = c["heads"], c["head_dim"]
    k_nat = torch.randn(t, D, generator=g).to(torch.bfloat16)
    garbage = torch.randn(1, 1, t, D, generator=g).to(torch.bfloat16)
    pos = ref.k_local_positions(t, sp, 1, w // r)  # [sp, t / sp]
    k_local = k_nat[pos.reshape(-1)].reshape(sp, 1, t // sp, D)  # SP rank r: rows [r t / sp, ...)
    tt_k_local = _put(mesh_device, k_local, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, None)))
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
    return k_nat, garbage, tt_k_local, sems, g


def _queries(mesh_device, g):
    """Fresh q [1, H, S, D] / w [1, 1, S, H] in chunk order; device (d, c) gets rows d W + c Sq + [0, Sq)."""
    c = CASE
    sp, tp, s, w, sq, t, r = _geometry(mesh_device)
    q = torch.randn(1, c["heads"], s, c["head_dim"], generator=g).to(torch.bfloat16)
    wt = (torch.randn(1, 1, s, c["heads"], generator=g) * c["gate_scale"]).to(torch.bfloat16)
    # mesh (d, c) <- block d * tp + c along the sequence: shard dim 2 over both axes (row-major)
    qd = (
        q.reshape(1, c["heads"], sp, tp, sq, c["head_dim"])
        .permute(2, 3, 1, 4, 5, 0)
        .reshape(sp, tp, c["heads"], sq, -1)
    )
    wd = wt.reshape(1, 1, sp, tp, sq, c["heads"]).permute(2, 3, 0, 1, 4, 5).reshape(sp, tp, 1, sq, c["heads"])
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, 1))
    tt_q = _put(mesh_device, qd.reshape(sp, tp * c["heads"], sq, -1), mapper)
    tt_w = _put(mesh_device, wd.reshape(sp, tp, sq, c["heads"]), mapper)
    return q, wt, tt_q, tt_w


def _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, start, natural=True, **extra):
    sp, tp, s, w, sq, t, r = _geometry(mesh_device)
    tt_k = _put(mesh_device, garbage, ttnn.ReplicateTensorToMesh(mesh_device))
    kv_len = min(t, -(-((start + s) // r) // 32) * 32)
    kw = dict(
        cluster_axis=0,
        topology=ttnn.Topology.Linear,
        num_links=1,
        chunk_start_idx=start,
        key_compression_ratio=r,
        program_config=ttnn.bringup.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=32, head_group_size=0),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=False,
        ),
        kv_len=kv_len,
        seq_subshard_axis=1,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=w,
        natural_query_blocks=natural,
    )
    kw.update(extra)
    return ttnn.bringup.ring_indexer_score_dsa(tt_q, tt_k, tt_w, tt_k_local, sems, **kw), kv_len


def _check(mesh_device, out, q, wt, k_nat, start, kv_len, tag):
    sp, tp, s, w, sq, t, r = _geometry(mesh_device)
    outs = [ttnn.to_torch(x).float().reshape(sq, t) for x in ttnn.get_device_tensors(out)]
    assert len(outs) == sp * tp
    for d in range(sp):
        for c in range(tp):
            blk = slice(d * w + c * sq, d * w + (c + 1) * sq)
            q0 = start + d * w + c * sq
            want = ref.dsa_score(q[0, :, blk], k_nat, wt[0, 0, blk], q0, key_stride=r)[:, :kv_len]
            got = outs[d * tp + c][:, :kv_len]
            future = torch.isneginf(want)
            n_mask = int((~torch.isneginf(got[future])).sum())
            n_nonfinite = int((~torch.isfinite(got[~future])).sum())
            assert n_mask == 0, f"{tag} start {start} dev ({d}, {c}): {n_mask} future columns not -inf"
            assert n_nonfinite == 0, f"{tag} start {start} dev ({d}, {c}): {n_nonfinite} visible columns not finite"
            a, b = got[~future], want[~future]
            pcc, rel = _pcc(a, b), float((a - b).norm() / b.norm())
            logger.info(
                f"{tag} start {start} dev ({d}, {c}): tokens from {q0}, kv_len {kv_len}, {int((~future).sum())} "
                f"visible, pcc {pcc:.7f}, rel {rel:.5f}"
            )
            assert pcc >= CASE["pcc"] and rel <= CASE["rel"], f"{tag} start {start} dev ({d}, {c}): {pcc}, {rel}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    ids=["2x2-ratio2-natural-q512-t3072-h32-d128-fp32dest"],
    indirect=True,
)
def test_ratio2_natural_query_blocks(mesh_device, device_params, expect_error):
    k_nat, garbage, tt_k_local, sems, g = _setup(mesh_device)
    for start in CASE["starts"]:
        q, wt, tt_q, tt_w = _queries(mesh_device, g)
        out, kv_len = _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, start)
        _check(mesh_device, out, q, wt, k_nat, start, kv_len, "natural")
        ttnn.deallocate(out)
        if start % CASE["chunk"] == 0:  # slab-aligned: the source's rotation-exact geometry is the same
            out, kv_len = _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, start, natural=False)
            _check(mesh_device, out, q, wt, k_nat, start, kv_len, "rotation-exact")
            ttnn.deallocate(out)
        for x in (tt_q, tt_w):
            ttnn.deallocate(x)

    q, wt, tt_q, tt_w = _queries(mesh_device, g)
    with expect_error(RuntimeError, "natural_query_blocks needs a block-cyclic key cache"):
        _call(
            mesh_device,
            tt_q,
            tt_w,
            tt_k_local,
            sems,
            garbage,
            0,
            block_cyclic_sp_axis=None,
            block_cyclic_chunk_local=None,
            seq_subshard_axis=None,
        )
    meta = _put(
        mesh_device,
        torch.tensor([0], dtype=torch.int32),
        ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    with expect_error(RuntimeError, "natural_query_blocks needs the host-scalar chunk_start_idx"):
        _call(
            mesh_device,
            tt_q,
            tt_w,
            tt_k_local,
            sems,
            garbage,
            0,
            key_compression_ratio=1,
            chunk_start_idx=None,
            kv_len=None,
            chunk_start_idx_tensor=meta,
        )
