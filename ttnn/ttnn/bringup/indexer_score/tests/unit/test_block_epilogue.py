# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ring_indexer_score_dsa's block epilogue (fork options block_keys / block_bias / block_max_out /
block_max_pin, deepseek_v41_flash_d_p P.16) on a 2x2 SP x TP mesh, DeepSeek-V4.1's layer-20 / re-index geometry:
ratio 1 keys, natural query blocks, SP ring on axis 0, TP query sub-shard on axis 1, block-cyclic key cache.

The epilogue works on the scores the op writes, so every check compares against the SAME op's plain scores (no
epilogue, same inputs), exactly:
    block_bias      columns [0, kv_len) of the biased scores == the plain scores where the key's block bias is 0,
                    -inf where it is -inf (blocks of block_keys logical keys)
    block_max_out   column b < kv_len / block_keys == max of the plain (or biased) bf16 scores of block b, as fp32;
                    columns past kv_len / block_keys keep the buffer's prior value (a sentinel)
    block_max_pin   the query's own block (the block of its newest key, position // block_keys) is +inf
    plain output    unchanged by block_max_out (the scores are the plain call's, bit for bit)
Starts: chunk-aligned, mid-W, mid-slab and the last window of the cache; block_keys 8 (DeepSeek) and 4.
Refusals: block_keys without a tensor, a tensor without block_keys, block_max_pin without block_max_out, a bf16
block_max_out, block_keys 16 with block_max_out, a too-narrow block_bias.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_TESTS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_indexer_score_tests_bep_{name}", _TESTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASE = {
    "mesh": [2, 2],
    "heads": 32,
    "head_dim": 128,
    "chunk": 2048,  # S (tokens); W = 1024, Sq = 512
    "chunks": 3,  # T = 6144 keys (ratio 1)
    "ratio": 1,
    "starts": [0, 32, 1088, 2048, 2592, 4064],
    "gate_scale": 1.0 / 64,
    "seed": 11,
}
SENTINEL = 12345.0

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")


def _put(mesh_device, t, mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t, device=mesh_device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def _geometry():
    sp, tp = CASE["mesh"]
    s, r = CASE["chunk"], CASE["ratio"]
    w = s // sp
    return sp, tp, s, w, w // tp, CASE["chunks"] * s // r, r


def _setup(mesh_device):
    sp, tp, s, w, sq, t, r = _geometry()
    g = torch.Generator().manual_seed(CASE["seed"])
    D = CASE["head_dim"]
    k_nat = torch.randn(t, D, generator=g).to(torch.bfloat16)
    garbage = torch.randn(1, 1, t, D, generator=g).to(torch.bfloat16)
    pos = ref.k_local_positions(t, sp, 1, w // r)
    k_local = k_nat[pos.reshape(-1)].reshape(sp, 1, t // sp, D)
    tt_k_local = _put(mesh_device, k_local, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, None)))
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
    return garbage, tt_k_local, sems, g


def _queries(mesh_device, g):
    sp, tp, s, w, sq, t, r = _geometry()
    H, D = CASE["heads"], CASE["head_dim"]
    q = torch.randn(1, H, s, D, generator=g).to(torch.bfloat16)
    wt = (torch.randn(1, 1, s, H, generator=g) * CASE["gate_scale"]).to(torch.bfloat16)
    qd = q.reshape(1, H, sp, tp, sq, D).permute(2, 3, 1, 4, 5, 0).reshape(sp, tp, H, sq, -1)
    wd = wt.reshape(1, 1, sp, tp, sq, H).permute(2, 3, 0, 1, 4, 5).reshape(sp, tp, 1, sq, H)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, 1))
    return _put(mesh_device, qd.reshape(sp, tp * H, sq, -1), mapper), _put(
        mesh_device, wd.reshape(sp, tp, sq, H), mapper
    )


def _per_chip(mesh_device, t_host, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    """[sp, tp, Sq, n] host -> [1, 1, Sq, n] per chip (chip (d, c) its own rows)."""
    sp, tp = CASE["mesh"]
    return _put(
        mesh_device,
        t_host.reshape(sp, tp, t_host.shape[-2], t_host.shape[-1]).contiguous(),
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, 1)),
        dtype=dtype,
        layout=layout,
    )


def _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, start, **extra):
    sp, tp, s, w, sq, t, r = _geometry()
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
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=False,
        ),
        kv_len=kv_len,
        seq_subshard_axis=1,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=w,
        natural_query_blocks=True,
    )
    kw.update(extra)
    out = ttnn.bringup.ring_indexer_score_dsa(tt_q, tt_k, tt_w, tt_k_local, sems, **kw)
    ttnn.deallocate(tt_k)
    return out, kv_len


def _chips(t, rows, cols):
    return [ttnn.to_torch(x).reshape(rows, cols) for x in ttnn.get_device_tensors(t)]


def _block_max(scores_bf16: torch.Tensor, bk: int) -> torch.Tensor:
    return scores_bf16.float().reshape(scores_bf16.shape[0], -1, bk).amax(dim=-1)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("block_keys", [8, 4], ids=["bk8", "bk4"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    ids=["2x2-ratio1-natural-q512-t6144-h32-d128"],
    indirect=True,
)
def test_block_epilogue(mesh_device, device_params, block_keys):
    sp, tp, s, w, sq, t, r = _geometry()
    bk = block_keys
    nb = -(-(t // bk) // 32) * 32  # bias / max columns (>= T / bk, multiple of 32)
    garbage, tt_k_local, sems, g = _setup(mesh_device)
    for start in CASE["starts"]:
        tt_q, tt_w = _queries(mesh_device, g)
        plain, kv_len = _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, start)
        nbv = kv_len // bk  # blocks of the written columns
        base = _chips(plain, sq, t)
        # bias: about half the blocks -inf, per chip row
        keep = torch.rand(sp, tp, sq, nb, generator=g) < 0.5
        bias = torch.where(keep, 0.0, float("-inf")).to(torch.bfloat16)
        tt_bias = _per_chip(mesh_device, bias, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        tt_max = _per_chip(mesh_device, torch.full((sp, tp, sq, nb), SENTINEL), ttnn.float32)
        # (a) block max (+ pin) of the plain scores; the scores themselves unchanged
        out, _ = _call(
            mesh_device,
            tt_q,
            tt_w,
            tt_k_local,
            sems,
            garbage,
            start,
            block_keys=bk,
            block_max_out=tt_max,
            block_max_pin=True,
        )
        got_scores, got_max = _chips(out, sq, t), _chips(tt_max, sq, nb)
        # (b) bias (+ block max of the biased scores, no pin)
        tt_max2 = _per_chip(mesh_device, torch.full((sp, tp, sq, nb), SENTINEL), ttnn.float32)
        out2, _ = _call(
            mesh_device,
            tt_q,
            tt_w,
            tt_k_local,
            sems,
            garbage,
            start,
            block_keys=bk,
            block_bias=tt_bias,
            block_max_out=tt_max2,
        )
        got_b, got_bmax = _chips(out2, sq, t), _chips(tt_max2, sq, nb)
        for i in range(sp * tp):
            d, c = divmod(i, tp)
            p = base[i][:, :kv_len]
            assert torch.equal(got_scores[i][:, :kv_len].view(torch.int16), p.view(torch.int16)), (start, i)
            want = _block_max(p, bk)
            q_pos = start + d * w + c * sq + torch.arange(sq)
            want[torch.arange(sq), q_pos // bk] = float("inf")
            assert torch.equal(got_max[i][:, :nbv], want), f"start {start} chip {i}: block max (pin)"
            assert bool((got_max[i][:, nbv:] == SENTINEL).all()), f"start {start} chip {i}: columns past kv_len"
            masked = ~keep[d, c][:, :nbv].repeat_interleave(bk, dim=-1)
            want_b = torch.where(masked, torch.tensor(float("-inf"), dtype=torch.bfloat16), p)
            assert torch.equal(got_b[i][:, :kv_len].view(torch.int16), want_b.view(torch.int16)), (start, i)
            assert torch.equal(got_bmax[i][:, :nbv], _block_max(want_b, bk)), f"start {start} chip {i}: biased max"
            assert bool((got_bmax[i][:, nbv:] == SENTINEL).all())
        logger.info(
            f"bk {bk} start {start}: kv_len {kv_len}, {nbv} blocks: bias / block max / pin exact on {sp * tp} chips"
        )
        for x in (plain, out, out2, tt_bias, tt_max, tt_max2, tt_q, tt_w):
            ttnn.deallocate(x)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    ids=["2x2"],
    indirect=True,
)
def test_block_epilogue_refusals(mesh_device, device_params, expect_error):
    sp, tp, s, w, sq, t, r = _geometry()
    garbage, tt_k_local, sems, g = _setup(mesh_device)
    tt_q, tt_w = _queries(mesh_device, g)
    nb = t // 8
    bias = _per_chip(mesh_device, torch.zeros(sp, tp, sq, nb), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    mx = _per_chip(mesh_device, torch.zeros(sp, tp, sq, nb), ttnn.float32)
    mx16 = _per_chip(mesh_device, torch.zeros(sp, tp, sq, t // 16), ttnn.float32)
    mx_bf16 = _per_chip(mesh_device, torch.zeros(sp, tp, sq, nb), ttnn.bfloat16)
    narrow = _per_chip(mesh_device, torch.zeros(sp, tp, sq, nb - 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    bias_rm = _per_chip(mesh_device, torch.zeros(sp, tp, sq, nb), ttnn.bfloat16)
    call = lambda **kw: _call(mesh_device, tt_q, tt_w, tt_k_local, sems, garbage, 0, **kw)  # noqa: E731
    with expect_error(RuntimeError, "must be > 0 exactly when block_bias or block_max_out"):
        call(block_keys=8)
    with expect_error(RuntimeError, "must be > 0 exactly when block_bias or block_max_out"):
        call(block_bias=bias)
    with expect_error(RuntimeError, "block_max_pin needs block_max_out"):
        call(block_keys=8, block_bias=bias, block_max_pin=True)
    with expect_error(RuntimeError, "block_bias must be a DRAM-interleaved"):
        call(block_keys=8, block_bias=bias_rm)
    with expect_error(RuntimeError, "block_max_out must be a DRAM-interleaved"):
        call(block_keys=8, block_max_out=mx_bf16)
    with expect_error(RuntimeError, "block_max_out needs block_keys <= 8"):
        call(block_keys=16, block_max_out=mx16)
    with expect_error(RuntimeError, "block_bias must be \\[1, 1, Sq"):
        call(block_keys=8, block_bias=narrow)
