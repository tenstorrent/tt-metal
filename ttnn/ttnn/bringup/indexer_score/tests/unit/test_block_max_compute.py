# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ring_indexer_score_dsa(block_max_compute=True) (deepseek_v41_flash_d_p PB.FBM): the 32-key block maxes
made by the COMPUTE kernel (a row reduce of each masked score tile) must be byte-identical to the writer path's
(block_keys 8, fp32 block_max_out, folded to blocks of 32 and narrowed to bf16, the PB.BM chain) and to torch's max of
the same call's plain bf16 scores, on every chip, at several starts; the scores themselves are unchanged (bit for bit);
blocks past kv_len / 32 keep the buffer's prior bytes (512-byte pages of 8 blocks: [Sq / 32 x NB / 8, 256]); ttnn.bringup.block_max_bf16(strip_major=True) turns the
strip-major output into [Sq, NB] with the same bytes. Geometry from the mesh (SP ring on axis 0, TP query sub-shard on
axis 1, block-cyclic key cache, natural query blocks), so ``check_block_max_compute(mesh)`` also runs on the model's
resident mesh (the model's tests/perf/test_fbm_probe.py). Refusals: block_max_compute without block_keys 32, with a pin,
with an fp32 / wrongly shaped buffer."""

import pytest
import torch
from loguru import logger

import ttnn

SENT = 0x7B7B  # bf16 bytes the op must leave alone past kv_len


def _put(mesh, t, mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(),
        device=mesh,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _geometry(mesh, ratio: int, sq: int, chunks: int):
    sp, tp = int(mesh.shape[0]), int(mesh.shape[1])
    w = sq * tp
    s = w * sp
    return sp, tp, s, w, sq, chunks * s // ratio


def _k_local_positions(t: int, sp: int, w_keys: int) -> torch.Tensor:
    """Key positions per SP shard of the block-cyclic cache: shard d holds stripes d, d + sp, ... of w_keys keys."""
    stripes = t // w_keys
    return torch.stack(
        [torch.cat([torch.arange(j * w_keys, (j + 1) * w_keys) for j in range(d, stripes, sp)]) for d in range(sp)]
    )


def check_block_max_compute(
    mesh, *, ratio: int = 2, sq: int = 64, chunks: int = 3, heads: int = 32, seed: int = 5, ckc=None, max_grid=None
) -> dict:
    sp, tp, s, w, _, t = _geometry(mesh, ratio, sq, chunks)
    D = 128
    g = torch.Generator().manual_seed(seed)
    k_nat = torch.randn(t, D, generator=g).to(torch.bfloat16)
    pos = _k_local_positions(t, sp, w // ratio)
    k_local = k_nat[pos.reshape(-1)].reshape(sp, 1, t // sp, D)
    tt_k_local = _put(mesh, k_local, ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, None)))
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    tt_k = _put(mesh, torch.zeros(1, 1, t, D), ttnn.ReplicateTensorToMesh(mesh))
    q = torch.randn(1, heads, s, D, generator=g).to(torch.bfloat16)
    wt = (torch.randn(1, 1, s, heads, generator=g) / 64).to(torch.bfloat16)
    qd = q.reshape(1, heads, sp, tp, sq, D).permute(2, 3, 1, 4, 5, 0).reshape(sp, tp * heads, sq, D)
    wd = wt.reshape(1, 1, sp, tp, sq, heads).permute(2, 3, 0, 1, 4, 5).reshape(sp, tp, sq, heads)
    mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, 1))
    tt_q, tt_w = _put(mesh, qd, mapper), _put(mesh, wd, mapper)
    nb = -(-(t // 32) // 32) * 32  # blocks of 32 keys, a multiple of 32
    if ckc is None:
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=False,
        )
    mg = max_grid if max_grid is not None else (0, 0)
    cfg = ttnn.bringup.IndexerScoreProgramConfig(
        q_chunk_size=64 if sq % 64 == 0 else 32,
        k_chunk_size=32,
        head_group_size=0,
        max_grid_x=mg[0],
        max_grid_y=mg[1],
    )

    def call(start, **extra):
        kv_len = min(t, -(-((start + s) // ratio) // 32) * 32)
        out = ttnn.bringup.ring_indexer_score_dsa(
            tt_q,
            tt_k,
            tt_w,
            tt_k_local,
            sems,
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            num_links=1,
            chunk_start_idx=start,
            key_compression_ratio=ratio,
            program_config=cfg,
            compute_kernel_config=ckc,
            kv_len=kv_len,
            seq_subshard_axis=1 if tp > 1 else None,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=w,
            natural_query_blocks=True,
            **extra,
        )
        return out, kv_len

    def chips(x, rows, cols, dt=None):
        return [ttnn.to_torch(c).reshape(rows, cols) for c in ttnn.get_device_tensors(x)]

    starts = sorted({0, 32, s + 32 * tp, s * (chunks - 1)})
    n_cases = 0
    for start in starts:
        plain, kv_len = call(start)
        nbv = kv_len // 32
        base = chips(plain, sq, t)
        bm8 = _put(
            mesh, torch.zeros(1, 1, sq, nb * 4), ttnn.ReplicateTensorToMesh(mesh), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT
        )
        out8, _ = call(start, block_keys=8, block_max_out=bm8)
        strip_prev = torch.full((1, 1, sq // 32 * nb // 8, 256), 0.0, dtype=torch.bfloat16)  # 512 B pages
        strip_prev.view(torch.int16).fill_(SENT)
        strip = _put(mesh, strip_prev, ttnn.ReplicateTensorToMesh(mesh), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        outc, _ = call(start, block_keys=32, block_max_out=strip, block_max_compute=True)
        bm_dev = _put(
            mesh, torch.zeros(1, 1, sq, nb), ttnn.ReplicateTensorToMesh(mesh), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.bringup.block_max_bf16(strip, strip_major=True, n=nbv, output_tensor=bm_dev)
        got_c, got_s, got_8, got_bm = (
            chips(outc, sq, t),
            chips(strip, sq // 32, nb * 32),
            chips(bm8, sq, nb * 4),
            chips(bm_dev, sq, nb),
        )
        for i in range(sp * tp):
            p = base[i][:, :kv_len]
            assert torch.equal(got_c[i][:, :kv_len].view(torch.int16), p.view(torch.int16)), (start, i, "scores")
            want = p.float().reshape(sq, nbv, 32).amax(-1).to(torch.bfloat16)  # torch's max of the written scores
            writer = got_8[i][:, : nbv * 4].reshape(sq, nbv, 4).amax(-1).to(torch.bfloat16)  # PB.BM's fold
            comp = got_s[i].reshape(sq // 32, nb, 32).permute(0, 2, 1).reshape(sq, nb)
            bad_w = (comp[:, :nbv].view(torch.int16) != writer.view(torch.int16)).nonzero()
            if len(bad_w):
                r_, c_ = int(bad_w[0][0]), int(bad_w[0][1])
                logger.error(
                    f"start {start} chip {i}: {len(bad_w)} block maxes differ from the writer path, first "
                    f"({r_}, {c_}): compute {comp[r_, c_].item()} writer {writer[r_, c_].item()} torch "
                    f"{want[r_, c_].item()} scores {p[r_, c_ * 32 : c_ * 32 + 32].tolist()}"
                )
            assert not len(bad_w), (start, i, "compute vs writer block maxes")
            assert torch.equal(comp[:, :nbv].view(torch.int16), want.view(torch.int16)), (start, i, "vs torch")
            rest = got_s[i].reshape(sq // 32, nb, 32)[:, nbv:].reshape(-1).view(torch.int16)
            assert bool((rest == SENT).all()), (start, i, "written past kv_len")
            assert torch.equal(got_bm[i][:, :nbv].view(torch.int16), comp[:, :nbv].view(torch.int16)), (start, i)
            n_cases += 1
        logger.info(f"block_max_compute start {start}: kv_len {kv_len}, {nbv} blocks: exact on {sp * tp} chips")
        for x in (plain, out8, outc, bm8, strip, bm_dev):
            ttnn.deallocate(x)
    for x in (tt_q, tt_w, tt_k, tt_k_local):
        ttnn.deallocate(x)
    return {"cases": n_cases, "starts": len(starts)}


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    ids=["2x2"],
    indirect=True,
)
@pytest.mark.parametrize("ratio", [2, 1], ids=["r2", "r1"])
def test_block_max_compute(mesh_device, device_params, ratio):
    if not ttnn.device.is_blackhole():
        pytest.skip("indexer_score is Blackhole-only")
    assert check_block_max_compute(mesh_device, ratio=ratio)["cases"] >= 4


def check_block_max_compute_refusals(mesh, expect_error) -> None:
    sp, tp, s, w, sq, t = _geometry(mesh, 2, 64, 3)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    D, H = 128, 32
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, 1))
    tt_q = _put(mesh, torch.zeros(sp, tp * H, sq, D), mapper)
    tt_w = _put(mesh, torch.zeros(sp, tp, sq, H), mapper)
    tt_k = _put(mesh, torch.zeros(1, 1, t, D), rep)
    tt_kl = _put(
        mesh, torch.zeros(sp, 1, t // sp, D), ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, None))
    )
    nb = -(-(t // 32) // 32) * 32
    strip = _put(mesh, torch.zeros(1, 1, sq // 32 * nb // 8, 256), rep, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    strip32 = _put(mesh, torch.zeros(1, 1, sq // 32 * nb // 8, 256), rep, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    rows = _put(mesh, torch.zeros(1, 1, sq, nb), rep, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    def call(**kw):
        return ttnn.bringup.ring_indexer_score_dsa(
            tt_q,
            tt_k,
            tt_w,
            tt_kl,
            sems,
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            chunk_start_idx=0,
            key_compression_ratio=2,
            program_config=ttnn.bringup.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=32, head_group_size=0),
            kv_len=s // 2,
            seq_subshard_axis=1 if tp > 1 else None,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=w,
            natural_query_blocks=True,
            **kw,
        )

    with expect_error(RuntimeError, "block_max_compute needs block_max_out"):
        call(block_max_compute=True)
    with expect_error(RuntimeError, "block_max_compute: needs block_keys 32"):
        call(block_keys=8, block_max_out=strip, block_max_compute=True)
    with expect_error(RuntimeError, "block_max_compute: needs block_keys 32"):
        call(block_keys=32, block_max_out=strip, block_max_pin=True, block_max_compute=True)
    with expect_error(RuntimeError, "block_max_out must be a DRAM-interleaved ROW_MAJOR bf16"):
        call(block_keys=32, block_max_out=strip32, block_max_compute=True)
    with expect_error(RuntimeError, "block_max_out must be \\[1, 1, Sq / 32"):
        call(block_keys=32, block_max_out=rows, block_max_compute=True)
    for x in (tt_q, tt_w, tt_k, tt_kl, strip, strip32, rows):
        ttnn.deallocate(x)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    ids=["2x2"],
    indirect=True,
)
def test_block_max_compute_refusals(mesh_device, device_params, expect_error):
    if not ttnn.device.is_blackhole():
        pytest.skip("indexer_score is Blackhole-only")
    check_block_max_compute_refusals(mesh_device, expect_error)
