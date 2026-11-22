# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.block_gather / block_keys (deepseek_v41_flash_d_p PB.BM): random scores with tied and -inf runs, every
block size 1..32, ids with none (0xFFFFFFFF) and out-of-range entries, widths that are and are not a multiple of 64
values; byte-exact against torch; the prepared program on fresh buffers (a cache hit) and a caller's output buffer;
determinism. ``check_block_select(mesh)`` is also called by the model's mesh-only probe on the resident runner."""

import pytest
import torch

import ttnn

SENT = 0xFFFFFFFF


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _up(t, mesh, dtype):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _down(t, mesh):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _u32(t):
    return t.to(torch.int64) & 0xFFFFFFFF


def ref_gather(s: torch.Tensor, ids: torch.Tensor, B: int) -> torch.Tensor:
    R, W = s.shape[-2:]
    nbw = W // B
    ids = _u32(ids)
    ok = ids < nbw
    cols = (ids.clamp(max=nbw - 1) * B).unsqueeze(-1) + torch.arange(B)
    v = torch.gather(s, -1, cols.reshape(*ids.shape[:-1], -1)).reshape(*ids.shape, B)
    v = torch.where(ok.unsqueeze(-1), v, torch.tensor(float("-inf"), dtype=s.dtype))
    return v.reshape(*ids.shape[:-1], -1)


def ref_keys(pos: torch.Tensor, ids: torch.Tensor, B: int, width: int) -> torch.Tensor:
    pos, ids = _u32(pos), _u32(ids)
    K = ids.shape[-1]
    okp = pos < K * B
    blk = torch.gather(ids, -1, (pos.clamp(max=K * B - 1) // B))
    ok = okp & (blk < width // B)
    return torch.where(ok, blk * B + pos % B, torch.full_like(pos, SENT))


def check_block_select(mesh, rows: int = 96, seed: int = 0) -> dict:
    """Every case byte-exact (asserted); returns counts."""
    g = torch.Generator().manual_seed(seed)
    n = 0
    for W, K, B in ((4096, 512, 8), (2080, 64, 1), (8192, 128, 32), (1024, 64, 16), (65536, 512, 8), (512, 16, 2)):
        if K * B > W * 4:
            continue
        s = torch.randn(1, 1, rows, W, generator=g).to(torch.bfloat16)
        s[..., :, ::7] = 0.5  # ties
        s[..., 3, W // 2 :] = float("-inf")  # a causal tail
        nb = W // B
        ids = torch.randint(0, nb, (1, 1, rows, K), generator=g)
        ids[..., 0, -5:] = SENT  # none (int64 0xFFFFFFFF: -1 as int32)
        ids[..., 1, 3] = nb  # just out of range
        ids[..., 2, :] = 0  # one block repeated
        ids_t = ids.to(torch.int64)
        st, it = _up(s, mesh, ttnn.bfloat16), _up(ids_t.to(torch.int32), mesh, ttnn.uint32)
        got = ttnn.bringup.block_gather(st, it, block=B)
        want = ref_gather(s, ids_t, B)
        assert torch.equal(_down(got, mesh).view(torch.int16), want.view(torch.int16)), (W, K, B)
        again = ttnn.bringup.block_gather(st, it, block=B, output_tensor=got)  # prepared hit, caller's buffer
        assert again is got and torch.equal(_down(again, mesh).view(torch.int16), want.view(torch.int16))
        pos = torch.randint(0, K * B, (1, 1, rows, min(512, K * B)), generator=g)
        pos[..., 4, :7] = SENT
        pt = _up(pos.to(torch.int32), mesh, ttnn.uint32)
        keys = ttnn.bringup.block_keys(pt, it, block=B, width=W)
        wk = ref_keys(pos, ids_t, B, W)
        assert torch.equal(_u32(_down(keys, mesh)), wk), (W, K, B)
        for t in (st, it, got, pt, keys):
            ttnn.deallocate(t)
        n += 1
    # block_max_bf16: fp32 maxes of bf16 values (ties, -inf, negative rows), factor 1 / 4, n as a value / a tensor
    for NI, F, nn in ((8192, 4, 1000), (8192, 1, 8192), (4096, 4, 3), (2048, 16, 128)):
        x = torch.randn(1, 1, rows, NI, generator=g).to(torch.bfloat16).float()
        x[..., 0, :] = -x[..., 0, :].abs() - 1  # every value negative
        x[..., 1, ::3] = float("-inf")
        x[..., 2, :] = float("-inf")
        x = x.to(torch.bfloat16).float()  # (bf16 values widened, as the score op's block maxes)
        xt = _up(x, mesh, ttnn.float32)
        want = x.view(1, 1, rows, NI // F, F).amax(-1).to(torch.bfloat16)
        m = min(-(-nn // 8) * 8, NI // F)
        for use_t in (False, True):
            prev = torch.full((1, 1, rows, NI // F), 7.0, dtype=torch.bfloat16)
            ot = _up(prev, mesh, ttnn.bfloat16)
            if use_t:
                lt = _up(torch.tensor([nn], dtype=torch.int32).view(1, 1, 1, 1), mesh, ttnn.uint32)
                got = ttnn.bringup.block_max_bf16(xt, factor=F, n_tensor=lt, output_tensor=ot)
                ttnn.deallocate(lt)
            else:
                got = ttnn.bringup.block_max_bf16(xt, factor=F, n=nn, output_tensor=ot)
            d = _down(got, mesh)
            bad = (d[..., :m].view(torch.int16) != want[..., :m].view(torch.int16)).nonzero()
            if len(bad):
                r_, c_ = int(bad[0][2]), int(bad[0][3])
                print(
                    "block_max_bf16 mismatch",
                    (NI, F, nn, use_t),
                    len(bad),
                    bad[:8].tolist(),
                    x[0, 0, r_, c_ * F : c_ * F + F].tolist(),
                    d[0, 0, r_, c_].item(),
                    want[0, 0, r_, c_].item(),
                )
            assert not len(bad), (NI, F, nn, use_t)
            assert torch.equal(d[..., m:].view(torch.int16), prev[..., m:].view(torch.int16)), "wrote past n"
            ttnn.deallocate(got)
        ttnn.deallocate(xt)
        n += 1
    # block_max_bf16(strip_major=True) (PB.FBM): [S, NB 32] strip-major bf16 -> [32 S, NB], the first n blocks (to 8)
    for S, NB, nn in ((8, 16384, 16384), (8, 2176, 2176), (3, 512, 77), (2, 264, 1), (5, 4096, 1000)):
        x = torch.randn(1, 1, S, NB * 32)
        x[..., 0, ::5] = float("-inf")
        x = x.to(torch.bfloat16)
        xt = _up(x.reshape(1, 1, S * NB // 8, 256), mesh, ttnn.bfloat16)  # (512-byte pages of 8 blocks)
        want = x.view(1, 1, S, NB, 32).permute(0, 1, 2, 4, 3).reshape(1, 1, S * 32, NB)
        m = min(-(-nn // 8) * 8, NB)
        for use_t in (False, True):
            prev = torch.full((1, 1, S * 32, NB), 7.0, dtype=torch.bfloat16)
            ot = _up(prev, mesh, ttnn.bfloat16)
            if use_t:
                lt = _up(torch.tensor([nn], dtype=torch.int32).view(1, 1, 1, 1), mesh, ttnn.uint32)
                got = ttnn.bringup.block_max_bf16(xt, strip_major=True, n_tensor=lt, output_tensor=ot)
                ttnn.deallocate(lt)
            else:
                got = ttnn.bringup.block_max_bf16(xt, strip_major=True, n=nn, output_tensor=ot)
            d = _down(got, mesh)
            bad = (d[..., :m].view(torch.int16) != want[..., :m].view(torch.int16)).nonzero()
            if len(bad):
                print("block_max_bf16 strip_major mismatch", (S, NB, nn, use_t), len(bad), bad[:8].tolist())
            assert not len(bad), (S, NB, nn, use_t)
            assert torch.equal(d[..., m:].view(torch.int16), prev[..., m:].view(torch.int16)), "wrote past n"
            ttnn.deallocate(got)
        ttnn.deallocate(xt)
        n += 1
    return {"cases": n}


def test_block_select(mesh):
    assert check_block_select(mesh)["cases"] >= 5
