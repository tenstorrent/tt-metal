# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., post_col=, comb_col=) and a bf16 F with stats_weight (deepseek_v41_flash_d_p P.19).

post_col / comb_col: post / comb read at column offsets of their tensors, so the packed mHC mixes [pre | post | comb]
are passed as both (post=comb=mixes, post_col=n, comb_col=2n): no slices. A bf16 F with stats_weight goes through its
own CB, unpacked to the fp32 DEST exactly (bf16 -> fp32 is exact).

Per chip (random per-device inputs, the whole box, FABRIC_2D):
  * offsets, plain program and stats program, comb^T and comb as stored, bf16 and fp32 F: X' (and stats)
    bit-identical to the call on the sliced post / comb;
  * bf16 F with stats_weight: X' and stats bit-identical to the call on F typecast to fp32 (and with the offsets);
  * negative control: offsets one column off give another X';
  * refusals (offsets beyond the tile row, a too-narrow tensor)."""

import pytest
import torch

import ttnn

N = 4
NG = N * (N + 2)


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _dev(t, mesh, dtype=ttnn.float32):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def _inputs(nd, T, C, seed):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(nd, 1, T, C, generator=g)
    x = torch.randn(nd, 1, T, N * C, generator=g) * 3
    mix = torch.rand(nd, 1, T, NG, generator=g)
    mix[..., N : 2 * N] *= 2
    w = torch.randn(1, 1, N * C, 32, generator=g) * (N * C) ** -0.5
    return f, x, mix, w


CFG = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)
SHAPES = [(1024, 2560), (512, 1280), (64, 416)]  # DS41 2x2 / 4x4 per-chip rows x hidden slice, small ragged


@pytest.mark.parametrize("T,C", SHAPES)
@pytest.mark.parametrize("comb_t", [True, False], ids=["combT", "comb_stored"])
@pytest.mark.parametrize("fdt", ["fp32", "bf16"])
def test_offsets_plain(mesh, T, C, comb_t, fdt):
    nd = mesh.get_num_devices()
    f, x, mix, _ = _inputs(nd, T, C, seed=T + C + int(comb_t))
    dt = ttnn.float32 if fdt == "fp32" else ttnn.bfloat16
    fd, xd, md = _dev(f, mesh, dt), _dev(x, mesh), _dev(mix, mesh)
    post, comb = _dev(mix[..., N : 2 * N], mesh), _dev(mix[..., 2 * N :], mesh)
    want = _host(ttnn.bringup.mhc_post(fd, xd, post, comb, comb_transposed=comb_t), mesh)
    got = _host(ttnn.bringup.mhc_post(fd, xd, md, md, comb_transposed=comb_t, post_col=N, comb_col=2 * N), mesh)
    assert torch.equal(got, want), "offsets: X' differs from the sliced call"
    off = _host(ttnn.bringup.mhc_post(fd, xd, md, md, comb_transposed=comb_t, post_col=N - 1, comb_col=2 * N), mesh)
    assert not torch.equal(off, want), "control: a post offset one column off gives the same X'"


@pytest.mark.parametrize("T,C", SHAPES)
@pytest.mark.parametrize("comb_t", [True, False], ids=["combT", "comb_stored"])
def test_offsets_and_bf16_stats(mesh, T, C, comb_t):
    nd = mesh.get_num_devices()
    f, x, mix, w = _inputs(nd, T, C, seed=3 * T + C + int(comb_t))
    xd, md = _dev(x, mesh), _dev(mix, mesh)
    wd = ttnn.from_torch(
        w,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    post, comb = _dev(mix[..., N : 2 * N], mesh), _dev(mix[..., 2 * N :], mesh)
    fb = _dev(f, mesh, ttnn.bfloat16)
    f32 = _dev(f.bfloat16().float(), mesh)  # = typecast(fb, fp32): bf16 -> fp32 is exact
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    kw = dict(stats_weight=wd, compute_kernel_config=cfg, comb_transposed=comb_t)
    x0, s0 = ttnn.bringup.mhc_post(f32, xd, post, comb, **kw)
    x0, s0 = _host(x0, mesh), _host(s0, mesh)
    for name, fin, p, c, extra in (
        ("offsets", f32, md, md, dict(post_col=N, comb_col=2 * N)),
        ("bf16 F", fb, post, comb, {}),
        ("bf16 F + offsets", fb, md, md, dict(post_col=N, comb_col=2 * N)),
    ):
        x1, s1 = ttnn.bringup.mhc_post(fin, xd, p, c, **kw, **extra)
        assert torch.equal(_host(x1, mesh), x0), f"{name}: X' differs"
        assert torch.equal(_host(s1, mesh), s0), f"{name}: stats differ"
    x2, _ = ttnn.bringup.mhc_post(fb, xd, md, md, **kw, post_col=N, comb_col=2 * N - 1)
    assert not torch.equal(_host(x2, mesh), x0), "control: a comb offset one column off gives the same X'"


def test_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    f, x, mix, _ = _inputs(nd, 64, 256, seed=1)
    fd, xd, md = _dev(f, mesh), _dev(x, mesh), _dev(mix, mesh)
    narrow = _dev(mix[..., :20], mesh)
    with expect_error(Exception, "must keep post and comb in one tile row"):
        ttnn.bringup.mhc_post(fd, xd, md, md, post_col=N, comb_col=20)  # comb beyond the tile row (20 + 16 > 32)
    with expect_error(Exception, "comb last dim"):
        ttnn.bringup.mhc_post(fd, xd, narrow, narrow, post_col=N, comb_col=2 * N)  # comb_col + n*n > 20 columns
