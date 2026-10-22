# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., stats_weight=W) on bf16 residual streams (deepseek_v41_flash_d_p P.20: the GPU code's bf16
residual streams). X and F bf16 (unpacked to the fp32 DEST exactly), X' written bf16 (the packer rounds the fp32 mix
once), the next hc step's partial row from the same pass: the projection X'_bf16 @ W on the FPU (the bf16 X' the next
step reads) and the sum of squares of the fp32 mix (before its rounding).

Per chip (random per-device inputs, the whole box, FABRIC_2D):
  * X' = the fp32-X call's X' (same values upcast) rounded to nearest, ties away (the packer), bit for bit;
  * the sum-of-squares column bit-identical to the fp32-X call's;
  * the projection columns vs float64 of bf16(X') @ W within 1.1 x the fp32-X call's own error vs float64 (the
    FPU's error at the given fidelity);
  * the plain program (no stats_weight) gives the same X';
  * with the packed-mixes offsets (post_col / comb_col) the same results;
  * negative control: the projection of the fp32 X' (not the bf16 one) is measurably different;
  * refusal: a bf16 residual with an fp32 F."""

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


def _dev(t, mesh, dtype=ttnn.float32, shard=True):
    return ttnn.from_torch(
        t.float().contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _rna_bf16(t):
    """fp32 -> bf16 rounded to nearest, ties away from zero (the packer's rounding)."""
    b = t.float().contiguous().view(torch.int32)
    sign = b & torch.tensor(-0x80000000, dtype=torch.int32)
    mag = ((b & 0x7FFFFFFF) + 0x8000) & ~0xFFFF
    return (sign | mag).view(torch.float32)


def _inputs(nd, T, C, seed):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(nd, 1, T, C, generator=g).bfloat16().float()
    x = (torch.randn(nd, 1, T, N * C, generator=g) * 3).bfloat16().float()
    mix = torch.rand(nd, 1, T, NG, generator=g)
    mix[..., N : 2 * N] *= 2
    w = torch.randn(1, 1, N * C, 32, generator=g) * (N * C) ** -0.5
    w[..., NG:] = 0  # as the model's hc_fn^T: 24 coefficient columns, zero padding
    return f, x, mix, w


CFG = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)
SHAPES = [(1024, 2560), (512, 1280), (64, 416)]  # DS41 2x2 / 4x4 per chip, small ragged


def _cols_rel(got, want):
    g, w = got.reshape(-1, got.shape[-1]).double(), want.reshape(-1, want.shape[-1]).double()
    return ((g - w).square().sum(0).sqrt() / w.square().sum(0).sqrt()).max().item()


@pytest.mark.parametrize("T,C", SHAPES)
@pytest.mark.parametrize("offsets", [False, True], ids=["sliced", "packed_mixes"])
def test_bf16_streams_stats(mesh, T, C, offsets):
    nd = mesh.get_num_devices()
    f, x, mix, w = _inputs(nd, T, C, seed=T + C + int(offsets))
    wd = _dev(w, mesh, shard=False)
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    if offsets:
        md = _dev(mix, mesh)
        pc = dict(post=md, comb=md, post_col=N, comb_col=2 * N)
    else:
        pc = dict(post=_dev(mix[..., N : 2 * N], mesh), comb=_dev(mix[..., 2 * N :], mesh), post_col=0, comb_col=0)

    def call(fdt, xdt, stats=True):
        args = (_dev(f, mesh, fdt), _dev(x, mesh, xdt), pc["post"], pc["comb"])
        kw = dict(post_col=pc["post_col"], comb_col=pc["comb_col"])
        if stats:
            return ttnn.bringup.mhc_post(*args, stats_weight=wd, compute_kernel_config=cfg, **kw)
        return ttnn.bringup.mhc_post(*args, **kw)

    xb_d, sb_d = call(ttnn.bfloat16, ttnn.bfloat16)
    assert xb_d.dtype == ttnn.bfloat16, xb_d.dtype
    xf_d, sf_d = call(ttnn.bfloat16, ttnn.float32)
    xb, sb, xf, sf = _host(xb_d, mesh), _host(sb_d, mesh), _host(xf_d, mesh), _host(sf_d, mesh)
    assert torch.equal(xb, _rna_bf16(xf)), "X' is not the fp32 X' rounded to nearest (ties away)"
    assert torch.equal(sb[..., NG], sf[..., NG]), "sum of squares differs from the fp32-X call's"
    assert torch.all(sb[..., NG + 1 :] == 0), "columns past n (n + 2) must be zero"
    plain = _host(call(ttnn.bfloat16, ttnn.bfloat16, stats=False), mesh)
    assert torch.equal(plain, xb), "the plain program's X' differs"
    want = (xb.double() @ w.double().reshape(N * C, 32))[..., :NG]
    r = _cols_rel(sb[..., :NG], want)
    r_f32 = _cols_rel(sb[..., :NG], (xf.double() @ w.double().reshape(N * C, 32))[..., :NG])
    # the fp32-X call's own projection error vs float64 (HiFi2 on the fp32 W: the FPU's fidelity, not this option's)
    r_own = _cols_rel(sf[..., :NG], (xf.double() @ w.double().reshape(N * C, 32))[..., :NG])
    print(
        f"T {T} C {C} offsets {offsets}: projection vs float64 bf16(X') @ W {r:.2e} (vs fp32 X' {r_f32:.2e}); "
        f"the fp32-X call vs its own float64 {r_own:.2e}"
    )
    assert r <= 1.1 * r_own, (r, r_own)
    assert r_f32 > r, "control: the projection should be of the bf16 X', not the fp32 one"


def test_bf16_streams_refusal(mesh, expect_error):
    nd = mesh.get_num_devices()
    f, x, mix, w = _inputs(nd, 64, 256, seed=5)
    md = _dev(mix, mesh)
    with expect_error(Exception, "or a bfloat16 residual with a bfloat16 input_tensor"):
        ttnn.bringup.mhc_post(
            _dev(f, mesh),
            _dev(x, mesh, ttnn.bfloat16),
            md,
            md,
            stats_weight=_dev(w, mesh, shard=False),
            compute_kernel_config=ttnn.ComputeConfigDescriptor(**CFG),
            post_col=N,
            comb_col=2 * N,
        )
