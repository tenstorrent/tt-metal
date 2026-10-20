# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., stats_weight=W) (deepseek_v41_flash_d_p P.14): the residual mix and, from the same pass,
the next mHC hc step's partial row stats = mhc_pre_xing_pack(X' @ W, X').

Per chip (random per-device inputs, the whole box, FABRIC_2D):
  * X' is bit-identical to mhc_post without stats_weight;
  * stats vs the unfused path on the device (ttnn.matmul(X', W) + mhc_pre_xing_pack): columns 0..n(n+2)-1 within
    1e-6 rel L2 (fp32 sums in another order; the per-product TF32 / srcA truncation is the same), the sum of squares
    within 1e-6;
  * stats vs float64 (X' from the device): the projection within the FPU's TF32 level (3e-3), the sum of squares
    within 1e-6;
  * negative control: a 1e-4 scale of W's columns or a dropped block shows; refusals."""

import pytest
import torch

import ttnn

N = 4


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _dev(t, mesh, dtype=ttnn.float32, shard=True):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).double()


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def _inputs(nd, T, C, n, seed):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(nd, 1, T, C, generator=g)
    x = torch.randn(nd, 1, T, n * C, generator=g) * 3
    post = 2 * torch.rand(nd, 1, T, n, generator=g)
    comb = torch.rand(nd, 1, T, n * n, generator=g)
    comb = comb / comb.reshape(nd, 1, T, n, n).sum(-2, keepdim=True).expand(nd, 1, T, n, n).reshape(nd, 1, T, n * n)
    ng = n * (n + 2)
    w = torch.zeros(1, 1, n * C, 32)
    w[..., :ng] = torch.randn(n * C, ng, generator=g) * (n * C) ** -0.5
    return f, x, post, comb, w


def _cc(mesh, fidelity):
    """(mhc_post's ComputeConfigDescriptor, ttnn.matmul's compute kernel config) at one fidelity."""
    return (
        ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False),
        ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        ),
    )


CASES = [
    # (T, C, n, comb_transposed, fidelity): DS41 2x2 (1024 rows x 2560 columns a chip), 4x4 (1280), small, ragged
    # column blocks (Ct = 13), n = 2, comb as stored, HiFi4
    (1024, 2560, 4, True, "HiFi2"),
    (1024, 1280, 4, True, "HiFi2"),
    (64, 256, 4, True, "HiFi2"),
    (128, 416, 4, True, "HiFi2"),
    (96, 512, 2, True, "HiFi2"),
    (256, 1024, 4, False, "HiFi4"),
]


@pytest.mark.parametrize("T, C, n, comb_t, fid", CASES, ids=[f"T{c[0]}-C{c[1]}-n{c[2]}-{c[3]}-{c[4]}" for c in CASES])
def test_mhc_post_stats(mesh, T, C, n, comb_t, fid):
    nd = mesh.get_num_devices()
    ng = n * (n + 2)
    f, x, post, comb, w = _inputs(nd, T, C, n, seed=T + C + n)
    f_d, x_d, p_d, c_d = (_dev(t, mesh) for t in (f, x, post, comb))
    w_d = _dev(w, mesh, shard=False)
    cc, mm_cc = _cc(mesh, getattr(ttnn.MathFidelity, fid))
    out, stats = ttnn.bringup.mhc_post(
        f_d, x_d, p_d, c_d, stats_weight=w_d, compute_kernel_config=cc, comb_transposed=comb_t
    )
    ref_out = ttnn.bringup.mhc_post(f_d, x_d, p_d, c_d, compute_kernel_config=cc, comb_transposed=comb_t)
    out_h = _host(out, mesh)
    assert torch.equal(out_h, _host(ref_out, mesh)), "X' differs from mhc_post without stats_weight"
    st = _host(stats, mesh)
    assert tuple(st.shape) == (nd, 1, T, 32), st.shape
    # the unfused device path
    mix = ttnn.matmul(out, w_d, dtype=ttnn.float32, compute_kernel_config=mm_cc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    packed = _host(ttnn.bringup.mhc_pre_xing_pack(mix, out, n=n), mesh)
    r_mix = _rel(st[..., :ng], packed[..., :ng])
    r_ss = _rel(st[..., ng], packed[..., ng])
    # float64
    want_mix = out_h @ w[0, 0, :, :ng].double()
    want_ss = out_h.square().sum(-1)
    r64_mix, r64_ss = _rel(st[..., :ng], want_mix), _rel(st[..., ng], want_ss)
    r64_unfused = _rel(packed[..., :ng], want_mix)  # the unfused path's own (TF32 / srcA) error
    print(
        f"T{T} C{C} n{n}: vs device path mix {r_mix:.2e} ss {r_ss:.2e}; vs float64 mix {r64_mix:.2e} "
        f"(unfused {r64_unfused:.2e}) ss {r64_ss:.2e}"
    )
    # measured (2x2 p150, 6 cases): vs device path mix <= 2.3e-6, ss <= 1.2e-7; vs float64 = the unfused path's
    assert r_mix <= 1e-5 and r_ss <= 1e-6, (r_mix, r_ss)
    assert r64_mix <= 1.05 * r64_unfused + 1e-6 and r64_ss <= 1e-6, (r64_mix, r64_unfused, r64_ss)
    assert torch.all(st[..., ng + 1 :] == 0), "columns past the sum of squares must stay W's zeros"
    # negative controls: a 0.1 % error in one projection column, a dropped column tile, fail the limits above
    bad = st[..., :ng].clone()
    bad[..., 0] *= 1 + 1e-3
    assert _rel(bad, packed[..., :ng]) > 1e-5, "a 0.1 % column error must fail the projection limit"
    x0 = out_h.clone()
    x0[..., :32] = 0
    assert _rel(st[..., ng], x0.square().sum(-1)) > 1e-6, "a dropped column tile must fail the sum-of-squares limit"


def test_mhc_post_stats_program_cache(mesh):
    """Two calls on fresh inputs (program cache hit, moved buffers) both match the unfused result."""
    nd = mesh.get_num_devices()
    for seed in (1, 2):
        f, x, post, comb, w = _inputs(nd, 256, 512, N, seed)
        f_d, x_d, p_d, c_d = (_dev(t, mesh) for t in (f, x, post, comb))
        w_d = _dev(w, mesh, shard=False)
        out, stats = ttnn.bringup.mhc_post(f_d, x_d, p_d, c_d, stats_weight=w_d)
        out_h = _host(out, mesh)
        assert _rel(_host(stats, mesh)[..., N * (N + 2)], out_h.square().sum(-1)) <= 1e-6


def test_mhc_post_stats_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    f, x, post, comb, w = _inputs(nd, 64, 256, N, 5)
    f_d, x_d, p_d, c_d = (_dev(t, mesh) for t in (f, x, post, comb))
    w_d = _dev(w, mesh, shard=False)
    with expect_error(Exception, "stats_weight"):
        ttnn.bringup.mhc_post(f_d, _dev(x, mesh, ttnn.bfloat16), p_d, c_d, stats_weight=w_d)
    with expect_error(Exception, "stats_weight"):
        ttnn.bringup.mhc_post(f_d, x_d, p_d, c_d, stats_weight=_dev(w[..., :128, :], mesh, shard=False))
    out = ttnn.bringup.mhc_post(f_d, x_d, p_d, c_d)  # without stats_weight: one tensor, as before
    assert isinstance(out, ttnn.Tensor)
