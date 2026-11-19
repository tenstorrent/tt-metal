# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.grouped_matmul (deepseek_v41_flash_d_p P.81): y[g] = x[g] @ w[g], one grid row a group.

Whole box (FABRIC_2D), replicated random inputs: against the stock batched ttnn.matmul (2D multicast, fuse_batch False:
the path it replaces) and float64; the prepared program on fresh buffers (a cache hit), a caller's output buffer,
determinism; shapes: the 8x4's wo_a (8 x [160, 4096] @ [4096, 1024], bfloat8_b, 11 x 10 grid) and odd ones (one group,
a column count that does not divide N, a bf16 output, a narrow grid)."""

import pytest
import torch

import ttnn
from ttnn.bringup.grouped_matmul_ttnn import supported


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _ckc(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )


def _dev(t, mesh, dtype):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _chip(t, i=0):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[i]).double()


CASES = [  # (G, M, K, N, output dtype, grid, block_w, depth)
    (8, 160, 4096, 1024, ttnn.bfloat8_b, (11, 10), 4, 4),  # the 8x4's wo_a (deepseek_v41_flash_d_p, ATTN_ROOFLINE)
    (8, 160, 4096, 1024, ttnn.bfloat8_b, (11, 10), None, 2),
    (1, 64, 256, 96, ttnn.bfloat16, (11, 10), None, 2),
    (3, 96, 512, 352, ttnn.bfloat8_b, (4, 3), 8, 3),
    (2, 32, 1024, 64, ttnn.bfloat16, (1, 2), None, 2),
]
IDS = [f"G{c[0]}-M{c[1]}-K{c[2]}-N{c[3]}-bw{c[6]}-d{c[7]}-{c[5][0]}x{c[5][1]}" for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_grouped_matmul(mesh, case):
    check(mesh, *case)


def check(mesh, G, M, K, N, odt, grid, bw=None, depth=2):
    g = torch.Generator().manual_seed(G * 7 + M)
    xt = torch.randn(1, G, M, K, generator=g)
    wt = torch.randn(1, G, K, N, generator=g) / K**0.5
    x = _dev(xt, mesh, ttnn.bfloat8_b)
    w = _dev(wt, mesh, ttnn.bfloat8_b)
    mg = ttnn.CoreCoord(*grid)
    assert supported(tuple(x.shape), tuple(w.shape), x.dtype, w.dtype, odt, mesh, mg, bw)
    ckc = _ckc(mesh)
    kw = dict(dtype=odt, compute_kernel_config=ckc, max_grid=mg, block_w=bw, depth=depth)
    y = ttnn.bringup.grouped_matmul(x, w, **kw)
    truth = _chip(x) @ _chip(w)
    ref = ttnn.matmul(x, w, dtype=odt, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = _chip(y)
    rel = float((got - truth).norm() / truth.norm())
    rel_ref = float((_chip(ref) - truth).norm() / truth.norm())
    d = (got - _chip(ref)).abs()
    print(
        f"[gmm] bw{bw} d{depth} G{G} M{M} K{K} N{N} {odt} grid{grid}: rel vs fp64 {rel:.3e} (stock {rel_ref:.3e}), max|d| vs stock "
        f"{float(d.max()):.3e} ({int((d > 0).sum())} of {d.numel()} differ)"
    )
    assert rel <= rel_ref * 1.05 + 1e-6, (rel, rel_ref)
    # every chip the same
    for i in range(1, len(ttnn.get_device_tensors(y))):
        assert torch.equal(_chip(y, i), got)
    # a cache hit on fresh buffers, into a caller's output: the same bytes (determinism)
    x2 = _dev(xt, mesh, ttnn.bfloat8_b)
    w2 = _dev(wt, mesh, ttnn.bfloat8_b)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, G, M, N]), odt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    y2 = ttnn.bringup.grouped_matmul(x2, w2, output_tensor=out, **kw)
    assert y2 is out
    assert torch.equal(_chip(y2), got)


def test_grouped_matmul_refuses(mesh):
    check_refuses(mesh)


def check_refuses(mesh):
    x = (1, 12, 64, 256)
    w = (1, 12, 256, 64)
    assert not supported(x, w, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, mesh, ttnn.CoreCoord(11, 10))
    assert not supported((1, 2, 64, 256), (1, 2, 128, 64), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, mesh)
