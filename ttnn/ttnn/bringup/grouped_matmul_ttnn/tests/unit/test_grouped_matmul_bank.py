# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.grouped_matmul, option ``w_layout`` (bank-local weights, bank.py; deepseek_v41_flash_d_p WA.2).

On the bring-up mesh (the resident runner's held mesh, BATCH_SHARED; run through
models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_fork_bank.py), replicated random inputs: the bank path against
today's interleaved path at the same knobs (bit-identical on every chip), against a torch reference, a repeat on fresh
buffers (same bytes, a prepared-program cache hit); shapes from the smallest up to the 8x4's wo_a (8 x [160 | 256,
4096] @ [4096, 1024], bfloat8_b); a G = 4 case (two banks a group); the CPU packing (every tile once, equal banks);
refusals."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize
from ttnn.bringup.grouped_matmul_ttnn import bank

BATCH_SHARED = True  # (a mesh-only op test: the resident runner's held mesh)


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


def _chips(t):
    return [ttnn.to_torch(d).double() for d in ttnn.get_device_tensors(t)]


def _maxdiff(a, b) -> float:
    return max(float((p - q).abs().max()) for p, q in zip(_chips(a), _chips(b)))


CASES = [  # (G, M, K, N, output dtype, block_w, depth, rt, cores_per_bank, w_inflight, x_ahead, shape, sender)
    (8, 32, 256, 1024, ttnn.bfloat8_b, 2, 2, 1, 12, 1, 0, "band", "start"),  # (smallest first)
    (8, 32, 256, 1024, ttnn.bfloat8_b, 2, 2, 1, 12, 2, 1, "band", "start"),
    (8, 32, 256, 1024, ttnn.bfloat8_b, 2, 3, 1, 12, 1, 2, "band", "anchor"),
    (8, 32, 256, 1024, ttnn.bfloat8_b, 2, 2, 1, 11, 2, 1, "row", "start"),
    (4, 64, 512, 1024, ttnn.bfloat16, 4, 3, 1, 12, 2, 1, "band", "start"),  # two banks a group
    (8, 96, 1024, 1024, ttnn.float32, 4, 4, 2, 10, 3, 3, "band", "start"),
    (8, 160, 4096, 1024, ttnn.bfloat8_b, 4, 4, 1, 12, 2, 2, "band", "start"),  # the 8x4's wo_a at 5120 (WO_A_GMM_*)
    (8, 256, 4096, 1024, ttnn.bfloat8_b, 2, 2, 1, 12, 1, 1, "band", "start"),  # the 8x4's wo_a at 8192 (GMM256)
]
IDS = [
    f"G{c[0]}-M{c[1]}-K{c[2]}-N{c[3]}-bw{c[5]}-d{c[6]}-rt{c[7]}-c{c[8]}-wi{c[9]}-xa{c[10]}-{c[11]}-{c[12]}"
    for c in CASES
]


@mesh_parametrize
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_grouped_matmul_bank(mesh_device, case):
    if mesh_device is None:
        pytest.skip("device test")
    check(mesh_device, *case)


def check(mesh, G, M, K, N, odt, bw, depth, rt, cpb, wi, xp, shape, sender):
    g = torch.Generator().manual_seed(G * 7 + M + K)
    xt = torch.randn(1, G, M, K, generator=g)
    wt = torch.randn(1, G, K, N, generator=g) / K**0.5
    grid = mesh.compute_with_storage_grid_size()
    lay = bank.plan(mesh, G, K, N, max_cores=110, cores_per_bank=cpb, shape=shape, sender=sender)
    assert lay is not None, (G, K, N, cpb, shape, sender)
    assert len(lay.slots) <= 110 and len({s.core for s in lay.slots}) == len(lay.slots)
    assert all(0 <= s.core[0] < grid.x and 0 <= s.core[1] < grid.y for s in lay.slots)
    x = _dev(xt, mesh, ttnn.bfloat8_b)
    w = _dev(wt, mesh, ttnn.bfloat8_b)
    wb = bank.upload(wt, lay, mesh)
    ckc = _ckc(mesh)
    kw = dict(dtype=odt, compute_kernel_config=ckc, max_grid=ttnn.CoreCoord(11, 10), block_w=bw, depth=depth, rt=rt)
    today = ttnn.bringup.grouped_matmul(x, w, **kw)
    y = ttnn.bringup.grouped_matmul(x, wb, w_layout=lay, w_inflight=wi, x_ahead=xp, **kw)
    truth = _chips(x)[0] @ _chips(w)[0]
    got = _chips(y)
    rel = float((got[0] - truth).norm() / truth.norm())
    d_today = _maxdiff(y, today)
    # a repeat on fresh buffers (a prepared-program cache hit, the addresses patched)
    x2 = _dev(xt, mesh, ttnn.bfloat8_b)
    y2 = ttnn.bringup.grouped_matmul(x2, wb, w_layout=lay, w_inflight=wi, x_ahead=xp, **kw)
    d_rep = _maxdiff(y2, y)
    print(
        f"[gmm-bank] {G}x[{M},{K}]@[{K},{N}] {odt} cores {len(lay.slots)}: vs today {d_today} repeat {d_rep} rel {rel:.2e}"
    )
    assert d_today == 0.0, f"bank path differs from today's: max |d| {d_today}"
    assert d_rep == 0.0, f"bank path not repeatable: max |d| {d_rep}"
    assert all(torch.equal(c, got[0]) for c in got), "chips differ"
    assert rel < 2e-2, rel  # (bfp8 inputs and weights vs their exact product: the shipped path's own error)
    for t in (x, w, wb, today, y, x2, y2):
        ttnn.deallocate(t)


@mesh_parametrize
def test_grouped_matmul_bank_refusals(mesh_device, expect_error):
    if mesh_device is None:
        pytest.skip("device test")
    mesh = mesh_device
    lay = bank.plan(mesh, 8, 256, 1024, max_cores=110)
    x = _dev(torch.randn(1, 8, 32, 256), mesh, ttnn.bfloat8_b)
    w = _dev(torch.randn(1, 8, 256, 1024), mesh, ttnn.bfloat8_b)
    with expect_error(AssertionError, "w_layout needs the bank tensor"):
        ttnn.bringup.grouped_matmul(x, w, w_layout=lay, compute_kernel_config=_ckc(mesh))
    assert bank.plan(mesh, 8, 256, 1024, max_cores=95) is None  # (96 cores over the cap)
    assert bank.plan(mesh, 8, 256, 1024, max_cores=110, cores_per_bank=11) is None  # (odd: no 2-row band)
    assert bank.plan(mesh, 3, 256, 96, max_cores=110) is None  # (9 units over 8 banks: no padding, OV-3)


class _FakeDev:
    """The three device queries bank.plan makes (a Galaxy chip: 8 banks, 12 x 10 grid, anchors in columns 0 / 7)."""

    def dram_grid_size(self):
        return SimpleNamespace(x=8, y=1)

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=12, y=10)

    def get_optimal_dram_bank_to_logical_worker_assignment(self, noc):
        return [SimpleNamespace(x=x, y=y) for x, y in [(0, 0), (0, 3), (0, 5), (0, 8), (7, 1), (7, 3), (7, 6), (7, 9)]]


@pytest.mark.parametrize("G, cpb", [(8, 12), (8, 10), (4, 12)])
def test_bank_pack_cpu(G, cpb):
    """Every tile once, equal banks, contiguous K-major regions: any BW block is tile0 + kb * BW * pn."""
    K, N = 512, 1024
    lay = bank.plan(_FakeDev(), G, K, N, max_cores=110, cores_per_bank=cpb)
    assert lay is not None and len(lay.slots) == 8 * cpb
    w = torch.arange(G * K * N, dtype=torch.float64).reshape(1, G, K, N)
    regs = bank.regions(w, lay)
    assert len({r.shape[0] for r in regs}) == 1 and regs[0].shape[0] == lay.tiles_per_bank
    tiles = w.reshape(G, K // 32, 32, N // 32, 32).permute(0, 1, 3, 2, 4)
    seen = set()
    for s in lay.slots:
        for k in range(lay.KT):
            for j in range(s.pn):
                assert torch.equal(regs[s.bank][s.tile0 + k * s.pn + j], tiles[s.g, k, s.n0 + j])
                seen.add((s.g, k, s.n0 + j))
    assert len(seen) == G * lay.KT * lay.NT
    for c, (x0, y0, x1, y1) in enumerate(lay.rect):
        inside = [s for s in lay.slots if s.cluster == c]
        assert all(x0 <= s.core[0] <= x1 and y0 <= s.core[1] <= y1 for s in inside)
        assert sum(s.sender for s in inside) == 1 and len(inside) == (x1 - x0 + 1) * (y1 - y0 + 1)
