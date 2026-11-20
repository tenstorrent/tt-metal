# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""fabric_reduce_scatter's opt-in exact_fp32_add (SFPU adds on an fp32 DEST, every operand unpacked to DEST) and
dim=3 (column blocks), added for deepseek_v41_flash_d_p P.5 (CCL_IMPL=fabric: the model's exact TP partial sums).

Inputs: fp32 partials with a wide spread of exponents (randn x 2^U[-12, 12]) and full 23-bit mantissas, so a TF32 read
(10 bits, K-30) or a 16-bit DEST shows at once. Checks per chip:
  - a group of 2 (cluster_axis 0 / 1): bit-identical to torch's fp32 a + b, and to the model's stock exact sum
    (ttnn.all_gather + ttnn.add, SFPU), for row blocks (dim 2) and column blocks (dim 3);
  - the whole mesh (a snake line of 4, three adds in the op's order): within 3 fp32 roundings of the float64 sum
    (|got - sum| <= 3 * 2^-24 * sum |x_i|, elementwise);
  - controls: exact_fp32_add off on the same fp32 inputs is NOT bit-exact (the FPU path), and the bound above rejects
    it; refusals (a bf16 input with exact_fp32_add, a column count that does not split into G tile blocks)."""

import pytest
import torch

import ttnn
from ttnn.bringup._testing.mesh import MESH_PARAMS

ROWS, COLS = 1024, 512  # per chip partial [1, 1, ROWS, COLS]


def _partials(rows, cols, n, w, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, n, w, generator=g, dtype=torch.float64)
    e = torch.randint(-12, 13, (rows, cols, n, w), generator=g).double()
    return (x * torch.pow(2.0, e)).float()  # fp32, normal numbers only


def _upload(mesh_device, parts):
    rows, cols = parts.shape[0], parts.shape[1]
    return ttnn.from_torch(
        parts,
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )


def _per_chip(t):
    return [ttnn.to_torch(d).float().reshape(d.shape[-2], d.shape[-1]) for d in ttnn.get_device_tensors(t)]


def _block(full, pos, g, dim):
    n, w = full.shape
    if dim == 2:
        return full[pos * n // g : (pos + 1) * n // g]
    return full[:, pos * w // g : (pos + 1) * w // g]


def _stock_exact(t, axis, g):
    """The model's stock exact sum (tt/ccl.py _gather_add): all_gather on dim 1 + ttnn.add (SFPU fp32)."""
    gat = ttnn.all_gather(t, dim=1, cluster_axis=axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    r, w = t.shape[-2], t.shape[-1]
    acc = ttnn.slice(gat, [0, 0, 0, 0], [1, 1, r, w])
    for c in range(1, g):
        acc = ttnn.add(acc, ttnn.slice(gat, [0, c, 0, 0], [1, c + 1, r, w]), dtype=ttnn.float32)
    return acc


@MESH_PARAMS
@pytest.mark.parametrize("dim", [2, 3], ids=["rows", "cols"])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_exact_pair(mesh_device, device_params, cluster_axis, dim):
    rows, cols = tuple(mesh_device.shape)
    g = (rows, cols)[cluster_axis]
    if g != 2:
        pytest.skip("a group of 2 on this axis is the bit-exact case")
    parts = _partials(rows, cols, ROWS, COLS, seed=11 + cluster_axis + 2 * dim)
    x = _upload(mesh_device, parts)
    x = ttnn.reshape(x, (1, 1, ROWS, COLS))
    out = ttnn.bringup.fabric_reduce_scatter(x, cluster_axis=cluster_axis, num_links=2, dim=dim, exact_fp32_add=True)
    want_shape = [1, 1, ROWS // g, COLS] if dim == 2 else [1, 1, ROWS, COLS // g]
    assert list(out.shape) == want_shape, (list(out.shape), want_shape)
    stock = _per_chip(_stock_exact(x, cluster_axis, g))
    got = _per_chip(out)
    off = _per_chip(ttnn.bringup.fabric_reduce_scatter(x, cluster_axis=cluster_axis, num_links=2, dim=dim))
    n_off_exact = 0
    for d in range(rows * cols):
        r, c = divmod(d, cols)
        pos = (r, c)[cluster_axis]
        a, b = (parts[0, c], parts[1, c]) if cluster_axis == 0 else (parts[r, 0], parts[r, 1])
        want32 = _block(a + b, pos, g, dim)  # torch fp32 add: one IEEE rounding
        want64 = _block(a.double() + b.double(), pos, g, dim)
        n_bad = int((got[d] != want32).sum())
        n_stock = int((got[d] != _block(stock[d], pos, g, dim)).sum())
        rel = float((got[d].double() - want64).norm() / want64.norm())
        rel_off = float((off[d].double() - want64).norm() / want64.norm())
        print(
            f"axis {cluster_axis} dim {dim} dev {d}: exact {n_bad} != fp32 sum, {n_stock} != stock gather_add, "
            f"rel vs fp64 {rel:.2e}; FPU path rel {rel_off:.2e}"
        )
        assert n_bad == 0, f"dev {d}: {n_bad} values differ from torch's fp32 sum"
        assert n_stock == 0, f"dev {d}: {n_stock} values differ from all_gather + ttnn.add"
        assert rel < 1e-7, rel
        n_off_exact += int(torch.equal(off[d], want32))
    assert n_off_exact == 0, "control: the FPU add path should not be bit-exact on fp32 partials"


@MESH_PARAMS
@pytest.mark.parametrize("dim", [2, 3], ids=["rows", "cols"])
def test_exact_full_mesh(mesh_device, device_params, dim):
    """cluster_axis=None: one snake line over every chip (G = 4 on 2x2), chip d owns block d (row-major)."""
    rows, cols = tuple(mesh_device.shape)
    n = rows * cols
    if rows < 2 or cols < 2:
        pytest.skip("needs a 2D mesh")
    parts = _partials(rows, cols, ROWS, COLS, seed=5 + dim)
    x = ttnn.reshape(_upload(mesh_device, parts), (1, 1, ROWS, COLS))
    want64 = parts.double().sum((0, 1))
    bound = 3 * 2.0**-24 * parts.double().abs().sum((0, 1))
    for exact in (True, False):
        out = ttnn.bringup.fabric_reduce_scatter(x, cluster_axis=None, num_links=2, dim=dim, exact_fp32_add=exact)
        worst = 0.0
        for d, t in enumerate(_per_chip(out)):
            err = (t.double() - _block(want64, d, n, dim)).abs()
            worst = max(worst, float((err / _block(bound, d, n, dim)).max()))
        print(f"full mesh dim {dim} exact {exact}: worst |err| / (3 ulp-bound) {worst:.3f}")
        if exact:
            assert worst <= 1.0, worst
        else:
            assert worst > 1.0, "control: the FPU path should miss the 3-rounding bound"


@MESH_PARAMS
def test_refusals(mesh_device, device_params, expect_error):
    rows, cols = tuple(mesh_device.shape)
    xb = ttnn.reshape(
        ttnn.from_torch(
            torch.zeros(rows, cols, 64, 64),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
        ),
        (1, 1, 64, 64),
    )
    with expect_error(AssertionError, "exact_fp32_add needs FLOAT32"):
        ttnn.bringup.fabric_reduce_scatter(xb, cluster_axis=1, num_links=2, exact_fp32_add=True)
    xf = ttnn.reshape(_upload(mesh_device, torch.zeros(rows, cols, 64, 32)), (1, 1, 64, 32))
    with expect_error(AssertionError, "columns 32 must split"):
        ttnn.bringup.fabric_reduce_scatter(xf, cluster_axis=1, num_links=2, dim=3, exact_fp32_add=True)
