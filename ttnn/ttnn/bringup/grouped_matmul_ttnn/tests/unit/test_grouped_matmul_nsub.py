# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.grouped_matmul bank path, column sub-blocks (GMM_N_SUB; deepseek_v41_flash_d_p FIX.42): a core whose rt x
column tiles exceed DEST (8 tiles at fp32 DEST) runs its columns in sub-blocks; before FIX.42 the op refused it.

On the bring-up mesh (BATCH_SHARED, the resident runner's held mesh), G = 1 bank layouts (N split over the 8 banks, the
wq_b / wo_b layout): (a) the same layout at rt 1 (no sub-blocks) and at an rt that needs them: bit-identical (same
products, same K order per tile); (b) a layout whose cores hold more than 8 column tiles at rt 1 (sub-blocks): vs a
torch fp64 reference of the device's bfp8 inputs, and a repeat on fresh buffers (same bytes)."""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize
from ttnn.bringup.grouped_matmul_ttnn import bank

BATCH_SHARED = True  # (a mesh-only op test: the resident runner's held mesh)

# (M, K, N, cores a bank, block_w, rt): N / 32 / 8 column tiles a bank over the bank's cores
SAME = [
    (64, 256, 2560, 2, 4, 2),  # 10 columns a bank, 5 a core: rt 1 fits (5), rt 2 needs sub-blocks (10 > 8)
    (128, 512, 4096, 4, 4, 4),  # 4 a core: rt 4 -> 16 > 8, sub-blocks of 2
]
WIDE = [
    (64, 256, 3072, 2, 4),  # 12 columns a bank, 6 a core: fits; (the reference case for the next)
    (64, 256, 5120, 2, 4),  # 20 a bank, 10 a core at rt 1: sub-blocks 5 + 5
    (256, 1280, 32768, 10, 8),  # wq_b at 8192 (band 10): 128 a bank, 13 / 12 a core: sub-blocks 7 + 6 / 6 + 6
]


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


def _inputs(mesh, M, K, N, cpb):
    g = torch.Generator().manual_seed(M + K + N)
    xt = torch.randn(1, 1, M, K, generator=g)
    wt = torch.randn(1, 1, K, N, generator=g) / K**0.5
    lay = bank.plan(mesh, 1, K, N, max_cores=110, cores_per_bank=cpb, shape="band")
    assert lay is not None, (K, N, cpb)
    return _dev(xt, mesh, ttnn.bfloat8_b), wt, lay, bank.upload(wt, lay, mesh)


def _call(mesh, x, wb, lay, bw, rt, one_group=True):
    return ttnn.grouped_matmul(
        x, wb, w_layout=lay, dtype=ttnn.bfloat16, compute_kernel_config=_ckc(mesh), block_w=bw, depth=2, rt=rt,
        one_group=one_group,
    )  # fmt: skip


@mesh_parametrize
@pytest.mark.parametrize("case", SAME, ids=[f"M{c[0]}-K{c[1]}-N{c[2]}-c{c[3]}-rt{c[5]}" for c in SAME])
def test_nsub_same_bytes(mesh_device, case):
    """rt 1 (no sub-blocks) vs an rt that needs them, one layout: bit-identical."""
    if mesh_device is None:
        pytest.skip("device test")
    M, K, N, cpb, bw, rt = case
    x, _, lay, wb = _inputs(mesh_device, M, K, N, cpb)
    assert rt * lay.pn_max > 8, (rt, lay.pn_max)  # (the case really takes the sub-block path)
    for og in (True, False):
        a = _call(mesh_device, x, wb, lay, bw, 1, og)
        b = _call(mesh_device, x, wb, lay, bw, rt, og)
        d = _maxdiff(a, b)
        assert d == 0.0, f"one_group {og}: rt 1 vs rt {rt} differ by {d}"
        for t in (a, b):
            ttnn.deallocate(t)


@mesh_parametrize
@pytest.mark.parametrize("case", WIDE, ids=[f"M{c[0]}-K{c[1]}-N{c[2]}-c{c[3]}" for c in WIDE])
def test_nsub_wide_cores(mesh_device, case):
    """Cores over DEST at rt 1: vs fp64 of the device's inputs, and a repeat."""
    if mesh_device is None:
        pytest.skip("device test")
    M, K, N, cpb, bw = case
    x, wt, lay, wb = _inputs(mesh_device, M, K, N, cpb)
    w8 = _dev(wt, mesh_device, ttnn.bfloat8_b)  # (the same bfp8 weights, interleaved: the reference's operand)
    truth = _chips(x)[0] @ _chips(w8)[0]
    a = _call(mesh_device, x, wb, lay, bw, 1)
    b = _call(mesh_device, x, wb, lay, bw, 1)
    rep = _maxdiff(a, b)
    rel = max(float((c - truth).norm() / truth.norm()) for c in _chips(a))
    # one bf16 rounding of the output (2^-9 relative) dominates a correct fp32-accumulated product; a wrong column,
    # sub-block offset or K block gives rel ~1
    assert rel < 2**-8, f"pn_max {lay.pn_max}: rel vs fp64 {rel}"
    assert rep == 0.0, f"repeat differs by {rep}"
    for t in (a, b, w8):
        ttnn.deallocate(t)
