# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.flat_routed_expert's options from deepseek_v41_flash_d_p P.3 that remain after FIX.9 removed the
DeepSeek-only h options (routing_weights, fp8_act_quant) with P.3's test_ds41_h_options.py:

    weights_nk             the weights given as nn.Linear stores them, [N, K] (FlatRoutedExpert(weights_nk=True))
    hidden=H               x rows wider than H (the dispatched row's extra columns are skipped)
    check_exact            FlatRoutedExpert refuses weights its block float would round

Every test runs on the box's 2x2 FABRIC_2D mesh; each chip runs its own local experts on its own buffer.

* test_nk_bit_identical: weights exact in bfp8 along both dims (one power-of-two scale per matrix): the [N, K] layout
  with weights_nk gives the same bits as the [K, N] layout without it.
* test_ds41_shapes: DeepSeek-V4.1's per-chip expert shapes (emb 5120, intermediate 2304, 24 local experts), e2m1 x
  2^e weights with a scale per 32 along K (exact in bfp8 [N, K]; asserted at load by check_exact), wide x rows
  (hidden), HiFi4 / fp32 DEST / bf16 x h y, vs a float64 reference (rel, PCC, worst row).
* test_check_exact: block_float_exact rejects a 16-value group bfp8 would round and accepts e2m1 x 2^e.
"""

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.bringup._testing.mesh import MESH_PARAMS
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert, block_float_exact

TILE = 32
HIFI4 = 4
CLAMPED_SILU = "clamped_silu"
LIM = 10.0  # ClampedSiluGlu's limit (se3_compute.cpp SE_ACT 3)


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=0)  # device d (row-major) gets block d


def _put(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh),
    )  # fmt: skip


def _expert64(x, w1, w3, w2):
    """x [n, H]; w1, w3 [I, H], w2 [H, I] (stored) -> y [n, H] fp32: w2 (silu(min(w1 x, 10)) * clamp(w3 x, +-10)),
    in float64."""
    g, u = x.double() @ w1.double().T, x.double() @ w3.double().T
    h = torch.nn.functional.silu(g.clamp(max=LIM)) * u.clamp(-LIM, LIM)
    return (h @ w2.double().T).float()


class Case:
    """Per device: epc local experts (global ids d * epc + le), x [rows, W] bf16 ROW_MAJOR (W >= H: the first H values
    are the token's), weights stored [N, K] (W1 / W3 [I, H], W2 [H, I])."""

    def __init__(self, mesh, *, H, I, epc, counts_local, rows, cap, seed, w_kind, x_scale=1.0, extra=0, nk=True):
        g = torch.Generator().manual_seed(seed)
        self.mesh, self.H, self.I, self.epc, self.rows, self.cap = mesh, H, I, epc, rows, cap
        self.n_dev = mesh.get_num_devices()
        E = epc * self.n_dev
        self.counts, self.regions = [], []
        for d in range(self.n_dev):
            cnt, reg, off = torch.zeros(E, dtype=torch.int64), torch.zeros(E, dtype=torch.int64), 0
            for le in range(epc):
                n = int(counts_local[(d + le) % len(counts_local)])
                cnt[d * epc + le], reg[d * epc + le] = n, off
                off += (n + TILE - 1) // TILE * TILE
            assert off <= rows, (off, rows)
            self.counts.append(cnt)
            self.regions.append(reg)
        self.x = (torch.randn(self.n_dev * rows, H + extra, generator=g) * x_scale).to(torch.bfloat16)
        self.w = [[None] * epc for _ in range(self.n_dev)]  # [dev][le] = (W1 [I, H], W3 [I, H], W2 [H, I]) fp32
        vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        for d in range(self.n_dev):
            for le in range(epc):
                mats = []
                for n, k in ((I, H), (I, H), (H, I)):
                    # e2m1 values times a power of two per 32 along K (DeepSeek's fp4 experts); e2m1_flat: one power
                    # of two for the matrix (exact in bfp8 along either dim)
                    code = torch.randint(0, 8, (n, k), generator=g)
                    sign = torch.randint(0, 2, (n, k), generator=g) * 2 - 1
                    if w_kind == "e2m1":
                        e = torch.randint(-9, -5, (n, k // 32), generator=g).float().repeat_interleave(32, dim=1)
                    else:
                        e = torch.full((n, k), -7.0)
                    mats.append(vals[code] * sign * torch.exp2(e))
                self.w[d][le] = tuple(mats)
        self.fe = FlatRoutedExpert(
            mesh,
            [[(a, b, c) if nk else (a.T.contiguous(), b.T.contiguous(), c.T.contiguous()) for a, b, c in dev]
             for dev in self.w],
            m=cap, H=H, I=I, gids=[[d * epc + le for le in range(epc)] for d in range(self.n_dev)], n_global=E,
            wdtype="bf8", act=CLAMPED_SILU, pin=1, x_bf16=True, h_bf16=True, weights_nk=nk, check_exact=True,
        )  # fmt: skip
        self.tt_x = _put(mesh, self.x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_reg = _put(mesh, torch.stack(self.regions).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_cnt = _put(mesh, torch.stack(self.counts).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def call(self):
        hidden = self.H if self.x.shape[-1] != self.H else 0
        return self.fe(
            self.tt_x, self.tt_cnt, self.tt_reg, y_bf16=True, down_fp32=True, math_fidelity=HIFI4, hidden=hidden
        )

    def got(self, out):
        return [ttnn.to_torch(t).float().reshape(self.rows, self.H) for t in ttnn.get_device_tensors(out)]

    def regions_of(self, d):
        """[(local slot, first row, count)] of device d."""
        return [(le, int(self.regions[d][d * self.epc + le]), int(self.counts[d][d * self.epc + le])) for le in
                range(self.epc)]  # fmt: skip


@pytest.mark.timeout(2400)
@MESH_PARAMS
def test_nk_bit_identical(mesh_device, device_params):
    kw = dict(H=2304, I=2304, epc=2, counts_local=[70, 33, 0], rows=192, cap=256, seed=3, w_kind="e2m1_flat")
    a = Case(mesh_device, nk=True, **kw)
    ya = a.got(a.call())
    b = Case(mesh_device, nk=False, **kw)
    yb = b.got(b.call())
    for d in range(a.n_dev):
        for le, r0, n in a.regions_of(d):
            assert torch.equal(ya[d][r0 : r0 + n], yb[d][r0 : r0 + n]), f"dev {d} le {le}: [N, K] != [K, N]"


@pytest.mark.timeout(3600)
@MESH_PARAMS
def test_ds41_shapes(mesh_device, device_params):
    """DeepSeek-V4.1 per-chip experts (emb 5120, intermediate 2304, 24 local), weights_nk + hidden, float64 reference."""
    c = Case(
        mesh_device, H=5120, I=2304, epc=24, counts_local=[150, 0, 64, 333, 1, 32, 96, 7], rows=4096, cap=2048,
        seed=5, w_kind="e2m1", x_scale=1.0, extra=192,
    )  # fmt: skip
    got = c.got(c.call())
    for d in range(c.n_dev):
        for le, r0, n in c.regions_of(d):
            if not n:
                continue
            rows = slice(d * c.rows + r0, d * c.rows + r0 + n)
            y = _expert64(c.x[rows, : c.H], *c.w[d][le])
            g = got[d][r0 : r0 + n]
            rel = float((g - y).norm() / y.norm())
            row = float(((g - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-30)).max())
            a_, b_ = (g - g.mean()).double().flatten(), (y - y.mean()).double().flatten()
            pcc = float(a_ @ b_ / (a_.norm() * b_.norm()))
            if le < 3 or rel > 0.01:
                logger.info(f"dev {d} expert {le} ({n} rows): rel {rel:.5f} worst row {row:.5f} pcc {pcc:.7f}")
            assert pcc >= 0.9999 and rel <= 0.01 and row <= 0.05, (d, le, rel, row, pcc)


def test_check_exact():
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    g = torch.Generator().manual_seed(1)
    w1 = vals[torch.randint(0, 8, (64, 64), generator=g)] * 2.0**-7
    assert block_float_exact(w1.reshape(-1, 32), "bf8")
    bad = w1.clone()
    bad[0, 0], bad[0, 1] = 6.0 * 2.0**-3, 0.5 * 2.0**-9  # one 16-value group spans 2^-1 .. 2^-10
    assert not block_float_exact(bad.reshape(-1, 32), "bf8")
