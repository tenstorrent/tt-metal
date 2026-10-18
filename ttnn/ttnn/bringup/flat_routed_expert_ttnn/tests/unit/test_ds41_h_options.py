# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.flat_routed_expert's DeepSeek-V4.1 options (deepseek_v41_flash_d_p P.3), against the torch semantics
of unified_routed_expert_ffn/tests/reference.py (routed_h, fp8_act_qdq, the same math P.2 gave that op):

    routing_weights=R      h_i = act(gate_i, up_i) * R[i, le]      (expert le, dispatched row i; fp32 SFPU)
    fp8_act_quant=True     h_i = e4m3(h_i / s) * s per 32 values of h, s = 2^ceil(log2(max(amax, 1e-4) / 448))
    weights_nk             the weights given as nn.Linear stores them, [N, K] (FlatRoutedExpert(weights_nk=True))
    hidden=H               x rows wider than H (the dispatched row's extra columns are skipped)

Every test runs on the box's 2x2 FABRIC_2D mesh; each chip runs its own local experts on its own buffer. The plans
are the op's own: intermediate 2304 (72 tile columns) gives 16-column units, 3 a core, so every third h tile is split
over two cores (the tail core quantizes it whole after the lead core's half arrives).

* test_identity_down: emb = intermediate = 2304, down = the identity, so y IS the quantized h; gate / up are selection
  weights (one power of two per output column), so every device GEMM is exact (DS41-16, DS41-60). y ==
  fp8_act_qdq(reference h) bit for bit except near ties (the device sigmoid's last ulp); a negative control requires
  the checker to fail corrupted copies. Variants: weight + quant, quant only, weight only (bf16 h vs bf16(h r)).
* test_nk_bit_identical: weights exact in bfp8 along both dims (one power-of-two scale per matrix): the [N, K] layout
  with weights_nk gives the same bits as the [K, N] layout without it.
* test_ds41_shapes: DeepSeek-V4.1's per-chip expert shapes (emb 5120, intermediate 2304, 24 local experts), e2m1 x
  2^e weights with a scale per 32 along K (exact in bfp8 [N, K]; asserted at load by check_exact), wide x rows, all
  options at HiFi4 / fp32 DEST / bf16 x h y, vs the float reference with the quant (rel, PCC, worst row).
* test_refusals: the host refuses what the kernels do not implement.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.bringup._testing.mesh import MESH_PARAMS
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert, block_float_exact

_REF = Path(__file__).resolve().parents[3] / "unified_routed_expert_ffn" / "tests" / "reference.py"
_spec = importlib.util.spec_from_file_location("bringup_flat_ds41_reference", _REF)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

TILE = 32
HIFI4 = 4
CLAMPED_SILU = "clamped_silu"


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=0)  # device d (row-major) gets block d


def _put(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh),
    )  # fmt: skip


def _e4m3_step(y):
    """The e4m3 grid spacing at |y| (|y| <= 448): 2^(floor(log2|y|) - 3) for normals, 2^-9 below 2^-6."""
    a = y.abs().clamp_min(2.0**-6)
    return torch.exp2(torch.floor(torch.log2(a)) - 3)


def quant_check(got, h_ref, max_ties=8, tie_steps=1e-3):
    """got [n, W] (the device's quantized h), h_ref [n, W] fp32 (reference before the quant, on the same exact GEMM
    values). Failures (empty = pass): every block on its e4m3 x ue8m0 grid, and got == fp8_act_qdq(h_ref) except at
    most `max_ties` values one e4m3 step off at a reference value within `tie_steps` steps of a rounding midpoint."""
    fails = []
    if not bool(torch.isfinite(got).all()):
        return ["non-finite output"]
    want = ref.fp8_act_qdq(h_ref)
    if not torch.equal(ref.fp8_act_qdq(got), got):
        fails.append(f"{int((ref.fp8_act_qdq(got) != got).sum())} values off their block's e4m3 x ue8m0 grid")
    diff = got != want
    nd = int(diff.sum())
    if nd > max_ties:
        fails.append(f"{nd} of {diff.numel()} values differ from the reference (max {max_ties} near ties)")
    if nd:
        s = ref.fp8_act_scale(h_ref).expand(-1, -1, 32).reshape(got.shape).double()
        y = (h_ref.double() / s)[diff]
        step = _e4m3_step(y.float()).double()
        dg = (got.double()[diff] - want.double()[diff]).abs() / s[diff]
        if bool((dg > step * 1.0001).any()):
            fails.append(f"{int((dg > step * 1.0001).sum())} differences larger than one e4m3 step")
        lo = torch.floor(y.abs() / step) * step
        mid = ((y.abs() - lo) / step - 0.5).abs()
        if bool((mid > tie_steps).any()):
            fails.append(f"{int((mid > tie_steps).sum())} differences not at a near tie (worst {float(mid.max()):.2e})")
        logger.info(f"quant: {nd} near-tie differences, worst midpoint distance {float(mid.max()):.2e} step")
    return fails


class Case:
    """Per device: epc local experts (global ids d * epc + le), x [rows, W] bf16 ROW_MAJOR (W >= H: the first H values
    are the token's), routing weights [rows, 32] fp32 TILE, weights stored [N, K] (W1 / W3 [I, H], W2 [H, I])."""

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
        W = H + extra
        self.x = (torch.randn(self.n_dev * rows, W, generator=g) * x_scale).to(torch.bfloat16)
        self.r = torch.rand(self.n_dev * rows, 32, generator=g) * 0.9 + 0.1  # positive, full fp32 mantissa
        self.w = [[None] * epc for _ in range(self.n_dev)]  # [dev][le] = (W1 [I, H], W3 [I, H], W2 [H, I]) fp32
        for d in range(self.n_dev):
            for le in range(epc):
                mats = []
                for nm, (n, k) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
                    if w_kind == "select" and nm == "w2":
                        assert H == I
                        w = torch.eye(H)
                    elif w_kind == "select":
                        # one power of two per output row of the stored [N, K] matrix: every in-tile dot has one nonzero
                        # product, so the device GEMM is exact (no in-tile rounding, DS41-16)
                        w = torch.zeros(n, k)
                        w[torch.arange(n), torch.randperm(k, generator=g)[:n]] = 1.0 if nm == "w1" else 0.5
                    elif w_kind in ("e2m1", "e2m1_flat"):
                        # e2m1 values times a power of two per 32 along K (DeepSeek's fp4 experts); e2m1_flat: one
                        # power of two for the matrix (exact in bfp8 along either dim)
                        vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
                        code = torch.randint(0, 8, (n, k), generator=g)
                        sign = torch.randint(0, 2, (n, k), generator=g) * 2 - 1
                        if w_kind == "e2m1":
                            e = torch.randint(-9, -5, (n, k // 32), generator=g).float().repeat_interleave(32, dim=1)
                        else:
                            e = torch.full((n, k), -7.0)
                        w = vals[code] * sign * torch.exp2(e)
                    else:
                        raise ValueError(w_kind)
                    mats.append(w)
                self.w[d][le] = tuple(mats)
        self.nk = nk
        self.fe = FlatRoutedExpert(
            mesh,
            [[(a, b, c) if nk else (a.T.contiguous(), b.T.contiguous(), c.T.contiguous()) for a, b, c in dev]
             for dev in self.w],
            m=cap, H=H, I=I, gids=[[d * epc + le for le in range(epc)] for d in range(self.n_dev)], n_global=E,
            wdtype="bf8", act=CLAMPED_SILU, pin=1, x_bf16=True, h_bf16=True, weights_nk=nk, routed=True,
            check_exact=True,
        )  # fmt: skip
        self.tt_x = _put(mesh, self.x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_r = _put(mesh, self.r, ttnn.float32, ttnn.TILE_LAYOUT)
        self.tt_reg = _put(mesh, torch.stack(self.regions).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_cnt = _put(mesh, torch.stack(self.counts).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def call(self, *, rw=True, quant=True, r_tensor=None, **kw):
        args = dict(
            y_bf16=True, down_fp32=True, math_fidelity=HIFI4, fp8_act_quant=quant,
            routing_weights=(self.tt_r if r_tensor is None else r_tensor) if rw else None,
            hidden=self.H if self.x.shape[-1] != self.H else 0,
        )  # fmt: skip
        args.update(kw)
        return self.fe(self.tt_x, self.tt_cnt, self.tt_reg, **args)

    def got(self, out):
        return [ttnn.to_torch(t).float().reshape(self.rows, self.H) for t in ttnn.get_device_tensors(out)]

    def regions_of(self, d):
        """[(local slot, first row, count)] of device d."""
        return [(le, int(self.regions[d][d * self.epc + le]), int(self.counts[d][d * self.epc + le])) for le in
                range(self.epc)]  # fmt: skip

    def rows_of(self, d, r0, n):
        return slice(d * self.rows + r0, d * self.rows + r0 + n)

    def want_h(self, d, le, r0, n, rw=True, col=None):
        """Reference h (fp32, before the quant) of device d's expert le rows [r0, r0 + n)."""
        rows = self.rows_of(d, r0, n)
        r = self.r[rows, le if col is None else col] if rw else None
        w1, w3, _ = self.w[d][le]
        return ref.routed_h(self.x[rows, : self.H], w1, w3, r)


@pytest.mark.timeout(2400)
@MESH_PARAMS
@pytest.mark.parametrize(
    "rw, quant, counts_local, rows, cap",
    [
        (True, True, [37, 0, 130, 64, 257], 640, 288),
        (False, True, [37, 0, 130, 64, 257], 640, 288),
        (True, False, [37, 0, 130, 64, 257], 640, 288),
        # one dominant expert: the schedule pins it and interleaves the small ones (entries reordered and split)
        (True, True, [700, 3, 0, 9], 800, 768),
    ],
    ids=["rw-q", "q", "rw", "rw-q-pinned"],
)
def test_identity_down(mesh_device, device_params, rw, quant, counts_local, rows, cap):
    c = Case(
        mesh_device, H=2304, I=2304, epc=4, counts_local=counts_local, rows=rows, cap=cap, seed=7,
        w_kind="select", x_scale=4.0, extra=96,
    )  # fmt: skip
    assert c.fe.plan["nu"] % 2 == 1, f"the test wants split h tiles (odd units), plan nu {c.fe.plan['nu']}"
    got = c.got(c.call(rw=rw, quant=quant))
    total = 0
    for d in range(c.n_dev):
        for le, r0, n in c.regions_of(d):
            if not n:
                continue
            total += n
            h = c.want_h(d, le, r0, n, rw=rw)
            g = got[d][r0 : r0 + n]
            if not quant:  # bf16 h: bf16(h r) after the identity down (bf16 y)
                rel = float((g - h).norm() / h.norm())
                assert rel <= 4e-3, f"dev {d} expert {le}: rel {rel:.5f} vs h r"
                h0 = c.want_h(d, le, r0, n, rw=False)
                ratio = (g * h0).sum(-1) / (h0 * h0).sum(-1)
                rr = c.r[c.rows_of(d, r0, n)]
                assert float((ratio - rr[:, le]).abs().max()) <= 2e-3, f"dev {d} expert {le}: row scale off its weight"
                continue
            fails = quant_check(g, h)
            assert not fails, f"dev {d} expert {le}: {fails}"
            if d == 0 and le == 2 and rw:  # negative controls: the checker must reject corrupted outputs
                s = ref.fp8_act_scale(h).expand(-1, -1, 32).reshape(h.shape)
                y = (h / s).clamp(-448, 448)
                rtz = torch.sign(y) * torch.floor(y.abs() / _e4m3_step(y)) * _e4m3_step(y) * s
                half = (h / (s / 2)).clamp(-448, 448).to(torch.float8_e4m3fn).float() * (s / 2)
                other = ref.fp8_act_qdq(c.want_h(d, le, r0, n, col=(le + 1) % c.epc))
                # a split tile quantized per half (each core's own amax) instead of whole
                hb = h.reshape(n, -1, 16)
                per_half = ref.fp8_act_qdq(hb.reshape(n * hb.shape[1], 16), block=16).reshape(n, -1)
                for name, bad in (
                    ("round toward zero", rtz), ("scale / 2", half), ("no quant", h.to(torch.bfloat16).float()),
                    ("another expert's weight column", other), ("per-half scales", per_half),
                ):  # fmt: skip
                    assert quant_check(bad, h), f"the checker passed a corrupted output ({name})"
    logger.info(f"{total} routed rows checked")


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
    """DeepSeek-V4.1 per-chip experts (emb 5120, intermediate 2304, 24 local), all options, float reference."""
    c = Case(
        mesh_device, H=5120, I=2304, epc=24, counts_local=[150, 0, 64, 333, 1, 32, 96, 7], rows=4096, cap=2048,
        seed=5, w_kind="e2m1", x_scale=1.0, extra=192,
    )  # fmt: skip
    got = c.got(c.call())
    for d in range(c.n_dev):
        for le, r0, n in c.regions_of(d):
            if not n:
                continue
            h = c.want_h(d, le, r0, n)
            y = (ref.fp8_act_qdq(h).double() @ c.w[d][le][2].double().T).float()
            g = got[d][r0 : r0 + n]
            rel = float((g - y).norm() / y.norm())
            row = float(((g - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-30)).max())
            a_, b_ = (g - g.mean()).double().flatten(), (y - y.mean()).double().flatten()
            pcc = float(a_ @ b_ / (a_.norm() * b_.norm()))
            if le < 3 or rel > 0.01:
                logger.info(f"dev {d} expert {le} ({n} rows): rel {rel:.5f} worst row {row:.5f} pcc {pcc:.7f}")
            assert pcc >= 0.9999 and rel <= 0.01 and row <= 0.05, (d, le, rel, row, pcc)


@pytest.mark.timeout(1200)
@MESH_PARAMS
def test_refusals(mesh_device, device_params):
    c = Case(mesh_device, H=2304, I=2304, epc=2, counts_local=[40, 9], rows=128, cap=256, seed=1, w_kind="e2m1_flat")
    rm_r = _put(mesh_device, c.r, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    bf_r = _put(mesh_device, c.r.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    short_r = _put(mesh_device, c.r[: c.n_dev * (c.rows - 32)], ttnn.float32, ttnn.TILE_LAYOUT)
    for name, kw in {
        "routing_weights ROW_MAJOR": dict(r_tensor=rm_r),
        "routing_weights bf16": dict(r_tensor=bf_r),
        "routing_weights with fewer rows than x": dict(r_tensor=short_r),
        "routing_weights with row-major y": dict(y_row_major=True, y_bf16=False),
    }.items():
        assert _refused(lambda: c.call(**kw)), f"not refused: {name}"
    # weights not exact in bfp8: check_exact refuses them at load
    w1, w3, w2 = c.w[0][0]
    bad = w1.clone()
    bad[0, 0], bad[0, 1] = 6.0 * 2.0**-3, 0.5 * 2.0**-9  # one 16-value group spans 2^-1 .. 2^-10
    assert not block_float_exact(bad.reshape(-1, 32), "bf8")
    assert block_float_exact(w1.reshape(-1, 32), "bf8")


def _refused(fn) -> bool:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 (TT_FATAL surfaces as RuntimeError)
        logger.info(f"refused: {str(e).splitlines()[0][:160]}")
        return True
    return False
