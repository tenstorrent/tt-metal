# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.unified_routed_expert_moe's DeepSeek-V4.1 options (deepseek_v41_flash_d_p P.2), against the torch
semantics in tests/reference.py (routed_h, fp8_act_qdq):

    routing_weights=R      h_i = act(gate_i, up_i) * R[i, le]      (expert le, dispatched row i; fp32 SFPU)
    fp8_act_quant=True     h_i = e4m3(h_i / s) * s per 32 hidden values, s = 2^ceil(log2(max(amax, 1e-4) / 448))
    weights_transposed     gate / up / down given as stored by nn.Linear, [N, K]

Every test runs on the box's 2x2 FABRIC_2D mesh; each chip runs its own experts on its own buffer.

* test_quant_and_weight_identity_down: down = the identity, so the op's output IS the quantized h; gate / up are
  selection weights (one nonzero per output column, a power of two), so every device GEMM is exact (random weights
  leave the FPU's in-tile rounding, DS41-16, in h: 2.6 % of the fp8 values then differ from a float64 reference).
  The check: every output block on its own e4m3 x ue8m0 grid, and out == fp8_act_qdq(reference h) bit for bit except
  at most 4 values one e4m3 step off at a near tie (the device sigmoid's last ulp). Measured: 0 differences over
  1528 rows x 1024 (2x2, seed 7). A negative control runs the same checker on corrupted copies (round toward zero,
  scale / 2, no quant, another expert's weight column) and requires each to fail; by hand (P.2) a kernel scale
  floor instead of ceil and a reader weight column off by one both failed.
* test_weight_only_identity_down: routing_weights without the quant: out == bf16(h * R[:, le]) to bf16 noise, every
  row's scale matches R of its own expert (another expert's column fails).
* test_transposed_bit_identical: [N, K] weights with weights_transposed give the same bits as the [K, N] weights
  without it (bf16 weights; and bfp8 [N, K] of e2m1 x 2^e values, exact in bfp8 along K, vs their bf16 [K, N]).
* test_ds41_shapes: DeepSeek-V4.1's expert shapes (emb 5120, hidden 2304) with all three options vs the float
  reference (rel, PCC, worst row).
* test_refusals: the host refuses the option combinations the kernels do not implement.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_TESTS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bringup_urffn_tests_reference_unit", _TESTS / "reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

TILE = 32
MESH = (2, 2)
N_DEV = MESH[0] * MESH[1]
DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "l1_small_size": 24576,
    "require_exact_physical_num_devices": True,
}
PARAMS = dict(argnames="mesh_device, device_params", argvalues=[(MESH, DEVICE_PARAMS)], indirect=True)


def _cfg(fp32=True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
        dst_full_sync_en=False,
    )


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=0)  # device d (row-major) gets block d


def _put(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh),
    )


def _routing(g, epc, counts_local, rows):
    """Per device: counts / regions [E] over global ids (chip d holds d*epc .. +epc-1), tile-aligned regions in
    local-slot order, and the dispatched buffer's row count."""
    E = epc * N_DEV
    counts, regions = [], []
    for d in range(N_DEV):
        cnt = torch.zeros(E, dtype=torch.int64)
        reg = torch.zeros(E, dtype=torch.int64)
        off = 0
        for le in range(epc):
            n = int(counts_local[(d + le) % len(counts_local)])
            cnt[d * epc + le] = n
            reg[d * epc + le] = off
            off += (n + TILE - 1) // TILE * TILE
        assert off <= rows, (off, rows)
        counts.append(cnt)
        regions.append(reg)
    return counts, regions


class Case:
    """Random inputs on the mesh: x [rows, H] bf16 ROW_MAJOR per chip, routing weights [rows, 32] fp32 per chip,
    per local expert W1 / W3 [I, H] and W2 [H, I] (stored, [N, K]) in `wdtype`; host copies of what the device holds."""

    def __init__(
        self,
        mesh,
        *,
        H,
        I,
        epc,
        counts_local,
        rows,
        cap,
        seed,
        wdtype=ttnn.bfloat8_b,
        x_scale=1.0,
        identity_down=False,
        w_kind="normal",
    ):
        g = torch.Generator().manual_seed(seed)
        self.mesh, self.H, self.I, self.epc, self.rows, self.cap = mesh, H, I, epc, rows, cap
        self.counts, self.regions = _routing(g, epc, counts_local, rows)
        self.x = (torch.randn(N_DEV * rows, H, generator=g) * x_scale).to(torch.bfloat16)
        # routing weights: positive, full fp32 mantissa, different per (row, column)
        self.r = torch.rand(N_DEV * rows, 32, generator=g) * 0.9 + 0.1
        compose = ttnn.ConcatMeshToTensor(mesh, dim=0)
        self.w = {"w1": [], "w3": [], "w2": []}  # device tensors, stored [N, K]
        self.wt = {"w1": [], "w3": [], "w2": []}  # device tensors, [K, N] (the op's default orientation)
        self.host = {"w1": [], "w3": [], "w2": []}  # [n_dev, N, K] as the device holds them
        for le in range(epc):
            for nm, (n, k) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
                if nm == "w2" and identity_down:
                    assert H == I
                    w = torch.eye(H).repeat(N_DEV, 1)
                elif w_kind == "select":
                    # one nonzero per output column (a permutation of the input columns, times a power of two): every
                    # in-tile dot has one nonzero product, so the device GEMM is exact (no in-tile rounding, DS41-16)
                    w = torch.zeros(N_DEV * n, k)
                    scale = 1.0 if nm == "w1" else 0.5
                    for dd in range(N_DEV):
                        perm = torch.randperm(k, generator=g)[:n] if n <= k else torch.randint(0, k, (n,), generator=g)
                        w[dd * n + torch.arange(n), perm] = scale
                elif w_kind == "e2m1":
                    # DeepSeek fp4 experts: e2m1 values times a power of two shared by 32 consecutive values along K
                    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
                    code = torch.randint(0, 8, (N_DEV * n, k), generator=g)
                    sign = torch.randint(0, 2, (N_DEV * n, k), generator=g) * 2 - 1
                    e = torch.randint(-9, -5, (N_DEV * n, k // 32), generator=g).float()
                    w = vals[code] * sign * torch.exp2(e).repeat_interleave(32, dim=1)
                else:
                    w = torch.randn(N_DEV * n, k, generator=g) * k**-0.5
                t = ttnn.from_torch(w, dtype=wdtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=_mapper(mesh))
                h = ttnn.to_torch(t, mesh_composer=compose).float().reshape(N_DEV, n, k)
                self.host[nm].append(h)
                self.w[nm].append(ttnn.to_device(t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                # the [K, N] copy in bf16: exactly the values the [N, K] tensor holds (a bfp8 W^T would re-round)
                wt = h.transpose(1, 2).reshape(N_DEV * k, n)
                self.wt[nm].append(_put(mesh, wt, ttnn.bfloat16, ttnn.TILE_LAYOUT))
        E = epc * N_DEV
        self.tt_x = _put(mesh, self.x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_r = _put(mesh, self.r, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_reg = _put(mesh, torch.stack(self.regions).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_cnt = _put(mesh, torch.stack(self.counts).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        gidx = torch.cat([d * epc + torch.arange(epc) for d in range(N_DEV)]).to(torch.int32)
        self.tt_gidx = _put(mesh, gidx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        assert list(self.tt_reg.shape) == [1, E] and list(self.tt_gidx.shape) == [epc]

    def call(
        self, *, rw=True, quant=True, transposed=True, cfg=None, activation="ClampedSiluGlu", high=True, r_tensor=None
    ):
        ws = self.w if transposed else self.wt
        kw = {}
        if rw:
            kw["routing_weights"] = self.tt_r if r_tensor is None else r_tensor
        return ttnn.bringup.unified_routed_expert_moe(
            self.tt_x,
            self.tt_reg,
            self.tt_cnt,
            self.tt_gidx,
            ws["w1"],
            ws["w3"],
            ws["w2"],
            max_dispatched_tokens_per_expert=self.cap,
            compute_kernel_config=cfg or _cfg(),
            activation=getattr(ttnn.bringup.RoutedExpertActivation, activation),
            high_precision=high,
            fp8_act_quant=quant,
            weights_transposed=transposed,
            **kw,
        )

    def regions_of(self, d):
        """[(local slot, first row, count)] of device d."""
        out = []
        for le in range(self.epc):
            e = d * self.epc + le
            out.append((le, int(self.regions[d][e]), int(self.counts[d][e])))
        return out

    def got(self, out):
        return [ttnn.to_torch(t).float().reshape(self.rows, self.H) for t in ttnn.get_device_tensors(out)]

    def want_h(self, d, le, r0, n, rw=True):
        """Reference h (fp32, before the quant) of device d's expert le rows [r0, r0 + n)."""
        rows = slice(d * self.rows + r0, d * self.rows + r0 + n)
        r = self.r[rows, le] if rw else None
        return ref.routed_h(self.x[rows], self.host["w1"][le][d], self.host["w3"][le][d], r)


def _e4m3_step(y):
    """The e4m3 grid spacing at |y| (|y| <= 448): 2^(floor(log2|y|) - 3) for normals, 2^-9 below 2^-6."""
    a = y.abs().clamp_min(2.0**-6)
    return torch.exp2(torch.floor(torch.log2(a)) - 3)


def quant_check(got, h_ref, max_ties=4, tie_steps=1e-3):
    """got [n, W] (the device's quantized h), h_ref [n, W] fp32 (reference before the quant, computed on the same
    exact GEMM values). Returns the failures (empty = pass): every block on its e4m3 x ue8m0 grid, and got ==
    fp8_act_qdq(h_ref) except at most `max_ties` values that are one e4m3 step off at a reference value within
    `tie_steps` steps of a rounding midpoint (the device sigmoid's last-ulp differences)."""
    fails = []
    if not bool(torch.isfinite(got).all()):
        return ["non-finite output"]
    want = ref.fp8_act_qdq(h_ref)
    if not torch.equal(ref.fp8_act_qdq(got), got):
        off = int((ref.fp8_act_qdq(got) != got).sum())
        fails.append(f"{off} values off their block's e4m3 x ue8m0 grid")
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
            fails.append(
                f"{int((mid > tie_steps).sum())} differences not at a near tie (worst {float(mid.max()):.2e} step)"
            )
        logger.info(f"quant: {nd} near-tie differences, worst midpoint distance {float(mid.max()):.2e} step")
    return fails


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(**PARAMS)
@pytest.mark.parametrize("transposed, rw", [(True, True), (False, True), (True, False)], ids=["nk-rw", "kn-rw", "nk"])
def test_quant_and_weight_identity_down(mesh_device, device_params, transposed, rw):
    c = Case(
        mesh_device,
        H=1024,
        I=1024,
        epc=4,
        counts_local=[37, 0, 130, 64, 257],
        rows=640,
        cap=288,
        seed=7,
        x_scale=4.0,
        identity_down=True,
        w_kind="select",
    )
    got = c.got(c.call(rw=rw, quant=True, transposed=transposed))
    total = 0
    for d in range(N_DEV):
        for le, r0, n in c.regions_of(d):
            if not n:
                continue
            total += n
            h = c.want_h(d, le, r0, n, rw=rw)
            g = got[d][r0 : r0 + n]
            fails = quant_check(g, h)
            assert not fails, f"dev {d} expert {le}: {fails}"
            # negative controls: the checker must reject corrupted outputs
            if d == 0 and le == 2 and rw:
                s = ref.fp8_act_scale(h).expand(-1, -1, 32).reshape(h.shape)
                y = (h / s).clamp(-448, 448)
                rtz = torch.sign(y) * torch.floor(y.abs() / _e4m3_step(y)) * _e4m3_step(y) * s
                half = (h / (s / 2)).clamp(-448, 448).to(torch.float8_e4m3fn).float() * (s / 2)
                rows = slice(d * c.rows + r0, d * c.rows + r0 + n)
                other = (le + 1) % c.epc
                wrong_col = ref.routed_h(c.x[rows], c.host["w1"][le][d], c.host["w3"][le][d], c.r[rows, other])
                for name, bad in (
                    ("round toward zero", rtz),
                    ("scale / 2", half),
                    ("no quant", h.to(torch.bfloat16).float()),
                    ("another expert's weight column", ref.fp8_act_qdq(wrong_col)),
                ):
                    assert quant_check(bad, h), f"the checker passed a corrupted output ({name})"
    logger.info(f"{total} routed rows checked")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(**PARAMS)
def test_weight_only_identity_down(mesh_device, device_params):
    c = Case(
        mesh_device,
        H=1024,
        I=1024,
        epc=4,
        counts_local=[37, 96, 130, 1],
        rows=448,
        cap=160,
        seed=11,
        identity_down=True,
    )
    got = c.got(c.call(rw=True, quant=False, transposed=True))
    for d in range(N_DEV):
        for le, r0, n in c.regions_of(d):
            if not n:
                continue
            h = c.want_h(d, le, r0, n)
            g = got[d][r0 : r0 + n]
            rel = float((g - h).norm() / h.norm())
            assert rel <= 4e-3, f"dev {d} expert {le}: rel {rel:.5f} vs bf16(h * r)"
            h0 = c.want_h(d, le, r0, n, rw=False)
            # each row's scale is its own expert's weight: ratio <got, h0> / <h0, h0> vs R[i, le]
            ratio = (g * h0).sum(-1) / (h0 * h0).sum(-1)
            rr = c.r[d * c.rows + r0 : d * c.rows + r0 + n]
            err_own = (ratio - rr[:, le]).abs().max()
            err_other = (ratio - rr[:, (le + 1) % c.epc]).abs().max()
            assert float(err_own) <= 2e-3, f"dev {d} expert {le}: row scale off its routing weight by {float(err_own)}"
            assert float(err_other) > 1e-2 or n == 1, "the routing weights of two columns are indistinguishable"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(**PARAMS)
@pytest.mark.parametrize("w_kind", ["bf16", "e2m1_bfp8"])
def test_transposed_bit_identical(mesh_device, device_params, w_kind):
    if w_kind == "bf16":
        c = Case(
            mesh_device, H=512, I=320, epc=2, counts_local=[70, 33], rows=192, cap=96, seed=3, wdtype=ttnn.bfloat16
        )
    else:  # bfp8 [N, K] holds e2m1 x 2^e (shared along K) exactly; the [K, N] side is the same values in bf16
        c = Case(
            mesh_device,
            H=512,
            I=320,
            epc=2,
            counts_local=[70, 33],
            rows=192,
            cap=96,
            seed=3,
            wdtype=ttnn.bfloat8_b,
            w_kind="e2m1",
        )
    for rw, quant in ((False, False), (True, True)):
        a = c.got(c.call(rw=rw, quant=quant, transposed=True, high=True))
        b = c.got(c.call(rw=rw, quant=quant, transposed=False, high=True))
        for d in range(N_DEV):
            for le, r0, n in c.regions_of(d):
                assert torch.equal(a[d][r0 : r0 + n], b[d][r0 : r0 + n]), f"{w_kind} rw={rw} q={quant} dev {d} le {le}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(**PARAMS)
def test_ds41_shapes(mesh_device, device_params):
    """DeepSeek-V4.1 expert shapes (emb 5120, hidden 2304), all options, float reference with the quant."""
    c = Case(
        mesh_device,
        H=5120,
        I=2304,
        epc=3,
        counts_local=[150, 0, 64, 333, 1],
        rows=768,
        cap=352,
        seed=5,
        w_kind="e2m1",
        x_scale=1.0,
    )
    got = c.got(c.call(rw=True, quant=True, transposed=True))
    for d in range(N_DEV):
        for le, r0, n in c.regions_of(d):
            if not n:
                continue
            h = c.want_h(d, le, r0, n)
            y = (ref.fp8_act_qdq(h).double() @ c.host["w2"][le][d].double().T).float()
            g = got[d][r0 : r0 + n]
            rel = float((g - y).norm() / y.norm())
            row = float(((g - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-30)).max())
            a, b = (g - g.mean()).double().flatten(), (y - y.mean()).double().flatten()
            pcc = float(a @ b / (a.norm() * b.norm()))
            logger.info(f"dev {d} expert {le} ({n} rows): rel {rel:.5f} worst row {row:.5f} pcc {pcc:.7f}")
            assert pcc >= 0.9999 and rel <= 0.01 and row <= 0.05, (d, le, rel, row, pcc)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(**PARAMS)
def test_refusals(mesh_device, device_params):
    c = Case(mesh_device, H=512, I=320, epc=2, counts_local=[40, 9], rows=128, cap=64, seed=1)
    bad_r = _put(mesh_device, c.r.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    short_r = _put(mesh_device, c.r[: N_DEV * (c.rows - 32)].contiguous(), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    cases = {
        "routing_weights without high_precision": dict(rw=True, quant=False, high=False),
        "fp8_act_quant without high_precision": dict(rw=False, quant=True, high=False),
        "routing_weights with a bf16 dest": dict(rw=True, quant=False, cfg=_cfg(fp32=False)),
        "fp8_act_quant with a bf16 dest": dict(rw=False, quant=True, cfg=_cfg(fp32=False)),
        "routing_weights with Silu": dict(rw=True, quant=False, activation="Silu"),
        "fp8_act_quant with GeluTanh": dict(rw=False, quant=True, activation="GeluTanh"),
        "bf16 routing_weights": dict(rw=True, quant=False, r_tensor=bad_r),
        "routing_weights with fewer rows than the buffer": dict(rw=True, quant=False, r_tensor=short_r),
        "routing_weights narrower than experts_per_chip": dict(
            rw=True,
            quant=False,
            r_tensor=_put(mesh_device, c.r[:, :1].contiguous(), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        ),
    }
    for name, kw in cases.items():
        assert _refused(lambda: c.call(**kw)), f"not refused: {name}"
    # [K, N] tensors passed as [N, K] (weights_transposed) do not fit x
    c.w, c.wt = c.wt, c.w
    assert _refused(lambda: c.call(rw=False, quant=False, transposed=True)), "not refused: [K, N] weights as [N, K]"


def _refused(fn) -> bool:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 (the op's TT_FATAL surfaces as RuntimeError)
        logger.info(f"refused: {str(e).splitlines()[0][:160]}")
        return True
    return False
