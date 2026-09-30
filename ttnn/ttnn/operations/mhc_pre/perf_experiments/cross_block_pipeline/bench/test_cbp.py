# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cross-block pipeline bake-off for mhc_pre (perf tournament, isolated copy of the op).

Variants (CBP_VARIANTS, comma-separated; see VARIANTS):
  base      the unmodified schedule (the op as committed)
  pipe      proj(b+1) before tail(b); writer P(b+1) right after S(b)
  pipe_rcf  + root combine(b) before proj(b+1)
  pipe_sJ   bf16 X / fp32 W: J streamed windows of proj(b+1) before tail(b), the rest after
  pipe_d3   pipe + cb_x_resident depth 3 (where it fits L1)

test_correct: every variant vs the golden reference + tolerances (eval/golden_tests/mhc_pre/helpers.py) over
CBP_SHAPES x CBP_DTYPES, two alternating seeds per cell (stale L1 must not mask a race). Prints worst rel-RMS.
test_perf (needs TT_METAL_DEVICE_PROFILER=1): per shape x dtype, variants interleaved CBP_REPEAT rounds, prints
DEVICE KERNEL DURATION medians + samples.
"""

import os

import pytest
import importlib

# bench-only torch (not imported by `import ttnn`: perf_experiments/ has no __init__.py); loaded via importlib so the
# ttnn-tree no-global-torch-import rule holds
torch = importlib.import_module("torch")
import ttnn

import ttnn.operations.mhc_pre.perf_experiments.cross_block_pipeline.mhc_pre_program_descriptor as pd
from ttnn.operations.mhc_pre.perf_experiments.cross_block_pipeline import mhc_pre
from eval.golden_tests.mhc_pre import helpers as H
import ttnn.operations.mhc_pre.perf_experiments.cross_block_pipeline.grad.mhc_pre_program_descriptor as grad_pd
from ttnn.operations.mhc_pre.perf_experiments.cross_block_pipeline.grad import mhc_pre as grad_mhc_pre

VARIANTS = {
    "grad": {},  # the graduated op (grad/: the diff of graduation.patch), no knobs
    "base": dict(PIPELINE=False),
    "pipe": dict(PIPELINE=True),
    "pipe_rcf": dict(PIPELINE=True, PIPE_ROOT_COMBINE_FIRST=True),
    "pipe_s1": dict(PIPELINE=True, PIPE_SPLIT_CHUNKS=1),
    "pipe_s2": dict(PIPELINE=True, PIPE_SPLIT_CHUNKS=2),
    "pipe_s3": dict(PIPELINE=True, PIPE_SPLIT_CHUNKS=3),
    "pipe_d3": dict(PIPELINE=True, PIPE_X_DEPTH=3),
    "pipe_rcf_d3": dict(PIPELINE=True, PIPE_ROOT_COMBINE_FIRST=True, PIPE_X_DEPTH=3),
    "sahead": dict(PIPELINE=True, PIPE_S_AHEAD=True),
    "sahead_rcf": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_ROOT_COMBINE_FIRST=True),
    "sahead_d3": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_X_DEPTH=3),
    "nohs": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True),
    "nohs_rcf": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_ROOT_COMBINE_FIRST=True),
    "nohs_rcoef": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_ROOT_COEF_FIRST=True),
    "nohs_rcf_rcoef": dict(
        PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_ROOT_COMBINE_FIRST=True, PIPE_ROOT_COEF_FIRST=True
    ),
    "sahead_rcoef": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_ROOT_COEF_FIRST=True),
    "nohs_rcoef_d3": dict(
        PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_ROOT_COEF_FIRST=True, PIPE_X_DEPTH=3
    ),
    "sahead_rcoef_d3": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_ROOT_COEF_FIRST=True, PIPE_X_DEPTH=3),
    "nohs_rcoef_tail": dict(
        PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_ROOT_COEF_FIRST=True, PIPE_TAIL_ONLY=True
    ),
    "nohs_rcoef_tail_d3": dict(
        PIPELINE=True,
        PIPE_S_AHEAD=True,
        PIPE_S_NOHS=True,
        PIPE_ROOT_COEF_FIRST=True,
        PIPE_TAIL_ONLY=True,
        PIPE_X_DEPTH=3,
    ),
    "sahead_rcoef_tail_d3": dict(
        PIPELINE=True, PIPE_S_AHEAD=True, PIPE_ROOT_COEF_FIRST=True, PIPE_TAIL_ONLY=True, PIPE_X_DEPTH=3
    ),
    "nohs_tail": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_TAIL_ONLY=True),
    "nohs_rtail_tail": dict(
        PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_TAIL_ONLY=True, PIPE_ROOT_TAIL_FIRST=True
    ),
    "nohs_rtail": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_ROOT_TAIL_FIRST=True),
    "nohs_none": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_NONE=True),
    "sahead_none": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_NONE=True),
    "nohs_d3": dict(PIPELINE=True, PIPE_S_AHEAD=True, PIPE_S_NOHS=True, PIPE_X_DEPTH=3),
}
DT = {
    "bx_fw": (ttnn.bfloat16, ttnn.float32),
    "fx_fw": (ttnn.float32, ttnn.float32),
    "fx_bw": (ttnn.float32, ttnn.bfloat16),
    "bx_bw": (ttnn.bfloat16, ttnn.bfloat16),
}
_KEY = "DEVICE KERNEL DURATION [ns]"


def _shapes(default):
    spec = os.environ.get("CBP_SHAPES", default)
    out = []
    for e in spec.split(","):
        e = e.strip()
        if not e:
            continue
        parts = [int(v) for v in e.split("x")]
        if len(parts) == 2:  # TxC -> (1, 1, T, 4C)
            out.append((1, 1, parts[0], 4 * parts[1]))
        else:  # AxBxTxC -> (A, B, T, 4C)
            out.append(tuple(parts[:-1]) + (4 * parts[-1],))
    return out


def _variants(default):
    return [v.strip() for v in os.environ.get("CBP_VARIANTS", default).split(",") if v.strip()]


def _dtypes(default):
    return [d.strip() for d in os.environ.get("CBP_DTYPES", default).split(",") if d.strip()]


def _extra():
    """CBP_EXTRA="NAME=pyexpr;..." descriptor knob overrides applied to every variant (interaction probes)."""
    out = {}
    for item in os.environ.get("CBP_EXTRA", "").split(";"):
        if item.strip():
            k, _, v = item.partition("=")
            out[k.strip()] = eval(v, {"ttnn": ttnn})
    return out


def _run(device, monkeypatch, variant, x, w, b, scale, dt):
    xd, wd = DT[dt]
    with monkeypatch.context() as m:
        for k, v in {**_extra(), **VARIANTS[variant]}.items():
            m.setattr(pd, k, v)
        tx = H.create_ttnn_input_tensor(x, device, dtype=xd, layout=ttnn.TILE_LAYOUT)
        tw = H.create_ttnn_input_tensor(w, device, dtype=wd, layout=ttnn.TILE_LAYOUT)
        tb = H.create_ttnn_input_tensor(b, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        op = grad_mhc_pre if variant == "grad" else mhc_pre
        out = op(tx, tw, tb, scale=scale, compute_kernel_config=H.make_compute_config())
        ttnn.synchronize_device(device)
        return out


def _relrms(got, ref):
    g = ttnn.to_torch(got).to(torch.float64).reshape(ref.shape)
    r = ref.to(torch.float64)
    return ((g - r).pow(2).mean().sqrt() / r.std().clamp_min(1e-30)).item()


def _plan_str(device, x_shape, dt):
    xd, wd = DT[dt]

    class _T:  # make_plan only needs padded_shape / shape / dtype / buffer_page_size
        def __init__(self, shape, dtype):
            self.padded_shape = list(shape[:-2]) + [((shape[-2] + 31) // 32) * 32, shape[-1]]
            self.shape = shape
            self.dtype = dtype

        def buffer_page_size(self):
            return 4096 if self.dtype == ttnn.float32 else 2048

    p = pd.make_plan(device, _T(x_shape, xd), _T((x_shape[-1], 24), wd), 4)
    return f"G={p.group_cores}({p.group_w}x{p.group_h}) bt={p.block_token_tiles} depth={p.x_block_depth} blocks={max((c + p.block_token_tiles - 1) // p.block_token_tiles for c in p.core_token_tiles)}"


def test_correct(device, monkeypatch):
    shapes = _shapes("1280x4096,640x1792,1000x1792,100x1792,1x2x256x8192")
    variants = _variants("base,pipe,pipe_rcf,pipe_s2")
    if variants[0] != "base":
        variants = ["base"] + [v for v in variants if v != "base"]
    dts = _dtypes("bx_fw,fx_fw,fx_bw,bx_bw")
    seeds = [int(s) for s in os.environ.get("CBP_SEEDS", "0,1").split(",")]
    fails = []
    for shape in shapes:
        n = 4
        w_shape = (shape[-1], n * (n + 2))
        for dt in dts:
            xd, wd = DT[dt]
            with monkeypatch.context() as m:
                for k, v in VARIANTS["pipe"].items():
                    m.setattr(pd, k, v)
                plan = _plan_str(device, shape, dt)
            base_ok, base_out = {}, {}
            for variant in variants:
                worst = [0.0, 0.0, 0.0]
                ok = True
                same = True
                for seed in seeds:
                    x, w, b, scale = H.make_inputs(shape, w_shape, dtype=xd, weight_dtype=wd, seed=seed)
                    y_ref, post_ref, comb_ref = H.pytorch_mhc_pre(x, w, b, scale=scale)
                    y, post, comb = _run(device, monkeypatch, variant, x, w, b, scale, dt)
                    lead = list(shape[:-1])
                    C = shape[-1] // n
                    try:
                        H.check_output(
                            y,
                            y_ref.to(H._TORCH_DTYPE[xd]),
                            shape=lead + [C],
                            dtype=xd,
                            expected_layout=ttnn.TILE_LAYOUT,
                            tolerance=H.TOLERANCES[("y", xd)],
                        )
                        H.check_output(
                            post,
                            post_ref,
                            shape=lead + [n],
                            dtype=ttnn.float32,
                            expected_layout=ttnn.TILE_LAYOUT,
                            tolerance=H.TOLERANCES[("coeff", xd)],
                        )
                        H.check_output(
                            comb,
                            comb_ref,
                            shape=lead + [n * n],
                            dtype=ttnn.float32,
                            expected_layout=ttnn.TILE_LAYOUT,
                            tolerance=H.TOLERANCES[("coeff", xd)],
                        )
                        H.check_doubly_stochastic(comb, comb_ref, n=n)
                    except Exception as e:  # noqa: BLE001
                        ok = False
                        # a golden miss the unmodified op also has on this (non-golden) seed is not the variant's
                        if variant == "base" or base_ok.get(seed, True):
                            fails.append((shape, dt, variant, seed, str(e)[:300]))
                    outs = [ttnn.to_torch(t) for t in (y, post, comb)]
                    if variant == "base":
                        base_ok[seed] = ok
                        base_out[seed] = outs
                    else:
                        same = same and all(torch.equal(a, b) for a, b in zip(outs, base_out[seed]))
                    for i, (got, ref) in enumerate(((y, y_ref), (post, post_ref), (comb, comb_ref))):
                        worst[i] = max(worst[i], _relrms(got, ref))
                print(
                    "CORRECT",
                    "x".join(map(str, shape)),
                    dt,
                    variant,
                    "PASS" if ok else ("FAIL" if all(base_ok.values()) or variant == "base" else "FAIL(=base)"),
                    "bitwise=base" if variant != "base" and same else "",
                    plan,
                    "relrms y/post/comb " + " ".join(f"{v:.2e}" for v in worst),
                )
    for f in fails:
        print("FAIL", f)
    # base-only golden misses (non-golden seeds) are reported, not gated: the gate is "no worse than base"
    assert not [f for f in fails if f[2] != "base"]


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                out.append(float(e.duration))
    return out


@pytest.mark.skipif(os.environ.get("TT_METAL_DEVICE_PROFILER") != "1", reason="needs TT_METAL_DEVICE_PROFILER=1")
def test_perf(device, monkeypatch):
    shapes = _shapes("1280x4096,640x7168,640x1792")
    variants = _variants("base,pipe,pipe_rcf,pipe_s2")
    dts = _dtypes("bx_fw")
    repeat = int(os.environ.get("CBP_REPEAT", 3))
    rows = []
    for shape in shapes:
        n = 4
        for dt in dts:
            xd, wd = DT[dt]
            torch.manual_seed(0)
            x = torch.randn(shape, dtype=torch.float32)
            w = torch.randn((shape[-1], 24), dtype=torch.float32) / shape[-1] ** 0.5
            b = torch.randn((1, 24), dtype=torch.float32)
            ns = {v: [] for v in variants}
            _read_ns(device)
            for v in variants:  # warm compile, not measured
                _run(device, monkeypatch, v, x, w, b, (1.0, 1.0, 1.0), dt)
                _read_ns(device)
            for _ in range(repeat):
                for v in variants:
                    y, post, comb = _run(device, monkeypatch, v, x, w, b, (1.0, 1.0, 1.0), dt)
                    ns[v] += _read_ns(device)
                    if "ABLATE" not in os.environ.get("MHC_PRE_KERNEL_DEFINES", ""):
                        for t in (y, post, comb):
                            assert torch.isfinite(ttnn.to_torch(t)).all()
            for v in variants:
                with monkeypatch.context() as m:
                    for k, val in {**_extra(), **VARIANTS[v]}.items():
                        m.setattr(pd, k, val)
                    plan = _plan_str(device, shape, dt)
                rows.append(("x".join(map(str, shape)), dt, v, ns[v], plan))
    for r in rows:
        s = sorted(r[3])
        med = s[len(s) // 2] / 1000 if s else float("nan")
        print("PERF", *r[:3], f"median {med:.1f}us |", " ".join(f"{v / 1000:.1f}" for v in r[3]), "|", r[4])
