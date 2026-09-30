# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""prelude_off_critical_path bench: A/B of the op copy in this dir, baseline vs PRELUDE_* variants.

Variants (defines of the copy's kernels, via mhc_pre_program_descriptor.PRELUDE_DEFINES):
  base  = unmodified copy of the op
  a     = PRELUDE_A        reader: non-blocking poll of the W share's trid between X page issues
  b1    = PRELUDE_B_EARLY  writer: bias DRAM reads issued at kernel start (fill where it was)
  b     = PRELUDE_B        writer: bias reads at kernel start + fill after block 0's partial send
  ab    = PRELUDE_A;PRELUDE_B

test_correctness: golden reference + tolerances (eval/golden_tests/mhc_pre/helpers.py), alternating seeds per
  variant (stale L1 must not mask a race). test_perf: interleaved in-process DEVICE KERNEL DURATION.
Env: PRELUDE_SHAPES="640x7168,..." (T x C, X = (1,1,T,4C)), PRELUDE_DTYPES="xbf16_wf32,..." ,
     PRELUDE_VARIANTS="base,a,b,ab", PRELUDE_REPEAT=5, PRELUDE_SEEDS=2, PRELUDE_GOLDEN_INPUTS=1 (use INPUTS).
Run: TT_METAL_DEVICE_PROFILER=1 ... scripts/run_safe_pytest.sh --device 1 --run-all -s <this file>
"""

import os

import pytest
import importlib

# bench-only torch (not imported by `import ttnn`: perf_experiments/ has no __init__.py); loaded via importlib so the
# ttnn-tree no-global-torch-import rule holds
torch = importlib.import_module("torch")
import ttnn

from eval.golden_tests.mhc_pre import helpers as H
from eval.golden_tests.mhc_pre.feature_spec import INPUTS

from ttnn.operations.mhc_pre.perf_experiments.prelude_off_critical_path import mhc_pre_program_descriptor as pd
from ttnn.operations.mhc_pre.perf_experiments.prelude_off_critical_path import mhc_pre

VARIANTS = {
    "base": "",
    "a": "PRELUDE_A",
    "b1": "PRELUDE_B_EARLY",
    "b": "PRELUDE_B",
    "ab": "PRELUDE_A;PRELUDE_B",
    "ab1": "PRELUDE_A;PRELUDE_B_EARLY",
    "f": "PRELUDE_FASTFILL",
    "af": "PRELUDE_A;PRELUDE_FASTFILL",
    "rf": "PRELUDE_B_READER;PRELUDE_FASTFILL",
    "arf": "PRELUDE_A;PRELUDE_B_READER;PRELUDE_FASTFILL",
    "sf": "PRELUDE_B_AFTER_SHARE;PRELUDE_FASTFILL",
    "asf": "PRELUDE_A;PRELUDE_B_AFTER_SHARE;PRELUDE_FASTFILL",
    "b1f": "PRELUDE_B_EARLY;PRELUDE_FASTFILL",
    "bf": "PRELUDE_B;PRELUDE_FASTFILL",
    "ab1f": "PRELUDE_A;PRELUDE_B_EARLY;PRELUDE_FASTFILL",
    "abf": "PRELUDE_A;PRELUDE_B;PRELUDE_FASTFILL",
}
_DT = {"bf16": ttnn.bfloat16, "f32": ttnn.float32}
_KEY = "DEVICE KERNEL DURATION [ns]"


def _dtypes():
    out = []
    for e in os.environ.get("PRELUDE_DTYPES", "xbf16_wf32").split(","):
        xs, ws = e.split("_")
        out.append((e, _DT[xs[1:]], _DT[ws[1:]]))
    return out


def _shapes():
    sel = os.environ.get("PRELUDE_SHAPES", "640x7168")
    if sel == "golden":
        return [tuple(c[0]) for c in INPUTS]
    return [(1, 1, int(t), 4 * int(c)) for t, c in (s.split("x") for s in sel.split(","))]


# variant name suffix "+fo": PRELUDE_SHARE_FLIPPED_ONLY (descriptor knob)
def _apply(monkeypatch, v):
    base, _, knob = v.partition("+")
    monkeypatch.setattr(pd, "PRELUDE_DEFINES", VARIANTS[base])
    monkeypatch.setattr(pd, "PRELUDE_SHARE_FLIPPED_ONLY", knob == "fo")
    # "+skN": the first N rows of the reader-NoC region (the most congested ones) read no W share;
    # "+skNmM": ... and the last M flipped rows too
    skip = None
    if knob.startswith("sk"):
        a, _, b = knob[2:].partition("m")
        na, nb = int(a), int(b or 0)
        skip = lambda f, na=na, nb=nb: set(range(f, f + na)) | set(range(f - nb, f))
    monkeypatch.setattr(pd, "PRELUDE_SHARE_SKIP_ROWS", skip)
    monkeypatch.setattr(pd, "W_SHARE_BEFORE_X", knob == "bx")  # "+bx": the parked R5 knob
    # "+grad": the real op's kernels with graduation.patch applied (kernels_grad/, no bench defines)
    kdir = pd.Path(pd.__file__).parent / {"grad": "kernels_grad", "grad2": "kernels_grad_ab1f"}.get(knob, "kernels")
    monkeypatch.setattr(pd, "KERNEL_DIR", kdir)


def _variants():
    return os.environ.get("PRELUDE_VARIANTS", "base,a,b,ab").split(",")


def _name(shape):
    return f"{'x'.join(map(str, shape[:-1]))}x{shape[-1] // 4}"


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                out.append(float(e.duration))
    return out


def _rel_rms(got, ref):
    g, r = got.to(torch.float64).flatten(), ref.to(torch.float64).flatten()
    return ((g - r).square().mean().sqrt() / (r.std() + 1e-30)).item()


def _run(device, shape, xdt, wdt, seed):
    w_shape = (shape[-1], 24)
    x, w, bias, scale = H.make_inputs(shape, w_shape, dtype=xdt, weight_dtype=wdt, seed=seed)
    tx = H.create_ttnn_input_tensor(x, device, dtype=xdt, layout=ttnn.TILE_LAYOUT)
    tw = H.create_ttnn_input_tensor(w, device, dtype=wdt, layout=ttnn.TILE_LAYOUT)
    tb = H.create_ttnn_input_tensor(bias, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    outs = mhc_pre(tx, tw, tb, scale=scale, compute_kernel_config=H.make_compute_config())
    return (x, w, bias, scale), outs


def test_correctness(device, monkeypatch):
    seeds = int(os.environ.get("PRELUDE_SEEDS", 2))
    fails, worst = [], {}
    refs = {}
    for shape in _shapes():
        for dname, xdt, wdt in _dtypes():
            for s in range(seeds):
                base_out, base_ok = None, None
                vs = ["base"] + [v for v in _variants() if v != "base"]
                for v in vs:  # alternate variants and seeds: each run sees the previous one's stale L1
                    _apply(monkeypatch, v)
                    (x, w, bias, scale), (y, post, comb) = _run(device, shape, xdt, wdt, seed=s)
                    key = (shape, dname, s)
                    if key not in refs:
                        refs[key] = H.pytorch_mhc_pre(x, w, bias, scale=scale)
                    y_ref, post_ref, comb_ref = refs[key]
                    n, C = 4, shape[-1] // 4
                    lead = list(shape[:-1])
                    try:
                        H.check_output(
                            y,
                            y_ref.to(H._TORCH_DTYPE[xdt]),
                            shape=lead + [C],
                            dtype=xdt,
                            expected_layout=ttnn.TILE_LAYOUT,
                            tolerance=H.TOLERANCES[("y", xdt)],
                        )
                        for t, r in ((post, post_ref), (comb, comb_ref)):
                            H.check_output(
                                t,
                                r,
                                shape=lead + [r.shape[-1]],
                                dtype=ttnn.float32,
                                expected_layout=ttnn.TILE_LAYOUT,
                                tolerance=H.TOLERANCES[("coeff", xdt)],
                            )
                        H.check_doubly_stochastic(comb, comb_ref, n=n)
                        ok = "PASS"
                    except Exception as e:  # noqa: BLE001
                        ok = f"FAIL {str(e)[:200]}"
                    got = [ttnn.to_torch(t) for t in (y, post, comb)]
                    if v == "base":
                        base_out, base_ok = got, ok
                        if ok != "PASS":
                            ok = "PRE-EXISTING " + ok  # the unmodified op misses the tolerance here too
                    else:
                        same = all(torch.equal(a, b) for a, b in zip(got, base_out))
                        if not same:
                            ok = "NOT-BIT-IDENTICAL-TO-BASE " + ok
                            fails.append((_name(shape), dname, v, s, ok))
                        elif ok != "PASS":
                            ok = "PRE-EXISTING(bit-identical to base) " + ok
                            if base_ok == "PASS":
                                fails.append((_name(shape), dname, v, s, ok))
                    rr = [_rel_rms(t, r) for t, r in zip(got, (y_ref, post_ref, comb_ref))]
                    wk = (v, dname)
                    worst[wk] = [max(a, b) for a, b in zip(worst.get(wk, [0, 0, 0]), rr)]
                    print(
                        f"CORR {_name(shape)} {dname} {v} seed{s} {ok} relrms y/post/comb "
                        + " ".join(f"{q:.2e}" for q in rr)
                    )
    for (v, d), rr in sorted(worst.items()):
        print(f"WORST {v} {d} relrms y/post/comb " + " ".join(f"{q:.2e}" for q in rr))
    assert not fails, fails


@pytest.mark.skipif(os.environ.get("TT_METAL_DEVICE_PROFILER") != "1", reason="needs TT_METAL_DEVICE_PROFILER=1")
def test_perf(device, monkeypatch):
    rep = int(os.environ.get("PRELUDE_REPEAT", 5))
    rows = []
    for shape in _shapes():
        for dname, xdt, wdt in _dtypes():
            torch.manual_seed(0)
            nc = shape[-1]
            x = torch.randn(shape, dtype=torch.float32)
            w = torch.randn((nc, 24), dtype=torch.float32) / nc**0.5
            b = torch.randn((1, 24), dtype=torch.float32)
            tx = ttnn.from_torch(x, dtype=xdt, layout=ttnn.TILE_LAYOUT, device=device)
            tw = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
            tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            ns = {v: [] for v in _variants()}
            for v in _variants():  # compile every variant first (the first call's kernel time is still valid)
                _apply(monkeypatch, v)
                mhc_pre(tx, tw, tb, scale=(1.0, 1.0, 1.0))
                ttnn.synchronize_device(device)
            _read_ns(device)
            for _ in range(rep):  # interleaved A/B rounds
                for v in _variants():
                    _apply(monkeypatch, v)
                    outs = mhc_pre(tx, tw, tb, scale=(1.0, 1.0, 1.0))
                    ttnn.synchronize_device(device)
                    ns[v] += _read_ns(device)
            for t in outs:
                assert torch.isfinite(ttnn.to_torch(t)).all()
            for v in _variants():
                rows.append((_name(shape), dname, v, ns[v]))
    for r in rows:
        s = sorted(r[3])
        med = s[len(s) // 2] / 1000 if s else float("nan")
        print("PERF", *r[:3], f"median {med:.1f}us |", " ".join(f"{q / 1000:.1f}" for q in r[3]))
