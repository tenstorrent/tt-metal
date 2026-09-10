"""passb_op_count -- isolated A/B of pass B's TWO broadcast-mul traversals.

The op's pass B is  out = x * stat<Col> * gamma<Row>, spelled as two block
traversals with a materialized block-sized intermediate (cb_normalized).  Every
variant here keeps the SAME two muls at the SAME precision; they differ only in
the ORDER of the two broadcast axes and in the CB pack lifecycle.  The `ceil`
variant is an ABLATION (deliberately wrong) that deletes the gamma traversal
outright, to bound what any traversal-removing route could ever be worth.

PRECISION CONTRACT FROZEN: every case pins math_fidelity / fp32_dest_acc_en /
math_approx_mode and the dtypes from feature_spec, identically for all variants.

Cases come straight out of eval/golden_tests/rms_norm_ttnn/feature_spec.py's
`perf` group (19 cases; #4 = (1,1,8192,1024) INTERLEAVED gamma = the focus) so the
domain sweep cannot drift from the spec.

MEASURED (blackhole p150b, one fresh-cache profiled run per variant, RMS_TRIALS
readings inside each build, median; every variant under the case's OWN pinned
math_fidelity / fp32_dest_acc_en / dtypes -- nothing precision-related was touched).

The idea as briefed -- "cut pass B's MATH tile-op count" -- is a measured NULL, and
`nomul` is why: replacing the gamma broadcast-mul with a bare CopyTile over the same
tiles (traversal kept, MULTIPLY deleted) changes nothing, while deleting the whole
traversal is worth up to 1.24x.  Pass B's second pass costs its TRAVERSAL, not its
math.

  case (feature_spec perf group)      base ns    ceil     nomul    swap    swapx
  --------------------------------------------------------------------------------
  00 32x1024   INTER  combine=T          4359   1.049       -     1.066      -
  03 32x7168   INTER  combine=T          8437   1.070       -     1.067    1.065
  04 32x1024.. FOCUS  combine=F         83959   1.011       -     1.00     1.00 (bit-exact)
  05 8192x2304 INTER  combine=F        179965   1.007       -     0.983    0.993 (bit-exact)
  06 8192x5120 INTER  combine=F        406860   1.003       -     1.000    0.995 (bit-exact)
  07 8192x7168 INTER  combine=F        556207   0.994       -     1.000      -
  08 32x1024   WIDTH  combine=T          2606   1.096   0.986     1.085    1.095
  09 32x2304   WIDTH  combine=T          2930   1.126       -     1.125      -
  10 32x5120   WIDTH  combine=T          3439   1.102       -     1.099      -
  11 32x7168   WIDTH  combine=T          3788   1.113   1.002     1.113    1.129
  12 8192x1024 BLOCK  combine=T         20404   1.244   0.982     1.014    1.022
  13 32x5120   INTER  gbr  combine=T     9616   1.034       -     1.046    1.036
  14 8192x5120 INTER  gbr  combine=F   613130   1.012       -     1.000      -
  15 8192x7168 INTER  gbr  combine=F  1044141   1.000       -     0.994      -
  16 32x5120   WIDTH  gbr  combine=T     4184   1.091       -     1.055    1.068
  17 7168x1024 BLOCK  gbr  combine=T    28936   1.168   1.004     1.010      -
  18 128x4096  INTER  wRM  combine=T    11307   1.075       -     1.079    1.074

  ceil / nomul are ABLATIONS (wrong on purpose).  swap/swapx keep the op correct:
  pcc equals base's to 6 dp on all 19 cases, and swapx is BIT-EXACT with base on
  every combine=False plan (its gate falls back to the shipped order there).

  PASS_B_BLK (subblock_w) re-sweep, on base:
    focus  (WT_CHUNK=32): 1 -> 87197, 2 -> 83799, 4 -> 83875, 8 -> 83910 ns
                          i.e. FLAT from 2 upward; the shipped rule already sits on
                          the plateau, nothing to re-tune here.
    12 BLOCK (WT_CHUNK=4): 1 -> 25370, 2 -> 21467, 4 -> 20413 ns (1.000/1.182/1.243)
    17 BLOCK (WT_CHUNK=4): 1 -> 36025, 2 -> 30014, 4 -> 28947 ns (1.000/1.200/1.245)
                          still CLIMBING at the cap -- and the cap is WT_CHUNK=4, not
                          the 8 DEST lanes, because a chain block cannot span rows.

  ua / uan (block-granular pass-B pack lifecycle) are a measured NULL: 0.969-1.015
  across cases 4, 8, 11, 12 -- bit-exact, no direction.

Run:
  RMS_NAMES=4 RMS_VARIANTS=base,swap scripts/tt-probe.sh rms_norm_ttnn <<'PY'
  import sys; sys.path.insert(0, "<this dir>"); import bench; bench.main()
  PY
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics
from pathlib import Path

import ttnn

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import shard_config
from eval.golden_tests.rms_norm_ttnn import feature_spec as FS

HERE = Path(__file__).resolve().parent
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_TRIALS = int(os.environ.get("RMS_TRIALS", "3"))
FOCUS = 4  # index into the perf group


def perf_cases():
    return [c for c in FS.LOOSE_CASES if c.get("group") == "perf"]


def label(i, c):
    shape = c["inputs"][0]
    ml = str(c["memory_layout"]).split(".")[-1][:5]
    g = c.get("gamma_mode", "gamma")
    f32 = "T" if c.get("fp32_dest_acc_en") else "F"
    gl = "wRM" if c.get("gamma_layout") == ttnn.ROW_MAJOR_LAYOUT else "wTI"
    return f"{i:02d}_{shape[-2]}x{shape[-1]}_{ml}_{g[:12]}_f32{f32}_{gl}"


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def build(device, c):
    import torch  # function-local: ttnn/ttnn must not import torch at module scope

    shape = list(c["inputs"][0])
    W = shape[-1]
    ex = c["extras"]
    dtype = c["dtype"]
    lay = c["layout"]
    torch.manual_seed(0)
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ml = c["memory_layout"]
    if "shard_shape" in ex:
        mc = shard_config(ex["shard_shape"], ex["core_grid"], ml, layout=lay, dtype=dtype, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(tx, dtype=dtype, layout=lay, device=device, memory_config=mc)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ex.get("math_fidelity", ttnn.MathFidelity.HiFi2)
    cfg.fp32_dest_acc_en = bool(c.get("fp32_dest_acc_en"))
    cfg.math_approx_mode = False

    kwargs = {"epsilon": 1e-12, "compute_kernel_config": cfg, "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}
    mode = c.get("gamma_mode", "gamma")
    glay = c.get("gamma_layout", ttnn.TILE_LAYOUT)
    gdt = c.get("gamma_dtype", ttnn.bfloat16)

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=gdt, layout=glay, device=device)

    if "gamma" in mode:
        t, v = _vec(1)
        kwargs["weight"] = v
        ref["weight"] = t.float()
    if "bias" in mode:
        t, v = _vec(2)
        kwargs["bias"] = v
        ref["bias"] = t.float()
    if "residual" in mode:
        torch.manual_seed(3)
        tr = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        kwargs["residual_input_tensor"] = ttnn.from_torch(tr, dtype=dtype, layout=lay, device=device, memory_config=mc)
        ref["residual_input_tensor"] = tr.float()

    expected = torch_rms_norm_ttnn(
        ref["input_tensor"],
        epsilon=1e-12,
        weight=ref.get("weight"),
        bias=ref.get("bias"),
        residual_input_tensor=ref.get("residual_input_tensor"),
    )
    live = [x] + [v for v in kwargs.values() if isinstance(v, ttnn.Tensor)]
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected, live


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def measure(device, c, keep_out):
    run, expected, live = build(device, c)
    out = run()
    got = ttnn.to_torch(out)
    p = pcc(got, expected)
    del out
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = []
    for _ in range(N_TRIALS):
        o = run()
        ttnn.synchronize_device(device)
        v = _read_kernel_ns(device)
        ttnn.deallocate(o)
        if v is not None:
            samples.append(v)
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return (statistics.median(samples), min(samples), p, got if keep_out else None)


def main():
    import torch  # function-local

    cases = perf_cases()
    names = os.environ.get("RMS_NAMES", str(FOCUS))
    idxs = list(range(len(cases))) if names == "all" else [int(n) for n in names.split(",")]
    variants = [v for v in os.environ.get("RMS_VARIANTS", "base,swap").split(",") if v]
    blks = [b for b in os.environ.get("RMS_BLKS", "").split(",")]  # "" = the op's own choice

    saved_dir, saved_pc = PD.KERNEL_DIR, PD._PC_NONE
    combos = [(v, b) for b in blks for v in variants]
    res = {}
    device = ttnn.open_device(device_id=0)
    try:
        for i in idxs:
            c = cases[i]
            lbl = label(i, c)
            base_out = None
            for v, b in combos:
                PD.KERNEL_DIR = HERE / f"k_{v}"
                PD._PC_NONE = saved_pc._replace(subblock_w=int(b)) if b else saved_pc
                try:
                    med, mn, p, got = measure(device, c, keep_out=True)
                except Exception as e:
                    print(f"RESULT {lbl} {v}@{b or 'auto'} ERROR {type(e).__name__}: {e}", flush=True)
                    PD.KERNEL_DIR, PD._PC_NONE = saved_dir, saved_pc
                    continue
                PD.KERNEL_DIR, PD._PC_NONE = saved_dir, saved_pc
                if base_out is None:
                    base_out = got
                    eq = "base"
                else:
                    eq = "bitexact" if torch.equal(got, base_out) else "differs"
                del got
                res[(i, v, b)] = (med, mn, p)
                print(
                    f"RESULT {lbl:38s} {v:5s}@{b or 'auto':4s} ns={med:10.0f} min={mn:10.0f} pcc={p:.6f} {eq}",
                    flush=True,
                )
            del base_out
    finally:
        PD.KERNEL_DIR, PD._PC_NONE = saved_dir, saved_pc
        ttnn.close_device(device)

    v0, b0 = combos[0]
    print(f"RESULT ==== speedup vs {v0}@{b0 or 'auto'} (>1 = faster) ====")
    for i in idxs:
        if (i, v0, b0) not in res:
            continue
        ref = res[(i, v0, b0)][0]
        row = "".join(f"  {v}@{b or 'auto'}={ref / res[(i, v, b)][0]:6.3f}" for v, b in combos if (i, v, b) in res)
        print(f"RESULT {label(i, cases[i]):38s}{row}")
