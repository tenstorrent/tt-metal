# Zone profile of the port sender startup, one non-traced call per variant, every chip / port.
#   VARS="base;sscopy;ssopen;ssearly;sswarm"  (each run with ",zones" appended)
import collections
import csv
import os
import shutil
from pathlib import Path

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
import ttnn  # noqa: E402

from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache  # noqa: E402
from eval.golden_tests.all_to_all_matmul.helpers import prepare_all_to_all_matmul  # noqa: E402
from eval.golden_tests.all_to_all_matmul.test_golden import _loose_cases  # noqa: E402
from ttnn.operations.all_to_all_matmul import all_to_all_matmul  # noqa: E402

import importlib.util, sys  # noqa: E402

_spec = importlib.util.spec_from_file_location("_ss_common", Path(__file__).parent / "_common.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

NAMES = [
    "psd_hdr_setup",
    "psd_hdr_pass2",
    "psd_sub_route",
    "psd_sub_store",
    "psd_open",
    "psd_fence",
    "psd_hdr_rest",
    "psd_wait_ready",
    "psd_send",
    "prd_all",
]
CSV = Path("generated/profiler/.logs/profile_log_device.csv")
OUT = Path(os.environ.get("OUTDIR", "/tmp/ss_zones"))


def analyze(f, label):
    rows = list(csv.reader(open(f)))[2:]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r[0], r[7])].append((int(r[1]), int(r[2]), r[3], r[10], r[11], int(r[5])))
    lines = [f"== {label}"]
    for dev in sorted({k[0] for k in by}):
        last = max((k[1] for k in by if k[0] == dev), key=int)
        ev = by[(dev, last)]
        t0 = min(t for *_, n, ty, t in ev if n.endswith("-KERNEL") and ty == "ZONE_START")
        t1 = max(t for *_, n, ty, t in ev if n.endswith("-KERNEL") and ty == "ZONE_END")
        # port sender kernel start (BRISC on port cores = cores having psd_* zones)
        lines.append(f"dev {dev}: kernel span {(t1 - t0) / 1350:.2f} us")
        starts, z = {}, collections.defaultdict(list)
        for x, y, risc, n, ty, t in ev:
            k = (x, y, risc, n)
            if ty == "ZONE_START":
                starts.setdefault(k, []).append(t)
            elif ty == "ZONE_END" and starts.get(k):
                s = starts[k].pop()
                z[n].append(((x, y, risc), (s - t0) / 1350, (t - t0) / 1350))
        for n in NAMES:
            L = z.get(n, [])
            if not L:
                continue
            ss = sorted(s for _, s, _ in L)
            es = sorted(e for _, _, e in L)
            ds = sorted(e - s for _, s, e in L)
            q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
            lines.append(
                f"  {n:15s} n={len(L):3d} start[min/med/max] {ss[0]:6.2f} {q(ss,.5):6.2f} {ss[-1]:6.2f}  "
                f"end {es[0]:6.2f} {q(es,.5):6.2f} {es[-1]:6.2f}  dur med {q(ds,.5):6.2f} max {ds[-1]:6.2f}"
            )
        acc = collections.defaultdict(list)
        for r in rows:
            if r[0] == dev and r[7] == last and r[10] in ("psd_acc_cb", "psd_acc_slot"):
                acc[r[10]].append(int(r[6]) / 1350)
        for n, L in sorted(acc.items()):
            lines.append(f"  {n:15s} per port us: " + " ".join(f"{v:6.2f}" for v in L))
    return "\n".join(lines)


def test_zones():
    # one variant per process: the device CSV is written when the mesh closes
    v = os.environ.get("VAR", "base")
    inputs, axes, extras = _loose_cases()[int(os.environ.get("CASE", "0"))].values
    cache = _FabricMeshCache()
    mesh = cache.get((1, 4), payload_bytes=extras.get("fabric_payload_bytes"))
    call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=all_to_all_matmul, **axes)
    OUT.mkdir(parents=True, exist_ok=True)
    toks = [t for t in v.split(",") if t and t != "base"]
    os.environ["A2AMM_ABLATE"] = ",".join(toks + ["zones"])
    for _ in range(3):
        out = call()
    check(out)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    os.environ["A2AMM_ABLATE"] = ""
    cache.close()
    dst = OUT / f"{v.replace(',', '+')}.csv"
    shutil.copy(CSV, dst)
    rep = analyze(dst, v + " (check OK)")
    print(rep, flush=True)
    (OUT / f"{v.replace(',', '+')}.txt").write_text(rep + "\n")


if __name__ == "__main__" or True:
    test_zones()
