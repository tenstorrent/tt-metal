"""Re-measure translated (blind) tilize cases against main's numbers (eval run 1049, BH).

Usage (from the clone root, python_env active):
  python3 ttnn/ttnn/operations/tilize/perf_experiments/sharded_vs_main/bench.py [--all] [-k SUBSTR] [--tag NAME]

Default set = the cases that were slower than main in run 1061's blind pass. --all = every
shared, measured case (regression check). Runs eval/eval_test_runner.sh (same device-perf
capture as the golden suite) and prints ratio = main_ns / new_ns (>1 is faster than main).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
TEST_FILE = "eval/golden_tests/tilize/test_translated.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-k", default=None, help="substring filter on test name")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--main", action="store_true", help="run production ttnn.tilize instead (baseline)")
    ap.add_argument("--repeat", type=int, default=1, help="runs; per-test median is reported")
    ap.add_argument("--vs", default="main_local", help="out/<tag> to use as main_ns where it has a number")
    args = ap.parse_args()

    base = json.loads((HERE / "baseline_main_1049.json").read_text())
    local = HERE / "out" / args.vs / "summary.json"
    if local.exists() and not args.main:  # same-box main numbers override run 1049's
        for t, ns in json.loads(local.read_text()).items():
            if ns:
                base[t]["main_ns"] = ns
    names = [t for t, v in base.items() if (args.all or v["slow"]) and (not args.k or args.k in t)]
    names.sort()
    out_dir = HERE / "out" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "select.txt").write_text("\n".join(names))
    env = dict(os.environ, BENCH_SELECT_FILE=str(out_dir / "select.txt"), PYTHONPATH=str(ROOT))
    if args.main:
        env["EVAL_TARGET_MODULE"] = "eval.adapters.main_tilize"
    samples = {}
    for i in range(args.repeat):
        run_dir = out_dir / f"run{i}"
        cmd = [
            "eval/eval_test_runner.sh",
            "--no-precompile",
            TEST_FILE,
            str(run_dir),
            "-p",
            "eval.adapters.select_plugin",
        ]
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for r in json.loads((run_dir / "test_results.json").read_text()):
            ok = r.get("status") == "passed" or (args.main and r.get("device_kernel_ns"))
            if ok and r.get("device_kernel_ns"):
                samples.setdefault(r["test_name"], []).append(r["device_kernel_ns"])
            elif r.get("status") != "passed":
                samples.setdefault(r["test_name"], [])
    results = {
        t: {"status": "passed", "device_kernel_ns": statistics.median(v)} if v else {"status": "failed"}
        for t, v in samples.items()
    }
    ratios, rows = [], []
    for t in names:
        r = results.get(t)
        if not r or r.get("status") != "passed" or not r.get("device_kernel_ns"):
            rows.append((0.0, f"  ----  {(r or {}).get('status', 'missing'):>8}  {t}"))
            continue
        ratio = base[t]["main_ns"] / r["device_kernel_ns"]
        ratios.append(ratio)
        rows.append(
            (
                ratio,
                f"  {ratio:5.2f}  main {base[t]['main_ns']:9.0f}  1061 {base[t]['run1061_ns']:9.0f}  "
                f"now {r['device_kernel_ns']:9.0f}  {t}",
            )
        )
    for _, line in sorted(rows):
        print(line)
    if ratios:
        print(
            f"\n{len(ratios)}/{len(names)} measured | slower than main: {sum(r < 1 for r in ratios)} | "
            f"median ratio {statistics.median(ratios):.3f} | min {min(ratios):.2f}"
        )
    json.dump(
        {t: results.get(t, {}).get("device_kernel_ns") for t in names}, open(out_dir / "summary.json", "w"), indent=1
    )


if __name__ == "__main__":
    sys.exit(main())
