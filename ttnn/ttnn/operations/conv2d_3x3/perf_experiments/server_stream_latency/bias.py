# usage: bias.py <csv> <case>: kw-fold bias landing (reader_kwf_bias_wait end), decode time, compute final start/end.
import re, subprocess, sys

out = subprocess.run(
    [
        sys.executable,
        __file__.replace("bias.py", "zones.py"),
        sys.argv[1],
        sys.argv[2],
        "reader_kwf_bias,compute_kwf_final,compute_kwf_matmul_shift",
    ],
    capture_output=True,
    text=True,
).stdout
rows = {}
for l in out.splitlines():
    m = re.match(r"\(\((\d+), (\d+)\), '(\w+)'\) \d+ \[([\d.]+)-([\d.]+)\]", l)
    if m:
        rows.setdefault((m[1], m[2]), {}).setdefault(m[3], (float(m[4]), float(m[5])))


def pct(v):
    v = sorted(v)
    return f"min {v[0]:.2f} p50 {v[len(v)//2]:.2f} max {v[-1]:.2f}"


w = [v for v in rows.values() if "reader_kwf_bias_wait" in v]
print("bias lands      ", pct([v["reader_kwf_bias_wait"][1] for v in w]))
print("bias decode us  ", pct([v["reader_kwf_bias_rows"][1] - v["reader_kwf_bias_wait"][1] for v in w]))
print("rows ready      ", pct([v["reader_kwf_bias_rows"][1] for v in w]))
print(
    "matmul end      ",
    pct([v["compute_kwf_matmul_shift"][1] for v in rows.values() if "compute_kwf_matmul_shift" in v]),
)
print("final end       ", pct([v["compute_kwf_final"][1] for v in rows.values() if "compute_kwf_final" in v]))
