"""ric_medians.py LOG [per_combo=12]: median DEVICE KERNEL DURATION of launches 2..N of each combo of a
test_rm_issue_cost.py run. Combos (variant-arm) are read from the RIC_COMBO lines of the log, in order;
GenericOp rows of the profiler CSV named in the log are split per_combo at a time, in CSV order."""
import csv
import re
import statistics
import sys

log = open(sys.argv[1], errors="replace").read()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
csv_path = re.findall(r"SAFE_PYTEST: PROFILER CSV: (\S+)", log)[-1]
combos = []
for m in re.finditer(r"^RIC_COMBO (\S+) (.*)$", log, re.M):
    if not combos or combos[-1][0] != m.group(1):
        combos.append((m.group(1), m.group(2)))
# A run whose tracy host capture failed has device columns only (no OP CODE): every row is then this op's.
rows = [r for r in csv.DictReader(open(csv_path)) if r.get("OP CODE", "Generic").startswith("Generic")]
# The precompile collect pass prints the combos once more before the real pass: keep the real pass's.
combos = combos[len(combos) - len(rows) // n :]
assert len(rows) == n * len(combos), (len(rows), n, len(combos))


def col(rs, c):
    v = [float(r[c]) for r in rs if r.get(c) not in (None, "", "-")]
    return f"{statistics.median(v):.0f}" if v else "-"


def spread(rs, c):
    v = [float(r[c]) for r in rs if r.get(c) not in (None, "", "-")]
    return f"{min(v):.0f}-{max(v):.0f}" if v else "-"


print(f"csv={csv_path}")
for i, (name, info) in enumerate(combos):
    rs = rows[i * n + 1 : (i + 1) * n]
    k = "DEVICE KERNEL DURATION [ns]"
    print(
        f"{name:28s} n={len(rs)} kernel={col(rs, k)} ({spread(rs, k)}) "
        f"BRISC={col(rs, 'DEVICE BRISC KERNEL DURATION [ns]')} NCRISC={col(rs, 'DEVICE NCRISC KERNEL DURATION [ns]')} "
        f"cores={rs[0].get('CORE COUNT', '?')} {info}"
    )
