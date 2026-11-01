"""parse.py <log> <tag>: per-row DEVICE KERNEL DURATION (perf summary line after each test id) + precision."""
import re
import sys

log, tag = sys.argv[1], sys.argv[2]
dur, prec, cur = {}, {}, None
for line in open(log, errors="replace"):
    m = re.search(r"test_lbf\[([^\]]+)\]", line)
    if m and "::" in line:
        cur = m.group(1)
    m = re.search(r"perf summary \(device=0\): count=1, min=(\d+)", line)
    if m and cur:
        dur[cur] = int(m.group(1))
    m = re.search(r"LBF_PRECISION (\S+) pcc=(\S+) rel_rms=(\S+)", line)
    if m and m.group(3) != "nan":
        prec[m.group(1)] = (m.group(2), m.group(3))
for row in dict.fromkeys(list(dur) + list(prec)):
    p = prec.get(row, ("-", "-"))
    print(f"{tag} {row} {dur.get(row, 'MISSING')} pcc={p[0]} rel={p[1]}")
