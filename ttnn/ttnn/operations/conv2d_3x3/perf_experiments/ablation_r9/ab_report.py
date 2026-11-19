# usage: ab_report.py <ab_results.txt> <ids.txt>: per-case base/final means (us), the focus table and per-inference total.
import ast, statistics, sys

runs = {}
for line in open(sys.argv[1]):
    label, vals = line.split(" ", 1)
    runs[label] = ast.literal_eval(vals)
ids = [l.strip() for l in open(sys.argv[2])]
base = [k for k in runs if k.startswith("base")]
final = [k for k in runs if k.startswith("final")]
# feature_spec _PERF_FOCUS order = focus ids order: (t_ideal, calls, priority)
focus = [
    ("stem.conv1", 745, 1, 15),
    ("stem.conv2", 15888, 1, 5),
    ("stem.conv3", 31775, 1, 11),
    ("res2", 3972, 3, 8),
    ("res3.0", 3972, 1, 21),
    ("res3.x", 3972, 4, 4),
    ("res4.0", 3972, 1, 22),
    ("res4.x", 3972, 5, 3),
    ("res5.0", 15888, 1, 20),
    ("res5.1", 15888, 1, 19),
    ("res5.2", 15888, 1, 18),
    ("aspp.convs.1", 31775, 2, 7),
    ("aspp.convs.2", 31775, 2, 6),
    ("aspp.convs.3", 15888, 4, 1),
    ("sem dec res3 fc0", 19859, 1, 13),
    ("inst dec res3 fc0", 9930, 1, 16),
    ("sem dec res3 fc1", 15888, 1, 14),
    ("sem dec res2 fc0", 71494, 1, 12),
    ("inst dec res2 fc0", 19859, 1, 17),
    ("sem res2 fc1", 63550, 3, 2),
    ("inst res2 fc1", 15888, 3, 10),
    ("center_head.1", 3972, 2, 9),
]
n = len(ids)
mb = [statistics.mean(runs[k][i] for k in base) for i in range(n)]
mf = [statistics.mean(runs[k][i] for k in final) for i in range(n)]
f0 = n - len(focus)
print("guard set (non-focus):")
for i in range(f0):
    rb = [runs[k][i] for k in base]
    rf = [runs[k][i] for k in final]
    print(
        f"  {ids[i]:70s} {mb[i]:8.2f} -> {mf[i]:8.2f} {100*(mf[i]/mb[i]-1):+6.1f}%  base {min(rb):.2f}-{max(rb):.2f} final {min(rf):.2f}-{max(rf):.2f}"
    )
print("focus:")
tb = tf = 0
rows = []
for j, (name, ideal, calls, prio) in enumerate(focus):
    i = f0 + j
    tgt = ideal / 0.7 / 1000
    tb += mb[i] * calls
    tf += mf[i] * calls
    rows.append((prio, name, mb[i], mf[i], tgt, ideal / 1000))
for prio, name, b, f, tgt, ideal in sorted(rows):
    print(
        f"| {prio} | {name} | {b:.2f} | {f:.2f} | {b/f:.2f}x | {tgt:.1f} | {100*ideal/b:.0f}% -> {100*ideal/f:.0f}% |"
    )
print(f"per inference: {tb:.0f} -> {tf:.0f} us ({100*(tf/tb-1):+.1f}%)")
