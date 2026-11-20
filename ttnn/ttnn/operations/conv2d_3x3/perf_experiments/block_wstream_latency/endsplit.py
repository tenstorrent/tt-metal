# usage: endsplit.py <csv> <run_index> <k_cols>: grid (logical x < k_cols) vs spare core kernel ends (host-only)
import sys, re, subprocess

out = subprocess.run(
    [sys.executable, __file__.replace("endsplit.py", "rounds_tl.py"), sys.argv[1], sys.argv[2], "writer_weight_fetch"],
    capture_output=True,
    text=True,
).stdout
k = int(sys.argv[3])
rows = []
for l in out.splitlines():
    m = re.match(r"\(\s*(\d+),(\d+)\) phys\(\d+, \d+\) end ([0-9.]+)", l)
    if m:
        rows.append((float(m.group(3)), int(m.group(1)), int(m.group(2))))
rows.sort(reverse=True)
grid = sorted(e for e, x, y in rows if x < k)
sp = sorted(e for e, x, y in rows if x >= k)
print("top", " ".join(f"({x},{y}):{e:.2f}" for e, x, y in rows[:6]))
print("grid n %d max %.2f med %.2f" % (len(grid), grid[-1], grid[len(grid) // 2]), end="")
if sp:
    print(" | spare n %d max %.2f med %.2f" % (len(sp), sp[-1], sp[len(sp) // 2]))
else:
    print()
