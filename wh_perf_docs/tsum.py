import sys, re, collections, subprocess
# tsum.py <run.cyc> <name>: steady-state per-tile summary from tilean.py
out = subprocess.run(["python3", "/tmp/tilean.py", sys.argv[1], "100000"], capture_output=True, text=True).stdout
rows = [l.replace("l1wait", "W").replace(" l1 ", " L ") for l in out.splitlines() if re.match(r"^\d+ \d+ \d+ ref", l)]
T = []
for l in rows:
    v = list(map(int, re.findall(r"-?\d+", l)))
    T.append((v[0], v[2], v[3:7], v[7:11]))
n = len(T); T = [t for t in T if 3 <= t[0] <= n - 4]
s = lambda j, k: sum(t[j][k] for t in T) / len(T)
print(f"{sys.argv[2]} tiles {n} mean len {sum(t[1] for t in T)/len(T):.1f} | DEST refused/tile p1-3 {s(2,1):.1f} {s(2,2):.1f} {s(2,3):.1f} | L1 wait/tile p0-3 {s(3,0):.1f} {s(3,1):.1f} {s(3,2):.1f} {s(3,3):.1f}")
print("   lengths", collections.Counter(t[1] for t in T).most_common(5))
