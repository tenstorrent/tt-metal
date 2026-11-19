# Step-level schedule model of the ring walks: each real step costs 1 unit of a core's matmul time; a step's pack needs
# its incoming partial, which lands L units after the producing step finished. Makespan per core.
import sys


def baseline(k, owners):
    steps = {}
    for c in range(k):
        seq = []
        for j in range(k):
            o = (c - 1 - j) % k
            if o < owners:
                dep = ((c - 1) % k, j - 1) if j >= 1 else None
                seq.append((j, dep))
        steps[c] = seq
    return steps


def bidir(k, owners):
    na, nb = k // 2, (k - 1) // 2
    steps = {}
    for c in range(k):
        seq = []
        for j in range(k):
            if j == k - 1:
                o, ch = c, "own"
            elif j % 2 == k % 2:
                o, ch = (c + na - j // 2) % k, "A"
            else:
                o, ch = (c - (nb - j // 2)) % k, "B"
            if o >= owners:
                continue
            if ch == "A":
                dep = ((c - 1) % k, j - 2) if j // 2 >= 1 else None
            elif ch == "B":
                dep = ((c + 1) % k, j - 2) if j // 2 >= 1 else None
            else:
                dep = ((c + 1) % k, j - 2) if nb >= 1 else None  # B final; A final merged after
            late = ((c - 1) % k, j - 1) if ch == "own" else None
            seq.append((j, dep, late))
        steps[c] = seq
    return steps


def simulate(steps, L, bid=False):
    fin = {}
    t = {c: 0.0 for c in steps}
    end = {}
    changed = True
    # iterate until fixed point (dependencies across cores)
    for _ in range(200):
        for c in steps:
            tc = 0.0
            for item in steps[c]:
                j, dep = item[0], item[1]
                start = tc
                if dep is not None:
                    if dep not in fin:
                        start = float("inf")
                    else:
                        start = max(start, fin[dep] + L - 1)  # matmul overlaps; the pack (end) needs the partial
                done = start + 1
                fin[(c, j)] = done
                tc = done
                if bid and len(item) > 2 and item[2] is not None:
                    l = item[2]
                    tc = max(tc, fin.get(l, float("inf")) + L)
            end[c] = tc
    return end


for k, owners in [(8, 8), (11, 8)]:
    for L in [0.2, 0.5, 1.0]:
        b = simulate(baseline(k, owners), L)
        d = simulate(bidir(k, owners), L, True)
        print(
            f"k={k} owners={owners} L={L}: baseline max {max(b.values()):.2f} per-core {[round(v,1) for v in b.values()]}"
        )
        print(f"                       bidir    max {max(d.values()):.2f} per-core {[round(v,1) for v in d.values()]}")


def relax(k, owners):
    steps = {}
    for c in range(k):
        seq = []
        for j in range(k):
            o = (c - 1 - j) % k
            if o < owners:
                dep = ((c - 1) % k, j - 2) if j >= 2 else None
                seq.append((j, dep))
        steps[c] = seq
    return steps


def simulate2(steps, L, bid=False):
    fin = {}
    end = {}
    for _ in range(300):
        for c in steps:
            tc = 0.0
            for item in steps[c]:
                j, dep = item[0], item[1]
                start = tc
                if dep is not None:
                    # dep may be a non-real step on the neighbour (relax): take the latest real step <= it
                    dc, dj = dep
                    cands = [jj for (cc, jj) in fin if cc == dc and jj <= dj]
                    if cands:
                        start = max(start, fin[(dc, max(cands))] + L - 1)
                done = start + 1
                fin[(c, j)] = done
                tc = done
                if bid and len(item) > 2 and item[2] is not None:
                    tc = max(tc, fin.get(item[2], 0) + L)
            end[c] = tc
    return end


print("--- relax model")
for k, owners in [(8, 8), (11, 8)]:
    for L in [0.5, 1.0, 1.5, 2.0]:
        b = simulate(baseline(k, owners), L)
        r = simulate2(relax(k, owners), L)
        d = simulate(bidir(k, owners), L, True)
        print(
            f"k={k} o={owners} L={L}: base {max(b.values()):.2f} relax {max(r.values()):.2f} bidir {max(d.values()):.2f}"
        )


def baseline_set(k, owner_cols):
    steps = {}
    for c in range(k):
        seq = []
        for j in range(k):
            o = (c - 1 - j) % k
            if o in owner_cols:
                dep = ((c - 1) % k, j - 1) if j >= 1 else None
                seq.append((j, dep))
        steps[c] = seq
    return steps


print("--- owner placement (baseline ring, k=11, 8 owners)")
for name, cols in [
    ("0..7", set(range(8))),
    ("spread 2,5,8 idle", {0, 1, 3, 4, 6, 7, 9, 10}),
    ("spread 3,7,10 idle", {0, 1, 2, 4, 5, 6, 8, 9}),
]:
    for L in [0.5, 1.0]:
        b = simulate(baseline_set(11, cols), L)
        print(f"{name:>20} L={L}: max {max(b.values()):.2f}")
