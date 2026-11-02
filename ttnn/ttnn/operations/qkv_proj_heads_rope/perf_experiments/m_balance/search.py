"""Brute-force mixed-h_c assignments: regions (h_c, num M-groups, rows) on <= cores; minimize busiest tiles."""
import itertools, sys

HC = (1, 2, 3, 4, 6, 12)


def search(mt, cores=110, top=8):
    res = []
    for counts in itertools.product(*[range(0, cores // (36 // h) + 1) for h in HC]):
        used = sum(c * (36 // h) for c, h in zip(counts, HC))
        if (
            used > cores
            or sum(counts) == 0
            or sum(1 for c in counts if c) > 3
            or any(c and h == 1 for c, h in zip(counts, HC))
        ):
            continue
        # distribute rows: binary-search T
        for T in range(1, 200):
            cap = sum(c * (T // (2 * h)) for c, h in zip(counts, HC))
            if cap >= mt:
                break
        regions = [(h, c, T // (2 * h)) for c, h in zip(counts, HC) if c]
        xrecv = max(36 // h for h, c, _ in regions)
        res.append((T, used, xrecv, regions))
    res.sort(key=lambda r: (r[0], r[2], r[1]))
    seen = set()
    for r in res:
        key = tuple((h, c) for h, c, _ in r[3])
        if key in seen:
            continue
        seen.add(key)
        print(r)
        top -= 1
        if not top:
            break


for mt in map(int, sys.argv[1:]):
    print("Mt", mt)
    search(mt)
