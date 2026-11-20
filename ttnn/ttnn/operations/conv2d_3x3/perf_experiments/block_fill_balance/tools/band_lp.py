"""Per-band resource bounds (noc_lp's model): the tap build consumes band 0, then 1, then 2 (kernel-row major). For band
sets {0}, {1}, {2}, {0,1,2}: the binding resource (us) with the base split and with the LP optimum."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import linkload  # noqa: E402
import noc_lp  # noqa: E402


def band_pieces(case):
    H, W, cin, cinp, d, shard_rows, sizes, tb = linkload.CASES[case]
    g = linkload.geom(H, W, cin, cinp, d)
    out = {0: [], 1: [], 2: []}
    for m, (start, n) in enumerate(linkload.ranges_of(sizes)):
        segs = linkload.pd._host_fill_segments(g, start, n)
        # band of each slab row: the plan's segments in kh order; a segment that shares tiles with the previous band
        # continues it, so assign rows by kh windows.
        o_first = start * 32
        o_last = min((start + n) * 32, g.num_out_px) - 1
        f = linkload.pd._block_first_tap(g, o_first)
        bl = linkload.pd._block_band_len(g, o_first, o_last)
        rows = [r for a, c in segs for r in range(a, a + c)]
        for r in rows:
            kh = 0
            for k in range(3):
                lo = f + k * g.dilation * g.input_width
                if lo // 32 <= r <= (lo + bl - 1) // 32:
                    kh = k
                    break
            out[kh].append((m, r // shard_rows))
    res = {}
    for kh, lst in out.items():
        by = {}
        for m, s in lst:
            by[(m, s)] = by.get((m, s), 0) + 1
        res[kh] = [(m, s, c) for (m, s), c in sorted(by.items())]
    return res, (cinp // 32) * tb


case = sys.argv[1] if len(sys.argv) > 1 else "aspp1"
sender = int(sys.argv[2]) if len(sys.argv) > 2 else None
bp, rb = band_pieces(case)
for name, sets in (("band0", [0]), ("band1", [1]), ("band2", [2]), ("all", [0, 1, 2])):
    pcs = []
    for kh in sets:
        pcs += bp[kh]
    base = [c if (s > m and m != sender) else 0 for m, s, c in pcs]
    rb_ = noc_lp.loads(pcs, base, rb)
    x = noc_lp.lp(pcs, rb)
    rl = noc_lp.loads(pcs, x, rb)
    tb = max(rb_.values()) / noc_lp.RATE
    tl = max(rl.values()) / noc_lp.RATE
    print(f"{case} {name}: rows {sum(c for *_, c in pcs)}  base {tb:.2f} us  lp {tl:.2f} us")
