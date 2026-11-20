"""For op_copy's balanced assignment (writer takes the first rows of each NoC1 piece, slab order), the band-0-only and
whole-fill resource bounds (noc_lp model), vs a band-0-first balance (_balance_fill_nocs on band 0, then bands 1-2)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import linkload  # noqa: E402
import noc_lp  # noqa: E402
import pdstub  # noqa: E402

pdc = pdstub.load(str(HERE.parent / "op_copy" / "conv2d_3x3_program_descriptor.py"), "pdc")
case = sys.argv[1] if len(sys.argv) > 1 else "aspp1"
H, W, cin, cinp, d, shard_rows, sizes, tb = linkload.CASES[case]
g = linkload.geom(H, W, cin, cinp, d)
row_bytes = (cinp // 32) * tb
cores = []  # per m: list of (row, shard, band)
for m, (start, n) in enumerate(linkload.ranges_of(sizes)):
    segs = linkload.pd._host_fill_segments(g, start, n)
    rows = [r for a, c in segs for r in range(a, a + c)]
    o_first = start * 32
    f = linkload.pd._block_first_tap(g, o_first)
    bl = linkload.pd._block_band_len(g, o_first, min((start + n) * 32, g.num_out_px) - 1)
    out = []
    for r in rows:
        band = next(k for k in range(3) if (f + k * d * W) // 32 <= r <= (f + k * d * W + bl - 1) // 32)
        out.append((r, r // shard_rows, band))
    cores.append(out)


def pieces_of(rows):
    """Consecutive rows of one shard -> pieces [(shard, [rows])]."""
    ps = []
    for r, s, b in rows:
        if ps and ps[-1][0] == s and ps[-1][1][-1] == r - 1:
            ps[-1][1].append(r)
        else:
            ps.append((s, [r]))
    return ps


def balance(subsets):
    """subsets: per m the rows to balance; returns per m the set of writer rows."""
    column = []
    pcs = []
    for m, rows in enumerate(subsets):
        ps = pieces_of(rows)
        pcs.append(ps)
        column.append((2 + m, [(2 + s, len(rr)) for s, rr in ps]))
    x1 = pdc._balance_fill_nocs(column, 12, row_bytes)
    w = []
    for ps, xs in zip(pcs, x1):
        ws = set()
        for (s, rr), k in zip(ps, xs):
            ws.update(rr[:k])
        w.append(ws)
    return w


def evaluate(name, writer):
    for label, bands in (("band0", {0}), ("all", {0, 1, 2})):
        pcs, x1 = [], []
        for m, rows in enumerate(cores):
            for s in sorted({s for r, s, b in rows if b in bands}):
                rr = [r for r, s2, b in rows if s2 == s and b in bands]
                pcs.append((m, s, len(rr)))
                x1.append(sum(1 for r in rr if r in writer[m]))
        res = noc_lp.loads(pcs, x1, row_bytes)
        print(f"  {name:8s} {label}: {max(res.values()) / noc_lp.RATE:.2f} us", end="")
    print()


direction = [{r for r, s, b in rows if s > m} for m, rows in enumerate(cores)]
evaluate("dir", direction)
evaluate("bal", balance(cores))
w0 = balance([[x for x in rows if x[2] == 0] for rows in cores])
w12 = balance([[x for x in rows if x[2] != 0] for rows in cores])
evaluate("bal0+12", [a | b for a, b in zip(w0, w12)])
