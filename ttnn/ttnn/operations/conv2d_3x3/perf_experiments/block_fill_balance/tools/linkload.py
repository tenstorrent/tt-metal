"""Host model of one BLOCK column's slab fill (aspp-like K split, one M-block per core): per core the slab rows by owner
shard and the RISC/NoC reading them, and per column link the bytes on NoC0 (+y) and NoC1 (-y). Physical column ring of
12 rows (worker rows y 2..11); grid row r on y 2 + r, input shard s on its owner row y 2 + s.

    python3 linkload.py [case]   (aspp1 | aspp2 | aspp3 | res51 | res52 | res50)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pdstub  # noqa: E402

pd = pdstub.load(str(HERE.parents[2] / "conv2d_3x3_program_descriptor.py"))

CASES = {
    # H, W, C_in, c_in_p (slice), d, shard rows (tile-rows), ranges, tile bytes
    "aspp1": (32, 64, 2048, 192, 6, 8, [7, 7, 6, 6, 6, 6, 6, 6, 7, 7], 1088),
    "aspp2": (32, 64, 2048, 192, 12, 8, [8, 8, 8, 4, 4, 4, 5, 7, 8, 8], 1088),
    "aspp3": (32, 64, 1024, 96, 18, 8, [6, 6, 6, 6, 8, 8, 6, 6, 6, 6], 1088),
}


def geom(H, W, cin, cinp, d):
    return pd.Conv3x3Geometry(
        input_height=H,
        input_width=W,
        c_in_p=cinp,
        c_out=256,
        padding=d,
        stride=1,
        dilation=d,
        output_height=H,
        output_width=W,
        c_in=cin,
        tile_input=True,
        in0_tile_bytes=1088,
        weight_tile_bytes=1088,
        out_tile_bytes=1088,
        has_bias=True,
        k_cols=cin // cinp,
    )


def ranges_of(sizes):
    out, a = [], 0
    for n in sizes:
        out.append((a, n))
        a += n
    return out


def hops0(src, dst, ring=12):
    return (dst - src) % ring  # +y


def links0(src, dst, ring=12):
    return [(src + i) % ring for i in range(hops0(src, dst, ring))]  # link y -> y + 1


def links1(src, dst, ring=12):
    return [(src - i) % ring for i in range(hops0(dst, src, ring))]  # link y -> y - 1


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "aspp1"
    H, W, cin, cinp, d, shard_rows, sizes, tb = CASES[case]
    g = geom(H, W, cin, cinp, d)
    rngs = ranges_of(sizes)
    row_bytes = (cinp // 32) * tb
    l0 = [0] * 12
    l1 = [0] * 12
    print(f"{case}: row {row_bytes} B")
    for m, (start, n) in enumerate(rngs):
        segs = pd._host_fill_segments(g, start, n)
        rows = [r for f, c in segs for r in range(f, f + c)]
        y = 2 + m
        writer_row = (m + 1) * shard_rows
        by = {}
        for r in rows:
            s = r // shard_rows
            by[s] = by.get(s, 0) + 1
            if r >= writer_row:
                for lk in links1(2 + s, y):
                    l1[lk] += row_bytes
            else:
                for lk in links0(2 + s, y):
                    l0[lk] += row_bytes
        print(
            f"  m {m} y {y} rows {len(rows):2d} ({len(rows)*row_bytes/1024:.0f} KB) segs {segs} by shard "
            + " ".join(f"s{s}:{c}(h0 {hops0(2+s, y)})" for s, c in sorted(by.items()))
        )
    print("  NoC0 link y->y+1 KB:", " ".join(f"{y}:{l0[y]/1024:.0f}" for y in range(12)))
    print("  NoC1 link y->y-1 KB:", " ".join(f"{y}:{l1[y]/1024:.0f}" for y in range(12)))


if __name__ == "__main__":
    main()
