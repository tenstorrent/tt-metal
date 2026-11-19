# Host-only check (no ttnn import): the descriptor's host-planned kw-fold fill (_kwf_live_segments / _kwf_fill_args,
# extracted from op_copy/conv2d_3x3_program_descriptor.py) against an independent model: every row compute reads
# (kw_fold.hpp run(): tile-rows [t0 - ht, t_end + hb) of each in-image kernel row) lands at the slab row compute's
# slab_bases() gives it, exactly once; pushes cover the slab exactly; the writer's pieces are a suffix and come only
# from higher shards on a split core.
import ast, sys
from types import SimpleNamespace

src = open(sys.argv[1] if len(sys.argv) > 1 else "op_copy/conv2d_3x3_program_descriptor.py").read()
tree = ast.parse(src)
want = {"_kwf_live_segments", "_kwf_fill_args", "_block_band_len", "_block_first_tap", "_div_up"}
ns = {
    "TILE": 32,
    "KWF_FILL_GROUP_ROWS": 8,
    "KWF_FILL_MAX_GROUPS": 16,
    "Conv3x3Geometry": object,
    "KWF_FILL_BALANCE": __import__("os").environ.get("BAL") == "1",
}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module([node], []), "desc", "exec"), ns)
    if isinstance(node, ast.Assign) and any(
        getattr(t, "id", "") in ("KWF_FILL_GROUP_ROWS", "KWF_FILL_MAX_GROUPS") for t in node.targets
    ):
        exec(compile(ast.Module([node], []), "desc", "exec"), ns)


def geom(H, W, d):
    return SimpleNamespace(
        input_height=H,
        input_width=W,
        output_height=H,
        output_width=W,
        dilation=d,
        padding=d,
        stride=1,
        band_last_tap=2,
        num_in_px=H * W,
        num_out_px=H * W,
    )


def slab_bases(W, num_in_px, d, o_first, o_end):
    f = o_first - d * W - d
    band_len = o_end - 1 - o_first + 2 * d + 1
    base = [0, 0, 0]
    next_free, prev = 0, None
    for kh in range(3):
        v_lo = f + kh * d * W
        lo, hi = max(v_lo, 0), min(v_lo + band_len, num_in_px)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // 32, (hi + 31) // 32
        row = prev[2] + (t_lo - prev[0]) if prev is not None and t_lo < prev[1] else next_free
        base[kh] = row - t_lo
        next_free = row + (t_hi - t_lo)
        prev = (t_lo, t_hi, row)
    return base


def check(name, H, W, d, per_core, bm, xchg, split_all=True, slab_rows=None):
    g = geom(H, W, d)
    tpr, m_tiles = W // 32, H * W // 32
    R = d * tpr
    if slab_rows is None:
        slab_rows = -(-(bm + 2 + 2 * R) // 4) * 4
    ncores = -(-m_tiles // per_core)
    shard_xy = [(i, 100 + i) for i in range(ncores)]
    row_bytes = 1
    pieces_total = groups_max = 0
    for core in range(ncores):
        start, actual = core * per_core, min(per_core, m_tiles - core * per_core)
        for m in range(-(-actual // bm)):
            t0, valid = start + m * bm, min(bm, actual - m * bm)
            rd, wr = ns["_kwf_fill_args"](g, t0, valid, xchg, slab_rows, per_core, core, split_all, shard_xy, row_bytes)
            # decode
            written = {}
            ngroups, nwg = rd[0], rd[1]
            i = 2
            pushed = 0
            for gi in range(ngroups):
                rows = rd[i]
                i += 1
                if gi < ngroups - nwg:
                    n = rd[i]
                    i += 1
                    for _ in range(n):
                        xy, off, dst = rd[i : i + 3]
                        i += 3
                        sh = xy >> 16
                        assert sh <= core or not split_all, (name, "reader reads a higher shard on a split core")
                        for k in range(dst & 0xFFFF):
                            srow = pushed + (dst >> 16) + k
                            assert srow not in written, (name, core, m, srow)
                            written[srow] = sh * per_core + off + k
                    pieces_total += n
                pushed += rows
            assert i == len(rd), (name, i, len(rd))
            assert pushed == slab_rows, (name, core, m, pushed, slab_rows)
            groups_max = max(groups_max, ngroups)
            assert wr[0] == nwg
            j = 1
            for _ in range(wr[0]):
                n = wr[j]
                j += 1
                for _ in range(n):
                    xy, off, dst = wr[j : j + 3]
                    j += 3
                    sh = xy >> 16
                    assert sh >= core
                    for k in range(dst & 0xFFFF):
                        srow = (dst >> 16) + k
                        assert srow not in written
                        written[srow] = sh * per_core + off + k
            assert j == len(wr)
            # model: rows compute reads
            base = slab_bases(W, H * W, d, t0 * 32, (t0 + valid) * 32)
            xt = xchg and m == 0 and t0 % tpr != 0  # exchange replaces both halos (one block per core)
            ht = 1 if (not xchg and t0 % tpr != 0) else 0
            t_end = t0 + valid
            hb = 1 if (not xchg and t_end % tpr != 0 and t_end < m_tiles) else 0
            need = {}
            for kh in range(3):
                for t in range(t0 - ht, t_end + hb):
                    r = t + (kh - 1) * R
                    if 0 <= r < m_tiles:
                        need[r + base[kh]] = r
            for srow, r in need.items():
                assert written.get(srow) == r, (name, core, m, srow, r, written.get(srow))
            assert set(written) == set(need), (name, core, m, sorted(set(written) - set(need)))
    print(f"{name}: ok, {ncores} cores, max groups {groups_max}, reader pieces {pieces_total}")


check("res2", 128, 256, 1, 10, 10, True)
check("res2_noxchg", 128, 256, 1, 10, 10, False)
check("res2_bm5", 128, 256, 1, 10, 5, False)
check("res3x", 64, 128, 1, 3, 3, True)
check("res3x_noxchg", 64, 128, 1, 3, 3, False)
check("ch1/sem res2", 128, 256, 1, 10, 10, True)
check("stem2", 256, 512, 1, 38, 38, True)
check("stem2_bm19", 256, 512, 1, 38, 19, False)
check("stem2_nosplit", 256, 512, 1, 38, 38, True, split_all=False)
check("d2", 64, 128, 2, 6, 6, True)
check("d2_bm3", 64, 128, 2, 6, 3, False)
check("w64", 32, 64, 1, 2, 2, True)
check("w64_bm1", 32, 64, 1, 3, 1, False)
check("ragged", 64, 128, 1, 5, 5, True)
check("ragged_bm2", 64, 128, 1, 5, 2, False)
check("d3_w96", 48, 96, 3, 7, 7, True)
import os

if os.environ.get("MB"):
    for bm in (256, 128, 64, 32, 16, 8):
        try:
            check(f"mb_whole_k_bm{bm}", 128, 256, 1, 256, bm, False)
        except AssertionError as e:
            print("FAIL", bm, str(e)[:200])
