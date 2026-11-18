# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device): the (bm, bk) menu per focus layer under the op's L1 solve.

Stubs `ttnn` so the descriptor module's solve can be imported without starting UMD. Blackhole p150b budget:
get_max_worker_l1_unreserved_size() = 1530976 B.

    python3 -I blocking_menu.py
"""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
MAX_UNRESERVED = 1530976
BF8_TILE = 1088
BF16_TILE = 2048


def load_descriptor():
    stub = types.ModuleType("ttnn")
    stub.get_max_worker_l1_unreserved_size = lambda: MAX_UNRESERVED
    sys.modules.setdefault("ttnn", stub)
    spec = importlib.util.spec_from_file_location("pd_copy", HERE / "op_copy" / "conv2d_3x3_program_descriptor.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pd_copy"] = mod
    spec.loader.exec_module(mod)
    return mod


def div_up(a, b):
    return -(-a // b)


def focus_cases(pd):
    """(name, geometry, grid, budget) for the six focus layers (TILE bf8_b act, bf8_b weights, bias)."""
    out = []

    def block(name, H, W, c_in, c_out, pad, dil, stride, shard, grid_xy):
        gx, gy = grid_xy
        c_in_p = shard[1]
        k_cols = div_up(c_in, c_in_p)
        rows = gy  # ROW_MAJOR: rows = size.y
        h_out = (H + 2 * pad - (2 * dil + 1)) // stride + 1
        w_out = (W + 2 * pad - (2 * dil + 1)) // stride + 1
        g = pd.Conv3x3Geometry(
            input_height=H,
            input_width=W,
            c_in_p=c_in_p,
            c_out=c_out,
            padding=pad,
            stride=stride,
            dilation=dil,
            output_height=h_out,
            output_width=w_out,
            c_in=c_in,
            tile_input=True,
            in0_tile_bytes=BF8_TILE,
            weight_tile_bytes=BF8_TILE,
            out_tile_bytes=BF8_TILE,
            has_bias=True,
            sharded_output=True,
            k_cols=k_cols,
        )
        core_m, _ = pd.m_split(g, rows)
        in_pages = (shard[0] // 32) * (shard[1] // 32)
        resident = in_pages * BF8_TILE + core_m * g.out_n_tiles * BF8_TILE
        out.append((name, g, (rows, 1), pd.l1_cb_budget_bytes(resident)))

    def height(name, H, W, c_in, c_out, pad, dil, stride, shard, cores):
        c_in_p = div_up(c_in, 32) * 32
        h_out = (H + 2 * pad - (2 * dil + 1)) // stride + 1
        w_out = (W + 2 * pad - (2 * dil + 1)) // stride + 1
        g = pd.Conv3x3Geometry(
            input_height=H,
            input_width=W,
            c_in_p=c_in_p,
            c_out=c_out,
            padding=pad,
            stride=stride,
            dilation=dil,
            output_height=h_out,
            output_width=w_out,
            c_in=c_in,
            tile_input=True,
            in0_tile_bytes=BF8_TILE,
            weight_tile_bytes=BF8_TILE,
            out_tile_bytes=BF8_TILE,
            has_bias=True,
            sharded_output=True,
            k_cols=1,
        )
        core_m, _ = pd.m_split(g, cores)
        in_pages = (shard[0] // 32) * (shard[1] // 32)
        resident = in_pages * BF8_TILE + core_m * g.out_n_tiles * BF8_TILE
        out.append((name, g, (cores, 1), pd.l1_cb_budget_bytes(resident)))

    block("res5_1_conv2", 32, 64, 512, 512, 4, 4, 1, [224, 64], (8, 10))
    block("res5_2_conv2", 32, 64, 512, 512, 8, 8, 1, [224, 64], (8, 10))
    block("aspp_convs_1", 32, 64, 2048, 256, 6, 6, 1, [256, 192], (11, 8))
    block("aspp_convs_2", 32, 64, 2048, 256, 12, 12, 1, [256, 192], (11, 8))
    height("semantic", 128, 256, 256, 256, 1, 1, 1, [320, 256], 103)
    block("res4_0_conv2", 64, 128, 256, 256, 1, 1, 2, [1024, 32], (8, 8))
    return out


def menu(pd, g, grid, budget):
    core_m, _ = pd.m_split(g, grid[0] * grid[1])
    kkh = g.tensor_kkh_tiles
    bks = [d for d in range(kkh, 0, -1) if kkh % d == 0]
    bms = [d for d in range(core_m, 0, -1) if core_m % d == 0]
    rows = []
    for bm in bms:
        for bk in bks:
            band_len, raw = pd._block_extents(g, core_m, bm)
            ub_def = pd._default_untilize_batch_rows(g, band_len, raw)
            best = None
            for ub in range(ub_def, 0, -1):
                f = pd._footprint(g, core_m, bm, bk, g.tensor_n_tiles, ub)
                if f["footprint_bytes"] <= budget:
                    best = (ub, f)
                    break
            f_full = pd._footprint(g, core_m, bm, bk, g.tensor_n_tiles, None)
            rows.append((bm, bk, f_full["num_k_blocks"], bk % g.c_in_tiles == 0, best, f_full["footprint_bytes"]))
    return core_m, rows


def harness_cases(pd):
    """(name, geometry, grid, budget) for every shape of test_conv2d_3x3_perf.py (HEIGHT, BLOCK, interleaved x 3
    layouts; interleaved on the 11x10 Blackhole p150b grid, RM C_in_p rounded to 32 = 64 B DRAM alignment)."""
    height = [
        ("stem_conv1", ((512, 1024, 8), 64, 1, 2), BF16_TILE, False, [4767, 8], 110),
        ("stem_conv3", ((256, 512, 64), 128, 1, 1), BF8_TILE, True, [1216, 64], 110),
        ("res2_conv2", ((128, 256, 64), 64, 1, 1), BF8_TILE, True, [320, 64], 103),
        ("res3_0_conv2", ((128, 256, 128), 128, 1, 2), BF8_TILE, True, [384, 128], 86),
        ("res3_conv2", ((64, 128, 128), 128, 1, 1), BF8_TILE, True, [96, 128], 86),
        ("sem_res2_fuse_conv0", ((128, 256, 288), 256, 1, 1), BF8_TILE, True, [320, 288], 103),
        ("stem_conv2", ((256, 512, 64), 64, 1, 1), BF8_TILE, True, [1216, 64], 110),
        ("sem_res3_fuse_conv0", ((64, 128, 320), 256, 1, 1), BF8_TILE, True, [96, 320], 86),
        ("inst_res3_fuse_conv0", ((64, 128, 320), 128, 1, 1), BF8_TILE, True, [96, 320], 86),
        ("sem_res3_fuse_conv1", ((64, 128, 256), 256, 1, 1), BF8_TILE, True, [96, 256], 86),
        ("inst_res2_fuse_conv0", ((128, 256, 160), 128, 1, 1), BF8_TILE, True, [320, 160], 103),
        ("sem_res2_fuse_conv1", ((128, 256, 256), 256, 1, 1), BF8_TILE, True, [320, 256], 103),
        ("inst_res2_fuse_conv1", ((128, 256, 128), 128, 1, 1), BF8_TILE, True, [320, 128], 103),
        ("center_head1", ((128, 256, 128), 32, 1, 1), BF8_TILE, True, [320, 128], 103),
    ]
    block = [
        ("res4_0_conv2", ((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, 2),
        ("res4_conv2", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, 1),
        ("res5_0_conv2", ((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, 1),
        ("res5_1_conv2", ((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, 1),
        ("res5_2_conv2", ((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, 1),
        ("aspp_convs1", ((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, 1),
        ("aspp_convs2", ((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, 1),
        ("aspp_convs3", ((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, 1),
    ]
    interleaved = [
        ((32, 64, 128), 128, 1),
        ((16, 16, 64), 128, 1),
        ((8, 16, 512), 256, 1),
        ((8, 16, 2048), 256, 1),
        ((64, 128, 128), 128, 1),
        ((128, 256, 64), 64, 1),
    ]
    out = []
    for name, ((H, W, ci), co, p, s), tb, tile, shard, cores in height:
        cip = div_up(ci, 32) * 32 if tile else shard[1]
        g = pd.Conv3x3Geometry(
            input_height=H,
            input_width=W,
            c_in_p=cip,
            c_out=co,
            padding=p,
            stride=s,
            dilation=1,
            output_height=(H + 2 * p - 3) // s + 1,
            output_width=(W + 2 * p - 3) // s + 1,
            c_in=ci,
            tile_input=tile,
            in0_tile_bytes=tb,
            weight_tile_bytes=BF8_TILE,
            out_tile_bytes=tb,
            has_bias=True,
            sharded_output=True,
            k_cols=1,
        )
        cm, _ = pd.m_split(g, cores)
        inp = (shard[0] // 32) * (shard[1] // 32) * tb if tile else shard[0] * shard[1] * 2
        out.append(("H:" + name, g, (cores, 1), pd.l1_cb_budget_bytes(inp + cm * g.out_n_tiles * tb)))
    for name, ((H, W, ci), co, p, d), shard, (gx, gy), bias, s in block:
        cip = shard[1]
        g = pd.Conv3x3Geometry(
            input_height=H,
            input_width=W,
            c_in_p=cip,
            c_out=co,
            padding=p,
            stride=s,
            dilation=d,
            output_height=(H + 2 * p - (2 * d + 1)) // s + 1,
            output_width=(W + 2 * p - (2 * d + 1)) // s + 1,
            c_in=ci,
            tile_input=True,
            in0_tile_bytes=BF8_TILE,
            weight_tile_bytes=BF8_TILE,
            out_tile_bytes=BF8_TILE,
            has_bias=bias,
            sharded_output=True,
            k_cols=div_up(ci, cip),
        )
        cm, _ = pd.m_split(g, gy)
        res = (shard[0] // 32) * (shard[1] // 32) * BF8_TILE + cm * g.out_n_tiles * BF8_TILE
        out.append(("B:" + name, g, (gy, 1), pd.l1_cb_budget_bytes(res)))
    for (H, W, ci), co, p in interleaved:
        for tag, tb, tile in (
            ("rm_bf16", BF16_TILE, False),
            ("tile_bf16", BF16_TILE, True),
            ("tile_bf8b", BF8_TILE, True),
        ):
            g = pd.Conv3x3Geometry(
                input_height=H,
                input_width=W,
                c_in_p=div_up(ci, 32) * 32,
                c_out=co,
                padding=p,
                stride=1,
                dilation=1,
                output_height=H + 2 * p - 2,
                output_width=W + 2 * p - 2,
                c_in=ci,
                tile_input=tile,
                in0_tile_bytes=tb,
                weight_tile_bytes=BF16_TILE,
                out_tile_bytes=tb,
                has_bias=True,
                sharded_output=False,
                k_cols=1,
            )
            out.append((f"I:{H}x{W}x{ci}->{co}_{tag}", g, (11, 10), pd.l1_cb_budget_bytes(0)))
    return out


def rule_b(pd, g, b, budget):
    """Option B on top of rule A: at rule A's bm, take whole-kh bk (3 Ct) with single-buffered weights when it only
    fits that way and the fp32 pack passes it saves outweigh the weight K-blocks it exposes (tile counts). None when
    B does not apply."""
    ct = g.c_in_tiles
    if not pd.tap_major_eligible(g) or b.block_k_tiles == 3 * ct or b.block_k_tiles % ct:
        return None
    bm, core_m, bk3 = b.block_m_tiles, b.core_m_tiles, 3 * ct
    band_len, raw = pd._block_extents(g, core_m, bm)
    for ub in range(pd._default_untilize_batch_rows(g, band_len, raw), 0, -1):
        f = pd._footprint(g, core_m, bm, bk3, g.tensor_n_tiles, ub, 1)
        if f["footprint_bytes"] <= budget:
            saved = (g.tensor_k_tiles // b.block_k_tiles - 3) * bm * g.tensor_n_tiles
            exposed = 3 * bk3 * g.tensor_n_tiles
            return dict(
                bm=bm, bk=bk3, ub=ub, pack_tiles_saved=saved, weight_tiles_exposed=exposed, take=saved > exposed
            )
    return None


def rules_table():
    """Base vs rule A vs option B over every perf-harness shape."""
    pd = load_descriptor()
    for name, g, grid, budget in harness_cases(pd) + [("F:" + n, g, gr, bu) for n, g, gr, bu in focus_cases(pd)]:
        pd.TAP_ALIGNED_BK = False
        pd.select_blocking.cache_clear()
        b0 = pd.select_blocking(g, grid, budget)
        pd.TAP_ALIGNED_BK = True
        pd.select_blocking.cache_clear()
        b1 = pd.select_blocking(g, grid, budget)
        pd.TAP_ALIGNED_BK = False
        pd.select_blocking.cache_clear()
        rb = rule_b(pd, g, b1, budget)
        changed = (b0.block_m_tiles, b0.block_k_tiles) != (b1.block_m_tiles, b1.block_k_tiles)
        print(
            f"{name:34s} base bm {b0.block_m_tiles:2d} bk {b0.block_k_tiles:3d} tap {int(pd.tap_major_im2col(g, b0))} | "
            f"A bm {b1.block_m_tiles:2d} bk {b1.block_k_tiles:3d} tap {int(pd.tap_major_im2col(g, b1))}"
            f"{'  CHANGED' if changed else ''}" + (f" | B {rb}" if rb else "")
        )


def main():
    if "--rules" in sys.argv:
        return rules_table()
    pd = load_descriptor()
    for name, g, grid, budget in focus_cases(pd):
        b = pd.select_blocking(g, grid, budget)
        tm = pd.tap_major_im2col(g, b)
        print(
            f"\n== {name}: c_in_p {g.c_in_p} ({g.c_in_tiles} t), kkh {g.tensor_kkh_tiles}, k_cols {g.k_cols}, "
            f"bn {g.tensor_n_tiles}, budget {budget}"
        )
        print(
            f"   current: bm {b.block_m_tiles} bk {b.block_k_tiles} kb {b.num_k_blocks} ub {b.untilize_batch_rows} "
            f"fp {b.footprint_bytes} tap_major {tm}"
        )
        core_m, rows = menu(pd, g, grid, budget)
        for bm, bk, nkb, aligned, best, fp_full in rows:
            fit = f"fits ub {best[0]} fp {best[1]['footprint_bytes']}" if best else f"NO FIT (full ub fp {fp_full})"
            print(f"   bm {bm:2d} bk {bk:2d} kb {nkb:2d} mblk {div_up(core_m, bm)} tap_aligned {int(aligned)}  {fit}")


if __name__ == "__main__":
    main()
