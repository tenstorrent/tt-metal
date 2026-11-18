# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only L1 analysis for the HEIGHT N-slice-outer walk (no device, no torch: `ttnn` is stubbed).

    python3 hns_l1.py [op_copy|base]

Prints, per HEIGHT [perf] case, the solved blocking and footprint of the op copy (its select_blocking) and, when the copy
has the HEIGHT N-slice solve, which walk it picked."""

import enum
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


class _Fid(enum.Enum):
    LoFi = 0
    HiFi2 = 2
    HiFi3 = 3
    HiFi4 = 4


stub = types.ModuleType("ttnn")
stub.MathFidelity = _Fid
stub.get_max_worker_l1_unreserved_size = lambda: 1530976
sys.modules["ttnn"] = stub

which = sys.argv[1] if len(sys.argv) > 1 else "op_copy"
src = HERE / "op_copy" / "conv2d_3x3_program_descriptor.py"
if which == "base":
    src = HERE.parents[1] / "conv2d_3x3_program_descriptor.py"
spec = importlib.util.spec_from_file_location("hns_pd", src)
pd = importlib.util.module_from_spec(spec)
sys.modules["hns_pd"] = pd
spec.loader.exec_module(pd)

T8 = 1088  # bf8_b tile
# (name, (H, W, C_in), C_out, stride, cores)
CASES = [
    ("stem2", (256, 512, 64), 64, 1, 110),
    ("stem3", (256, 512, 64), 128, 1, 110),
    ("res2", (128, 256, 64), 64, 1, 103),
    ("res3_0", (128, 256, 128), 128, 2, 86),
    ("res3_x", (64, 128, 128), 128, 1, 86),
    ("semdec_r3_fc0", (64, 128, 320), 256, 1, 86),
    ("instdec_r3_fc0", (64, 128, 320), 128, 1, 86),
    ("semdec_r3_fc1", (64, 128, 256), 256, 1, 86),
    ("semdec_r2_fc0", (128, 256, 288), 256, 1, 103),
    ("instdec_r2_fc0", (128, 256, 160), 128, 1, 103),
    ("sem_r2_fc1", (128, 256, 256), 256, 1, 103),
    ("inst_r2_fc1", (128, 256, 128), 128, 1, 103),
    ("center1", (128, 256, 128), 32, 1, 103),
]


def geometry(H, W, c_in, c_out, stride):
    h_out = (H + 2 - 3) // stride + 1
    w_out = (W + 2 - 3) // stride + 1
    return pd.Conv3x3Geometry(
        input_height=H,
        input_width=W,
        c_in_p=c_in,
        c_out=c_out,
        padding=1,
        stride=stride,
        dilation=1,
        output_height=h_out,
        output_width=w_out,
        c_in=c_in,
        tile_input=True,
        in0_tile_bytes=T8,
        weight_tile_bytes=T8,
        out_tile_bytes=T8,
        has_bias=True,
        sharded_output=True,
    )


def main():
    for name, (H, W, c_in), c_out, stride, cores in CASES:
        g = geometry(H, W, c_in, c_out, stride)
        core_m, _ = pd.m_split(g, cores)
        in_rows = pd._div_up(H * W, cores * 32)
        resident = in_rows * g.c_in_tiles * T8 + core_m * g.tensor_n_tiles * T8
        budget = pd.l1_cb_budget_bytes(resident)
        b = pd.select_blocking(g, (cores, 1), budget)
        kwf = pd.kw_fold_mode(g, b, True, budget, _Fid.LoFi)
        print(
            f"{name:15s} K={g.tensor_k_tiles:3d} N={g.tensor_n_tiles:2d} core_m={core_m:2d} bm={b.block_m_tiles:2d} "
            f"bk={b.block_k_tiles:2d} nkb={b.num_k_blocks} nslice={int(b.nslice)} slice={getattr(b, 'slice_n_tiles', '-')} "
            f"wd={b.weight_depth} center={b.center_mode} ring={int(b.ring)} ub={b.untilize_batch_rows} "
            f"slab={b.slab_tile_rows} kwf={int(kwf)} fp={b.footprint_bytes / 1024:.0f}/{budget / 1024:.0f} KB"
        )


if __name__ == "__main__":
    main()


def breakdown(name, slice_n=1, depth=2, bias_fold=True, ub_cap=None, center="copy", ring=True, bm=None):
    """Footprint of a HEIGHT N-slice-outer walk at the K-blocked solve's bm (or `bm`), per CB, in KB.
    center: 'copy' (all three kw = 1 K-blocks in cb_center_tiles), 'shard' (kh = 1 read from the input shard,
    kh = 0 / 2 copied), 'none' (tilized)."""
    case = next(c for c in CASES if c[0] == name)
    _, (H, W, c_in), c_out, stride, cores = case
    g = geometry(H, W, c_in, c_out, stride)
    core_m, _ = pd.m_split(g, cores)
    in_rows = pd._div_up(H * W, cores * 32)
    resident = in_rows * g.c_in_tiles * T8 + core_m * g.tensor_n_tiles * T8
    budget = pd.l1_cb_budget_bytes(resident)
    b = pd.select_blocking(g, (cores, 1), budget)
    bm = bm or b.block_m_tiles
    bk = b.block_k_tiles
    K, N, ct = g.tensor_k_tiles, g.tensor_n_tiles, g.c_in_tiles
    band_len, raw_rows = pd._block_extents(g, core_m, bm)
    ub = pd._default_untilize_batch_rows(g, band_len, raw_rows)
    if ub_cap:
        ub = min(ub, ub_cap)
    slab_rows = pd.ring_rows(g, core_m, bm) if ring else pd._div_up(raw_rows, ub) * ub
    center_tiles = {"copy": 3 * bm * ct, "shard": 2 * bm * ct, "none": 0}[center] if bk == ct else 0
    tilized = K - (3 * ct if (center != "none" and bk == ct) else 0)
    parts = dict(
        slab=slab_rows * ct * 2048,
        input_tiles=2 * ub * ct * T8,
        im2col_rm=2 * bk * 2048,
        zero_src=32 * g.c_in_p * 2,
        in0=bm * tilized * T8,
        center=center_tiles * T8,
        weights=depth * K * slice_n * T8,
        partials=0 if bias_fold else 2 * bm * slice_n * 4096,
        bias=N * T8,
        table=bm * 16,
    )
    total = sum(parts.values())
    print(
        f"{name} bm={bm} bk={bk} s={slice_n} D={depth} fold={int(bias_fold)} ub={ub} center={center} ring={int(ring)}: "
        + " ".join(f"{k}={v / 1024:.0f}" for k, v in parts.items())
        + f" | total {total / 1024:.0f} / {budget / 1024:.0f} KB {'FITS' if total <= budget else 'over'}"
    )
    return total, budget
