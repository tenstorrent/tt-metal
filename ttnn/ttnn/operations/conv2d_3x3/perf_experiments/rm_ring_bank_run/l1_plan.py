# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (no device, ttnn stubbed): the blocking op_base and op_copy (ring + bank-run) solve on the FLUX.2 bh2x4
focus cases, the ring's L1 and the fill pixels per core.

    python3 -I l1_plan.py [knob=value ...]      e.g. RM_RING_NSTG=2 RM_RING_CHUNK_BYTES=8192

Budget = 1,334,368 B (l1_cb_budget_bytes(0, 65536) on the p150b)."""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
BUDGET = 1334368
UNRESERVED = BUDGET + 128 * 1024 + 65536

# idx -> (tag, H, W, C_in, C_out)
FOCUS = {
    41: ("P1", 1026, 2048, 32, 128),
    35: ("P3", 1026, 2048, 128, 3),
    43: ("P7", 258, 512, 128, 512),
    36: ("P4", 514, 1024, 32, 128),
    38: ("P10", 130, 256, 128, 512),
    33: ("P9", 514, 1024, 128, 3),
    42: ("P2", 514, 1024, 64, 256),
    44: ("P5", 1026, 2048, 64, 128),
    37: ("P8", 258, 512, 64, 256),
    47: ("P13", 514, 1024, 128, 512),
    34: ("P15", 66, 128, 128, 512),
    46: ("P6", 1026, 2048, 64, 256),
    39: ("P11", 514, 1024, 64, 128),
    45: ("P12", 514, 1024, 128, 256),
    40: ("P14", 258, 512, 128, 256),
    32: ("P16", 130, 256, 32, 128),
    31: ("P17", 66, 128, 32, 128),
}


def load_pd(which):
    fake = types.ModuleType("ttnn")
    fake.get_max_worker_l1_unreserved_size = lambda: UNRESERVED
    fake.MathFidelity = types.SimpleNamespace(LoFi="LoFi", HiFi2="HiFi2", HiFi3="HiFi3", HiFi4="HiFi4")
    fake.ProgramDescriptor = object
    sys.modules["ttnn"] = fake
    spec = importlib.util.spec_from_file_location(f"pd_{which}", HERE / which / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"pd_{which}"] = m
    spec.loader.exec_module(m)
    return m


def geometry(pd, H, W, c_in, c_out):
    kw = dict(
        input_height=H,
        input_width=W,
        c_in_p=-(-c_in // 32) * 32,
        c_out=c_out,
        padding=(0, 0, 1, 1),
        stride=1,
        dilation=1,
        output_height=H - 2,
        output_width=W,
        c_in=c_in,
        tile_input=False,
        in0_tile_bytes=2048,
        weight_tile_bytes=2048,
        out_tile_bytes=2048,
        has_bias=True,
        l1_small_bytes=65536,
    )
    if "bank_run_input" in pd.Conv3x3Geometry.__dataclass_fields__:
        kw["bank_run_input"] = True
    return pd.Conv3x3Geometry(**kw)


def fill_px(pd, g, b):
    """Max over cores of fill pixels: base = every block's in-image bands (merged where they overlap); ring = the core's
    in-image union."""
    worst = 0
    for start, actual in pd.core_m_ranges(g, b.core_m_tiles):
        tot = 0
        segs = []
        for bs in range(0, actual, b.block_m_tiles):
            tiles = min(b.block_m_tiles, actual - bs)
            o0 = (start + bs) * 32
            o1 = min((start + bs + tiles) * 32, g.num_out_px) - 1
            band = pd._block_band_len(g, o0, o1)
            f = pd._block_first_tap(g, o0)
            iv = []
            for kh in range(3):
                v = f + kh * g.input_width
                a, z = max(v, 0), min(v + band, g.num_in_px)
                if a < z:
                    iv.append((a, z))
            segs += iv
            m = []
            for a, z in iv:
                if m and a <= m[-1][1]:
                    m[-1] = (m[-1][0], max(m[-1][1], z))
                else:
                    m.append((a, z))
            tot += sum(z - a for a, z in m)
        if getattr(b, "rm_ring_px", 0):
            segs.sort()
            m = []
            for a, z in segs:
                if m and a <= m[-1][1]:
                    m[-1] = (m[-1][0], max(m[-1][1], z))
                else:
                    m.append((a, z))
            tot = sum(z - a for a, z in m)
        worst = max(worst, tot)
    return worst


def main():
    knobs = dict(a.split("=") for a in sys.argv[1:])
    base, copy = load_pd("op_base"), load_pd("op_copy")
    for k, v in knobs.items():
        setattr(copy, k, type(getattr(copy, k))(eval(v)))
    print(
        "idx  tag   |  base bm bk mb res foot_kB fill_px | ring bm bk mb res ring_px ring_kB stage_kB foot_kB fill_px"
    )
    for idx, (tag, H, W, ci, co) in FOCUS.items():
        row = [f"{idx:3d}  {tag:5s} |"]
        for pd in (base, copy):
            g = geometry(pd, H, W, ci, co)
            b = pd.select_blocking(g, (11, 10), BUDGET)
            pb = g.c_in_p * 2
            row.append(
                f"{b.block_m_tiles:3d} {b.block_k_tiles:2d} {b.max_m_blocks_per_core:3d} {int(b.weights_resident)}"
            )
            if pd is copy:
                rp = getattr(b, "rm_ring_px", 0)
                row.append(f"{rp:6d} {rp * pb / 1024:7.1f} {b.rm_stage_n * b.rm_stage_px * pb / 1024:6.1f}")
            row.append(f"{b.footprint_bytes / 1024:7.1f} {fill_px(pd, g, b):6d} |")
        print(" ".join(row))


if __name__ == "__main__":
    main()
