# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (no device, ttnn stubbed): blocking and L1 of the FLUX.2 bh2x4 focus cases for op_base and op_copy
(band_refill_overlap). python3 -I l1_plan.py [op_base|op_copy]"""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
BUDGET = 1334368
UNRESERVED = BUDGET + 128 * 1024 + 65536

FOCUS = [
    (41, 1026, 2048, 32, 128),
    (35, 1026, 2048, 128, 3),
    (43, 258, 512, 128, 512),
    (36, 514, 1024, 32, 128),
    (38, 130, 256, 128, 512),
    (33, 514, 1024, 128, 3),
    (42, 514, 1024, 64, 256),
    (44, 1026, 2048, 64, 128),
    (37, 258, 512, 64, 256),
    (47, 514, 1024, 128, 512),
    (34, 66, 128, 128, 512),
    (46, 1026, 2048, 64, 256),
    (39, 514, 1024, 64, 128),
    (45, 514, 1024, 128, 256),
    (40, 258, 512, 128, 256),
    (32, 130, 256, 32, 128),
    (31, 66, 128, 32, 128),
]


def load_pd(which):
    fake = types.ModuleType("ttnn")
    fake.get_max_worker_l1_unreserved_size = lambda: UNRESERVED
    fake.MathFidelity = types.SimpleNamespace(LoFi="LoFi", HiFi2="HiFi2", HiFi3="HiFi3", HiFi4="HiFi4")
    fake.ProgramDescriptor = object
    sys.modules.setdefault("ttnn", fake)
    spec = importlib.util.spec_from_file_location(f"brf_{which}", HERE / which / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"brf_{which}"] = m
    spec.loader.exec_module(m)
    return m


def geometry(pd, H, W, c_in, c_out):
    extra = {"rm_dram_interleaved": True} if "rm_dram_interleaved" in pd.Conv3x3Geometry.__dataclass_fields__ else {}
    return pd.Conv3x3Geometry(
        **extra,
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


def main(which):
    pd = load_pd(which)
    for idx, H, W, c_in, c_out in FOCUS:
        g = geometry(pd, H, W, c_in, c_out)
        b = pd.select_blocking(g, (11, 10), BUDGET)
        nk = b.num_k_blocks
        min_fp = pd._footprint(
            g, b.core_m_tiles, b.block_m_tiles, b.block_k_tiles, b.block_n_tiles, resident=b.weights_resident
        )["footprint_bytes"]
        stg = pd._brf_staging(g, b.im2col_rm_depth_rows * b.block_k_tiles * 2048)
        extra = {k: getattr(b, k) for k in ("bro_nstg", "bro_chunk_px") if hasattr(b, k)}
        print(
            f"idx{idx} {H}x{W} {c_in}->{c_out}: bm {b.block_m_tiles} bk {b.block_k_tiles} nk {nk} nb "
            f"{b.max_m_blocks_per_core} res {b.weights_resident} band {b.band_len_px} stride {b.band_stride_px} "
            f"depth {b.im2col_rm_depth_rows} foot {b.footprint_bytes / 1024:.0f} KB (min {min_fp / 1024:.0f}) "
            f"left {(BUDGET - b.footprint_bytes) / 1024:.0f} KB (min-left {(BUDGET - min_fp) / 1024:.0f}) "
            f"stage(alias) {stg} {extra}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "op_base")
