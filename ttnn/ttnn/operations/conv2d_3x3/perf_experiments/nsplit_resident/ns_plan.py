# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (ttnn stubbed, no device): the blocking the N-split solve picks per FLUX.2 bh2x4 focus case, against the
base (op_base) blocking, plus the per-core work (computed tile-MACs incl. the ragged tail, input rows filled).

    python3 -I ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nsplit_resident/ns_plan.py

Budget = l1_cb_budget_bytes(0, 65536) on the p150b (1,334,368 B, from the round-16 rm_ring_slab plan)."""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
BUDGET = 1334368
UNRESERVED = BUDGET + 128 * 1024 + 65536
GRID = (11, 10)

# (idx, tag, H, W, C_in, C_out)
FOCUS = [
    (43, "P7", 258, 512, 128, 512),
    (38, "P10", 130, 256, 128, 512),
    (47, "P13", 514, 1024, 128, 512),
    (34, "P15", 66, 128, 128, 512),
    (41, "P1", 1026, 2048, 32, 128),
    (35, "P3", 1026, 2048, 128, 3),
    (36, "P4", 514, 1024, 32, 128),
    (33, "P9", 514, 1024, 128, 3),
    (42, "P2", 514, 1024, 64, 256),
    (44, "P5", 1026, 2048, 64, 128),
    (37, "P8", 258, 512, 64, 256),
    (46, "P6", 1026, 2048, 64, 256),
    (39, "P11", 514, 1024, 64, 128),
    (45, "P12", 514, 1024, 128, 256),
    (40, "P14", 258, 512, 128, 256),
    (32, "P16", 130, 256, 32, 128),
    (31, "P17", 66, 128, 32, 128),
]


def load(name):
    fake = types.ModuleType("ttnn")
    fake.get_max_worker_l1_unreserved_size = lambda: UNRESERVED
    fake.MathFidelity = types.SimpleNamespace(LoFi="LoFi", HiFi2="HiFi2", HiFi3="HiFi3", HiFi4="HiFi4")
    fake.ProgramDescriptor = object
    sys.modules.setdefault("ttnn", fake)
    spec = importlib.util.spec_from_file_location(f"nsp_{name}", HERE / name / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"nsp_{name}"] = m
    spec.loader.exec_module(m)
    return m


def geometry(pd, H, W, c_in, c_out):
    return pd.Conv3x3Geometry(
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


def describe(g, b):
    ng = getattr(b, "n_groups", 1)
    cores = b.num_m_groups * ng
    computed = b.max_m_blocks_per_core * b.block_m_tiles
    macs = computed * b.block_n_tiles * g.tensor_k_tiles
    return (
        f"ng {ng} cores {cores} core_m {b.core_m_tiles} bm {b.block_m_tiles} bk {b.block_k_tiles} bn {b.block_n_tiles} "
        f"nb {b.max_m_blocks_per_core} res {int(b.weights_resident)} computed {computed}/{b.core_m_tiles} "
        f"MACs/core {macs} foot {b.footprint_bytes / 1024:.0f} KB"
    )


def main():
    base, copy = load("op_base"), load("op_copy")
    for idx, tag, H, W, c_in, c_out in FOCUS:
        gb = geometry(base, H, W, c_in, c_out)
        bb = base.select_blocking(gb, GRID, BUDGET)
        gc = geometry(copy, H, W, c_in, c_out)
        rows = [f"idx{idx} {tag} {H}x{W} {c_in}->{c_out}", f"  base: {describe(gb, bb)}"]
        copy.NSPLIT_FORCE = None
        copy.select_blocking.cache_clear()
        bc = copy.select_blocking(gc, GRID, BUDGET)
        rows.append(f"  auto: {describe(gc, bc)}")
        for ng in (2, 4):
            copy.NSPLIT_FORCE = ng
            copy.select_blocking.cache_clear()
            try:
                bf = copy.select_blocking(gc, GRID, BUDGET)
                rows.append(f"  ng={ng}: {describe(gc, bf)}")
            except ValueError as e:
                rows.append(f"  ng={ng}: {e}")
        copy.NSPLIT_FORCE = None
        copy.select_blocking.cache_clear()
        print("\n".join(rows))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def flux_cases():
    """The FLUX compare harness's case list (index = FLUX_CMP_CASES idx), built from its source without importing it."""
    import itertools

    src = (
        HERE.parents[5] / "tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py"
    ).read_text()
    body = src[src.index("def _decoder_convs") : src.index("CASES = _build_cases()")]
    ns = {"itertools": itertools, "IMAGE_SIZES": (1024, 2048)}
    exec(body, ns)
    return ns["_build_cases"]()


def reach_report():
    """Every FLUX compare case (1x1 on TILE input, bh2x4 on ROW_MAJOR and on TILE): does the N split engage?"""
    base, copy = load("op_base"), load("op_copy")
    for idx, c in enumerate(flux_cases()):
        layouts = (True,) if c["group"] == "1x1" else (False, True)
        for tile in layouts:
            if c["c_in"] < 8:
                tile = False
            H, W, ci, co, st, pad = c["H"], c["W"], c["c_in"], c["c_out"], c["stride"], c["padding"]
            ho = (H + pad[0] + pad[1] - 3) // st + 1
            wo = (W + pad[2] + pad[3] - 3) // st + 1
            out = []
            for pd in (base, copy):
                g = pd.Conv3x3Geometry(
                    input_height=H,
                    input_width=W,
                    c_in_p=-(-ci // 32) * 32 if tile or ci % 32 == 0 else -(-ci // 8) * 8,
                    c_out=co,
                    padding=pad if len(set(pad)) > 1 else pad[0],
                    stride=st,
                    dilation=1,
                    output_height=ho,
                    output_width=wo,
                    c_in=ci,
                    tile_input=tile,
                    in0_tile_bytes=2048,
                    weight_tile_bytes=2048,
                    out_tile_bytes=2048,
                    has_bias=True,
                    l1_small_bytes=65536,
                )
                try:
                    out.append(describe(g, pd.select_blocking(g, GRID, BUDGET)))
                except ValueError as e:
                    out.append(str(e)[:60])
            tag = "SPLIT" if "ng 1 " not in out[1] else "same "
            print(
                f"{tag} idx{idx} {c['group']} {'TILE' if tile else 'RM  '} {H}x{W} s{st} {ci}->{co}: base [{out[0]}] copy [{out[1]}]"
            )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "reach":
    reach_report()
