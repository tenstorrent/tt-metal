# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Builds e2e_kernels/<variant>/: a copy of the op's kernels/ whose per-row issue (store_block of
the staged writer, load_tile_row of the row reader) goes through kernels/row_issue.hpp.

    python3 make_e2e_kernels.py            # base (plain copy), cursor, state, state_bulk

The op's own files are only read. Re-run after the op's kernels change.
"""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP_KERNELS = HERE.parents[1] / "kernels"
OUT = HERE / "e2e_kernels"

VARIANTS = {
    "cursor": "ISSUE_CURSOR",
    "state": "ISSUE_STATE",
    "state_bulk": "ISSUE_STATE_BULK",
    "run_major": "ISSUE_STATE_BULK",  # + R1 full-row blocks staged run-major, one write per bank phase
}

# R1 (staged_reorder) only: the reorder places row r of a full-row block's tile-row at run slot
# run_pos(r) instead of r; the writer (tiled_stage == false) writes such tile-rows per bank phase.
REORDER_OLD = """        uint32_t dst = stage_base + i * Geometry::stage_tile_row_bytes + j * TILE_ROW_BYTES;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t half_base = half * FACE_HW;
            if (live_rows <= half_base) {
                break;
            }
            const uint32_t rows = min_u32(live_rows - half_base, FACE_HW);
            uint32_t src = tile + half * 2 * FACE_BYTES;
            for (uint32_t r = 0; r < rows; ++r) {
                dkl::read_with_state<FACE_ROW_BYTES>(noc, dst, src);"""
REORDER_NEW = """        const uint32_t row_base = stage_base + i * Geometry::stage_tile_row_bytes + j * TILE_ROW_BYTES;
        const bool run_major = Geometry::full_row_blocks && batch.block.col == 0 &&
                               batch.block.cols == Geometry::tensor_cols_tiles;
        uint32_t dst = row_base;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t half_base = half * FACE_HW;
            if (live_rows <= half_base) {
                break;
            }
            const uint32_t rows = min_u32(live_rows - half_base, FACE_HW);
            uint32_t src = tile + half * 2 * FACE_BYTES;
            for (uint32_t r = 0; r < rows; ++r) {
                if (run_major) {
                    dst = row_base + row_issue::run_pos(half_base + r, live_rows) * Geometry::stage_row_bytes;
                }
                dkl::read_with_state<FACE_ROW_BYTES>(noc, dst, src);"""

WRITER_RUN_OLD = """            for (uint32_t i = 0; i < block.rows; ++i) {
                const uint32_t g = block.tile_row + i;
                row_issue::write_rows<row_issue::G6_ROW_ISSUE>("""
WRITER_RUN_NEW = """            const bool run_major = !tiled_stage && Geometry::full_row_blocks && block.col == 0 &&
                                   block.cols == Geometry::tensor_cols_tiles;
            for (uint32_t i = 0; i < block.rows; ++i) {
                const uint32_t g = block.tile_row + i;
                if (run_major) {
                    row_issue::write_run_major_rows(
                        output, first_row_page_of(g), live_rows_of(g), segment_bytes, tile_row_src,
                        Geometry::stage_row_bytes);
                    tile_row_src += Geometry::stage_tile_row_bytes;
                    continue;
                }
                row_issue::write_rows<row_issue::G6_ROW_ISSUE>("""

WRITER_OLD = """            for (uint32_t i = 0; i < block.rows; ++i) {
                const uint32_t g = block.tile_row + i;
                const uint32_t live_rows = live_rows_of(g);
                const uint32_t first_page = first_row_page_of(g);
                uint32_t src = tile_row_src;
                for (uint32_t r = 0; r < live_rows; ++r) {
                    noc_async_write(src, output.get_noc_addr(first_page + r, segment_offset), segment_bytes);
                    src += Geometry::stage_row_bytes;
                }
                tile_row_src += Geometry::stage_tile_row_bytes;
            }"""
WRITER_NEW = """            for (uint32_t i = 0; i < block.rows; ++i) {
                const uint32_t g = block.tile_row + i;
                row_issue::write_rows<row_issue::G6_ROW_ISSUE>(
                    output,
                    first_row_page_of(g),
                    live_rows_of(g),
                    segment_offset,
                    segment_bytes,
                    tile_row_src,
                    Geometry::stage_row_bytes);
                tile_row_src += Geometry::stage_tile_row_bytes;
            }"""

READER_OLD = """    for (uint32_t r = 0; r < live_rows; ++r) {
        noc_async_read(input.get_noc_addr(first_page + r, segment_offset), stage, segment_bytes);
        stage += Geometry::stage_row_bytes;
    }"""
READER_NEW = """    row_issue::read_rows<row_issue::G6_ROW_ISSUE>(
        input, first_page, live_rows, segment_offset, segment_bytes, stage, Geometry::stage_row_bytes);"""

INCLUDE_ANCHOR = '#include "g6_to_layout_block_walk.hpp"\n'


def patch(text, old, new, include, variant_enum):
    assert old in text, "op kernel changed: patch anchor not found"
    text = text.replace(old, new)
    decl = f"namespace row_issue {{\nconstexpr uint32_t G6_ROW_ISSUE = {variant_enum};\n}}\n"
    return text.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + include + decl, 1)


def main(names):
    for name in names:
        dst = OUT / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(OP_KERNELS, dst)
        if name == "base":
            continue
        shutil.copy(HERE / "kernels" / "row_issue.hpp", dst / "row_issue.hpp")
        inc = '#include "row_issue.hpp"\n'
        w = dst / "g6_to_layout_staged_writer.cpp"
        w.write_text(patch(w.read_text(), WRITER_OLD, WRITER_NEW, inc, f"row_issue::{VARIANTS[name]}"))
        r = dst / "g6_to_layout_row_reader.cpp"
        r.write_text(patch(r.read_text(), READER_OLD, READER_NEW, inc, f"row_issue::{VARIANTS[name]}"))
        if name == "run_major":
            s = dst / "g6_to_layout_staged_reader.cpp"
            s.write_text(patch(s.read_text(), REORDER_OLD, REORDER_NEW, inc, f"row_issue::{VARIANTS[name]}"))
            assert s.read_text().count("dst += Geometry::stage_row_bytes;") == 1
            t = s.read_text().replace(
                "                dst += Geometry::stage_row_bytes;",
                "                if (!run_major) {\n                    dst += Geometry::stage_row_bytes;\n                }",
            )
            s.write_text(t)
            w.write_text(w.read_text().replace(WRITER_RUN_OLD, WRITER_RUN_NEW))
            assert "write_run_major_rows" in w.read_text()
        print("built", dst)


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", *VARIANTS])
