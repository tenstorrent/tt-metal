"""Apply the Perf 18 ablation hooks (CONV2D_3X3_ABLATE=<comma tokens>) to an op copy in place.

Tokens: fill (ROW_MAJOR ring / bank-run DRAM reads and their de-interleave), weights (writer weight rounds), store
(output DRAM writes), tilize (compute tap-direct tilize), bias (the bias pass inside the last K-block), matmul (the
K-block walk's matmul + partials / output packs). Each stub keeps the CB / trid / semaphore sync.
Usage: python apply_hooks.py <op_dir>
"""

import sys
from pathlib import Path

op = Path(sys.argv[1])


def sub(path, old, new, count=1):
    p = op / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:80], s.count(old))
    p.write_text(s.replace(old, new))


D = "conv2d_3x3_program_descriptor.py"
sub(
    D,
    '    readers = [\n        ttnn.KernelDescriptor(\n            kernel_source=str(KERNEL_DIR / "conv2d_3x3_reader.cpp"),',
    '    import os as _os\n\n    _abl = [(f"CONV2D_3X3_ABLATE_{t.upper()}", "1") for t in _os.environ.get("CONV2D_3X3_ABLATE", "").split(",") if t]\n'
    '    readers = [\n        ttnn.KernelDescriptor(\n            kernel_source=str(KERNEL_DIR / "conv2d_3x3_reader.cpp"),',
)
sub(
    D,
    "            + _rm_ring_defines\n            + _rs_defines,\n",
    "            + _rm_ring_defines\n            + _rs_defines\n            + _abl,\n",
)
sub(
    D,
    "            + _host_fill_defines\n            + _wr_defines\n",
    "            + _host_fill_defines\n            + _wr_defines\n            + _abl\n",
)
sub(
    D,
    "        + _wr_defines\n        + _rs_defines,\n",
    "        + _wr_defines\n        + _rs_defines\n        + _abl,\n",
)

# Ring fill: DRAM bank-run reads and the per-pixel de-interleave.
RR = "kernels/conv2d_3x3_rm_ring.hpp"
sub(
    RR,
    "            reads.read(input.get_noc_addr(a + s), base + s * slot_bytes, run * pixel_bytes);\n",
    "#ifndef CONV2D_3X3_ABLATE_FILL\n"
    "            reads.read(input.get_noc_addr(a + s), base + s * slot_bytes, run * pixel_bytes);\n"
    "#endif\n",
)
sub(
    RR,
    "                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, true>(noc, NCRISC_RD_CMD_BUF, src, dst);\n"
    "                reads.count_packets(1);\n",
    "#ifndef CONV2D_3X3_ABLATE_FILL\n"
    "                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, true>(noc, NCRISC_RD_CMD_BUF, src, dst);\n"
    "                reads.count_packets(1);\n"
    "#endif\n",
)

# Per-block bank-run fill (one M-block per core) and the per-pixel fill.
R = "kernels/conv2d_3x3_reader.cpp"
sub(
    R,
    "#elif defined(BANK_RUN_FILL)\n            {\n                // Bank-run fill.",
    "#elif defined(BANK_RUN_FILL)\n            {\n#ifndef CONV2D_3X3_ABLATE_FILL\n                // Bank-run fill.",
)
sub(
    R,
    "                // The staging overwrote any canonical zero rows the block FIFO kept (zero-once): forget them.\n",
    "#endif\n                // The staging overwrote any canonical zero rows the block FIFO kept (zero-once): forget them.\n",
)
sub(
    R,
    "                    reads.begin(kh + 1);\n                    if (x_lo < x_hi) {\n                        // Each core starts its band",
    "                    reads.begin(kh + 1);\n#ifdef CONV2D_3X3_ABLATE_FILL\n                    if (false) {\n#else\n"
    "                    if (x_lo < x_hi) {\n#endif\n                        // Each core starts its band",
)

W = "kernels/conv2d_3x3_writer.cpp"
sub(
    W,
    '        MaybeDeviceZoneScope("writer_weight_fetch");\n        if (server_receiver) {',
    '        MaybeDeviceZoneScope("writer_weight_fetch");\n#ifdef CONV2D_3X3_ABLATE_WEIGHTS\n        if (true) {\n        } else\n#endif\n        if (server_receiver) {',
)
sub(
    W,
    "                        const uint32_t row_page = (m_tile_start + local_tile_row) * tensor_n_tiles + n_tile_start;\n"
    "                        for (uint32_t n = 0; n < block_n_tiles; ++n) {\n"
    "                            noc_async_write_page(row_page + n, output, l1 + n * out_tile_bytes);\n"
    "                        }\n",
    "                        const uint32_t row_page = (m_tile_start + local_tile_row) * tensor_n_tiles + n_tile_start;\n"
    "#ifndef CONV2D_3X3_ABLATE_STORE\n"
    "                        for (uint32_t n = 0; n < block_n_tiles; ++n) {\n"
    "                            noc_async_write_page(row_page + n, output, l1 + n * out_tile_bytes);\n"
    "                        }\n"
    "#endif\n",
)

C = "kernels/conv2d_3x3_compute.cpp"
sub(
    C,
    "        for (uint32_t t = 0; t < taps; ++t) {\n            if constexpr (skip_center) {\n                if (t == 1) {\n                    if constexpr (tw == 1) {",
    "#ifdef CONV2D_3X3_ABLATE_TILIZE\n        out.reserve_back(taps * tw);\n        out.push_back(taps * tw);\n"
    "        for (uint32_t p = 0; p < pages; p += tw) {\n            in.wait_front(tw);\n            in.pop_front(tw);\n        }\n"
    "        table.pop_front(1);\n        continue;\n#endif\n"
    "        for (uint32_t t = 0; t < taps; ++t) {\n            if constexpr (skip_center) {\n                if (t == 1) {\n                    if constexpr (tw == 1) {",
)

K = "kernels/conv2d_3x3_kblock_walk.hpp"
# Bias pass: the row-broadcast adds and their packs (waits / pops / reserves / pushes kept).
sub(
    K,
    "            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {\n"
    "                const uint32_t col = i1 * sb_w;\n"
    "                tile_regs_acquire();\n"
    "                for (uint32_t r = 0; r < sb_h; ++r) {\n",
    "#ifndef CONV2D_3X3_ABLATE_BIAS\n"
    "            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {\n"
    "                const uint32_t col = i1 * sb_w;\n"
    "                tile_regs_acquire();\n"
    "                for (uint32_t r = 0; r < sb_h; ++r) {\n",
)
sub(
    K,
    "                compute_kernel_lib::pack_subblock_row_strided(0, out_cb, col, bn, sb_h, sb_w);\n"
    "                tile_regs_release();\n"
    "            }\n"
    "            cb_pop_front(partials_cb, row_group_tiles);\n",
    "                compute_kernel_lib::pack_subblock_row_strided(0, out_cb, col, bn, sb_h, sb_w);\n"
    "                tile_regs_release();\n"
    "            }\n"
    "#endif\n"
    "            cb_pop_front(partials_cb, row_group_tiles);\n",
)
# Matmul: the subblock loop (reload, matmul, pack) of every K-block.
sub(
    K,
    "            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {\n"
    "                const uint32_t col = i1 * sb_w;\n"
    "                tile_regs_acquire();\n"
    "                if constexpr (reload) {\n",
    "#ifndef CONV2D_3X3_ABLATE_MATMUL\n"
    "            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {\n"
    "                const uint32_t col = i1 * sb_w;\n"
    "                tile_regs_acquire();\n"
    "                if constexpr (reload) {\n",
)
sub(
    K,
    "                        0, partials_cb, i0 * row_group_tiles + col, bn, sb_h, sb_w);\n"
    "                }\n"
    "                tile_regs_release();\n"
    "            }\n",
    "                        0, partials_cb, i0 * row_group_tiles + col, bn, sb_h, sb_w);\n"
    "                }\n"
    "                tile_regs_release();\n"
    "            }\n"
    "#endif\n",
)
print("hooks applied")
