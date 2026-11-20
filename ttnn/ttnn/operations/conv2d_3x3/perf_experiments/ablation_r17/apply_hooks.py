"""Apply the Perf 17 ablation hooks (CONV2D_3X3_ABLATE=<comma tokens>) to the op in place.

Tokens: fill (bank-run / per-pixel ROW_MAJOR fill reads), weights (writer weight rounds), store (output DRAM writes),
tilize (compute tap-direct tilize), bias (bias add), matmul (matmul_block). Each stub keeps the CB / trid sync.
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
    "            + _host_fill_defines\n            + _brf_defines,\n",
    "            + _host_fill_defines\n            + _brf_defines\n            + _abl,\n",
)
sub(
    D,
    "            + _host_fill_defines\n            + _wr_defines\n",
    "            + _host_fill_defines\n            + _wr_defines\n            + _abl\n",
)
sub(
    D,
    "        + _ring_arrival_defines\n        + _wr_defines,\n",
    "        + _ring_arrival_defines\n        + _wr_defines\n        + _abl,\n",
)

R = "kernels/conv2d_3x3_reader.cpp"
sub(
    R,
    "#ifdef BANK_RUN_FILL\n            {\n                // Bank-run fill.",
    "#ifdef BANK_RUN_FILL\n            {\n#ifndef CONV2D_3X3_ABLATE_FILL\n                // Bank-run fill.",
)
sub(
    R,
    "                // The staging overwrote any canonical zero rows the block FIFO kept (zero-once): forget them.\n",
    "#endif\n                // The staging overwrote any canonical zero rows the block FIFO kept (zero-once): forget them.\n",
)
sub(
    R,
    "                    reads.begin(kh + 1);\n                    if (x_lo < x_hi) {\n                        // Each core starts its band",
    "                    reads.begin(kh + 1);\n#ifdef CONV2D_3X3_ABLATE_FILL\n                    if (false) {\n#else\n                    if (x_lo < x_hi) {\n#endif\n                        // Each core starts its band",
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
sub(
    C,
    "            using WrPre = ResidentWeightsPreKBlock<PreKBlockFn, cb_weights, wr_round_tiles>;\n"
    "            const WrPre wr_pre{pre_k_block, &wr_rounds_waited};\n",
    "            using WrPre = ResidentWeightsPreKBlock<PreKBlockFn, cb_weights, wr_round_tiles>;\n"
    "            const WrPre wr_pre{pre_k_block, &wr_rounds_waited};\n"
    "#ifdef CONV2D_3X3_ABLATE_MATMUL\n"
    "            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {\n"
    "                wr_pre(kb, num_k_blocks, kb + 1 == num_k_blocks);\n"
    "                im2col_tiles.wait_front(block_m_tiles * block_k_tiles);\n"
    "                im2col_tiles.pop_front(block_m_tiles * block_k_tiles);\n"
    "            }\n"
    "            if constexpr (false)\n"
    "#endif\n",
)
sub(
    C,
    "#else\n            ckl::matmul_block<\n                /*transpose=*/false,\n                /*packer_l1_acc=*/true,\n                last_block_target,\n",
    "#else\n"
    "#ifdef CONV2D_3X3_ABLATE_MATMUL\n"
    "            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {\n"
    "                pre_k_block(kb, num_k_blocks, kb + 1 == num_k_blocks);\n"
    "                im2col_tiles.wait_front(block_m_tiles * block_k_tiles);\n"
    "                im2col_tiles.pop_front(block_m_tiles * block_k_tiles);\n"
    "                weights.wait_front(block_k_tiles * block_n_tiles);\n"
    "                weights.pop_front(block_k_tiles * block_n_tiles);\n"
    "            }\n"
    "            if constexpr (false)\n"
    "#endif\n"
    "            ckl::matmul_block<\n                /*transpose=*/false,\n                /*packer_l1_acc=*/true,\n                last_block_target,\n",
)
sub(
    C,
    "        if constexpr (has_bias && !k_split && in0_groups == 1 && !slab_direct) {\n            add_bias(bias_shape);\n        }\n",
    "        if constexpr (has_bias && !k_split && in0_groups == 1 && !slab_direct) {\n"
    "#ifdef CONV2D_3X3_ABLATE_BIAS\n"
    "            partials.wait_front(partials_tiles);\n            partials.pop_front(partials_tiles);\n"
    "            for (uint32_t g = 0; g < in0_num_subblocks; ++g) {\n"
    "                output_tiles.reserve_back(out_subblock_h * block_n_tiles);\n"
    "                output_tiles.push_back(out_subblock_h * block_n_tiles);\n            }\n"
    "#else\n            add_bias(bias_shape);\n#endif\n        }\n",
)
print("hooks applied")
