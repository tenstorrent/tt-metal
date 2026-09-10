// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_state_reuse bench — TRIVIAL reader (NCRISC).
//
// Held CONSTANT across every writer variant (perf-lab concept isolation): this
// is a straight per-page `noc_async_read<page_bytes>` loop into `cb_out`
// directly (no compute stage — a pure TILE-to-TILE copy, mirroring the real
// tilize op's `is_retile` path where cb_output_tiles has the reader as its
// sole producer). Its job is only to hand the writer real, correct tile bytes
// so the writer's bit-identity check is meaningful; it is never the thing
// under measurement here.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(2);  // R
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);   // C
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t col_tile_offset = get_compile_time_arg_val(6);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
    constexpr auto in_args = TensorAccessorArgs<8>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    const auto in_acc = TensorAccessor(in_args, src_addr);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t col_base = col_tile_offset + w_chunk * block_width_tiles;

        for (uint32_t r = row_start; r < row_end; ++r) {
            const uint32_t page_base = r * tensor_col_tiles + col_base;
            cb_reserve_back(cb_out, block_width_tiles);
            uint32_t l1_write_addr = get_write_ptr(cb_out);
            {
                MaybeDeviceZoneScope("reader_issue");
#ifdef BENCH_ABLATE_READS
                // Decisive fabric-vs-RISC test only (see bench.py): drop the
                // payload, keep the CB synchronization, so the WRITER can be
                // measured with zero read traffic sharing the NoC/DRAM.
                (void)page_base;
#else
                for (uint32_t i = 0; i < block_width_tiles; ++i) {
                    noc_async_read<page_bytes>(in_acc.get_noc_addr(page_base + i), l1_write_addr, page_bytes);
                    l1_write_addr += page_bytes;
                }
#endif
            }
            {
                MaybeDeviceZoneScope("reader_barrier");
                noc_async_read_barrier();
            }
            cb_push_back(cb_out, block_width_tiles);
        }
    }
}
