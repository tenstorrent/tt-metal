// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// linear_gelu writer (BRISC, NoC1): store_output_block. Drains cb_output_tiles per subblock in the
// matmul's subblock-major order (subblock row i, subblock column j, then tiles row-major inside it),
// writing out page m * n_tiles + n. Flush before each pop, one barrier at the end.
// W3 (S % 32 != 0): rows last_valid_rows..31 of every tile of a tail tile row (m % seq_tiles == seq_tiles - 1) are
// zeroed in the CB page before its NoC write, so padding is +0.0 whatever x's padding held (matmul rows are
// independent). Compiled out for aligned S.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_h = get_compile_time_arg_val(1);
    constexpr uint32_t block_w = get_compile_time_arg_val(2);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t last_valid_rows = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_rows = get_arg_val<uint32_t>(2);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(3);
    const uint32_t num_n_blocks = get_arg_val<uint32_t>(4);
    const uint32_t sb_h = get_arg_val<uint32_t>(5);
    const uint32_t sb_w = get_arg_val<uint32_t>(6);
    const uint32_t tail_sb_h = get_arg_val<uint32_t>(7);
    const uint32_t tail_sb_w = get_arg_val<uint32_t>(8);

    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);

    constexpr uint32_t face_bytes = tile_bytes / 4;
    constexpr uint32_t face_row_bytes = face_bytes / 16;
    Noc noc;
    // Zero the padded rows of the tail-row tiles at [l1_addr, l1_addr + count * tile_bytes); one L1 barrier.
    auto zero_padded_rows = [&](uint32_t l1_addr, uint32_t count) {
        for (uint32_t t = 0; t < count; ++t) {
            const uint32_t tile_addr = l1_addr + t * tile_bytes;
            if constexpr (last_valid_rows < 16) {
                constexpr uint32_t row_off = last_valid_rows * face_row_bytes;
                noc.async_write_zeros(CoreLocalMem<uint32_t>(tile_addr + row_off), face_bytes - row_off);
                noc.async_write_zeros(CoreLocalMem<uint32_t>(tile_addr + face_bytes + row_off), face_bytes - row_off);
                noc.async_write_zeros(CoreLocalMem<uint32_t>(tile_addr + 2 * face_bytes), 2 * face_bytes);
            } else {
                constexpr uint32_t row_off = (last_valid_rows - 16) * face_row_bytes;
                noc.async_write_zeros(
                    CoreLocalMem<uint32_t>(tile_addr + 2 * face_bytes + row_off), face_bytes - row_off);
                noc.async_write_zeros(
                    CoreLocalMem<uint32_t>(tile_addr + 3 * face_bytes + row_off), face_bytes - row_off);
            }
        }
        noc.write_zeros_l1_barrier();
    };

    const uint32_t num_row_blocks = (core_rows + block_h - 1) / block_h;
    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        const uint32_t first_row = row_block_idx * block_h;
        const bool tail = core_rows - first_row < block_h;
        const uint32_t rows = tail ? core_rows - first_row : block_h;
        const uint32_t cur_sb_h = tail ? tail_sb_h : sb_h;
        const uint32_t cur_sb_w = tail ? tail_sb_w : sb_w;
        const uint32_t sb_tiles = cur_sb_h * cur_sb_w;
        const uint32_t m_base = row_start + first_row;
        // Occupancy: waits on the packer plus the write issue for one row-block.
        MaybeDeviceZoneScope("writer_drain");

        for (uint32_t n_block_idx = 0; n_block_idx < num_n_blocks; ++n_block_idx) {
            const uint32_t n_base = n_tile_start + n_block_idx * block_w;
            for (uint32_t sb_row = 0; sb_row < rows; sb_row += cur_sb_h) {
                for (uint32_t sb_col = 0; sb_col < block_w; sb_col += cur_sb_w) {
                    cb_wait_front(cb_output_tiles, sb_tiles);
                    uint32_t l1_addr = get_read_ptr(cb_output_tiles);
                    for (uint32_t r = 0; r < cur_sb_h; ++r) {
                        const uint32_t m = m_base + sb_row + r;
                        if constexpr (last_valid_rows != 0) {
                            if (m % seq_tiles == seq_tiles - 1) {
                                zero_padded_rows(l1_addr, cur_sb_w);
                            }
                        }
                        const uint32_t page_row = m * n_tiles + n_base + sb_col;
                        for (uint32_t c = 0; c < cur_sb_w; ++c) {
                            noc_async_write_page(page_row + c, out_accessor, l1_addr);
                            l1_addr += tile_bytes;
                        }
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_output_tiles, sb_tiles);
                }
            }
        }
    }
    {
        MaybeDeviceZoneScope("writer_barrier");
        noc_async_write_barrier();
    }
}
