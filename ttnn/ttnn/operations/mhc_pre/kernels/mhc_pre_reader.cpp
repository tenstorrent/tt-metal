// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre reader (NCRISC, NoC0): the X stream only (W and the bias are the writer's — see mhc_pre_writer.cpp —
// so the X reads start at t = 0 and the W / X DRAM reads ride different NoCs).
//
// load_resident_constants (once): reduce scalers (SUM / REDUCE_ROW, 1.0; fp32 X also MAX / REDUCE_SCALAR).
// load_x_block (per block): block_token_tiles x core_k_tiles X tiles of this rank's stream-column slice,
//   L1 slot (t, c, i) = t*core_k_tiles + c*n + i. One NoC burst per block, published in x_stream_chunks
//   slot-ordered chunks: chunk j's reads carry transaction id j + 1, and chunk j is pushed as soon as its
//   own reads landed (the compute's projection waits per K tile, so it runs under the rest of the burst).
//   The CB is always pushed by the NOMINAL block size (block_token_tiles * core_k_tiles_max; the last chunk
//   carries the padding) so the FIFO never wraps inside a block, whatever this rank's core_k_tiles or the
//   ragged last block's extent.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t n_streams = get_compile_time_arg_val(2);
    constexpr uint32_t block_token_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t core_k_tiles_max = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(6);
    constexpr bool needs_max_scaler = get_compile_time_arg_val(7) != 0;  // fp32 X grid split
    constexpr uint32_t x_stream_chunks = get_compile_time_arg_val(8);    // 1 = one publish per block
    static_assert(x_stream_chunks >= 1 && x_stream_chunks <= 15, "one NoC transaction id per chunk (1..15)");
    constexpr auto x_args = TensorAccessorArgs<9>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t t_start = get_arg_val<uint32_t>(1);
    const uint32_t core_token_tiles = get_arg_val<uint32_t>(2);
    const uint32_t c_start = get_arg_val<uint32_t>(3);
    const uint32_t core_c_tiles = get_arg_val<uint32_t>(4);
    const uint32_t num_blocks = get_arg_val<uint32_t>(5);

    constexpr uint32_t tensor_k_tiles = n_streams * tensor_c_tiles;
    constexpr uint32_t x_block_pages = block_token_tiles * core_k_tiles_max;  // nominal push per block
    const uint32_t core_k_tiles = n_streams * core_c_tiles;
    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);

    auto load_x_block = [&](uint32_t block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        const uint32_t extent =
            (core_token_tiles - row0) < block_token_tiles ? (core_token_tiles - row0) : block_token_tiles;
        const uint32_t real_pages = extent * core_k_tiles;
        const uint32_t chunk_pages = (real_pages + x_stream_chunks - 1) / x_stream_chunks;
        cb_reserve_back(cb_x_resident, x_block_pages);
        const uint32_t base = get_write_ptr(cb_x_resident);
        uint32_t p = 0;  // L1 slot = t*core_k_tiles + c*n + i
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t m = t_start + row0 + t;
            const uint32_t row_page = m * tensor_k_tiles + c_start;
            for (uint32_t c = 0; c < core_c_tiles; ++c) {
                for (uint32_t i = 0; i < n_streams; ++i) {
                    if (p % chunk_pages == 0) {
                        noc_async_read_set_trid(1 + p / chunk_pages);  // sticky for this chunk's reads
                    }
                    noc_async_read_page(row_page + i * tensor_c_tiles + c, x_acc, base + p * x_tile_bytes);
                    ++p;
                }
            }
        }
        noc_async_read_set_trid(0);
        uint32_t pushed = 0;
        for (uint32_t j = 0; j * chunk_pages < real_pages; ++j) {
            {
                DeviceZoneScopedN("R-chunk");
                noc_async_read_barrier_with_trid(1 + j);
            }
            const uint32_t end = (j + 1) * chunk_pages < real_pages ? (j + 1) * chunk_pages : x_block_pages;
            cb_push_back(cb_x_resident, end - pushed);
            pushed = end;
        }
        if (pushed < x_block_pages) {  // empty block (never issued): keep the nominal push count
            cb_push_back(cb_x_resident, x_block_pages - pushed);
        }
    };

    // Scalers first (a local zero fill + a few stores, well under a microsecond), then the X stream.
    dataflow_kernel_lib::
        prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    if constexpr (needs_max_scaler) {
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_SCALAR>(1.0f);
    }
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        DeviceZoneScopedN("R-x");
        load_x_block(block_idx);
    }
}
