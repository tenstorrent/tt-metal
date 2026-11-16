// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// W stream: this core's (tensor_k_tiles x core_n_tiles) slice into a whole-slice resident cb_w_stream.
// Every K-block batch is issued up front under its own read trid; compute is released per K-block.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_w_stream = get_compile_time_arg_val(0);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t w_reads_in_flight = get_compile_time_arg_val(4);
    constexpr auto w_args = TensorAccessorArgs<5>();

    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_start = get_arg_val<uint32_t>(1);
    const uint32_t core_n_tiles = get_arg_val<uint32_t>(2);

    const uint32_t w_page_bytes = get_tile_size(cb_w_stream);
    const auto w = TensorAccessor(w_args, w_addr, w_page_bytes);

    Noc noc;
    CircularBuffer cb_w(cb_w_stream);
    const uint32_t block_tiles = k_block_tiles * core_n_tiles;
    cb_w.reserve_back(num_k_blocks * block_tiles);

    auto trid_for = [](uint32_t b) { return 1 + (b % w_reads_in_flight); };

    // The CB write pointer advances on every push, so a batch lands (b - blocks_pushed) blocks past it.
    auto issue_block = [&](uint32_t b, uint32_t blocks_pushed) {
        dataflow_kernel_lib::set_read_trid(noc, trid_for(b));
        uint32_t offset = (b - blocks_pushed) * block_tiles * w_page_bytes;
        uint32_t row_page = b * k_block_tiles * tensor_n_tiles + n_start;
        for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
            for (uint32_t n = 0; n < core_n_tiles; ++n) {
                noc.async_read(w, cb_w, w_page_bytes, {.page_id = row_page + n}, {.offset_bytes = offset});
                offset += w_page_bytes;
            }
            row_page += tensor_n_tiles;
        }
    };

    for (uint32_t b = 0; b < w_reads_in_flight; ++b) {
        issue_block(b, 0);
    }
    for (uint32_t b = 0; b < num_k_blocks; ++b) {
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid_for(b));
        if (b + w_reads_in_flight < num_k_blocks) {
            issue_block(b + w_reads_in_flight, b);
        }
        cb_w.push_back(block_tiles);
    }
    dataflow_kernel_lib::set_read_trid(noc, 0);
}
