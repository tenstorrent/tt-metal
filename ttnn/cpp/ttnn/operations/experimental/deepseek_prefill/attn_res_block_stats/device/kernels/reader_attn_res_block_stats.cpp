// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/attn_res_block_stats_items.hpp"

namespace {

// bf16 1.0 into column 0 of a zeroed bf16 tile: the column of ones the squared row is
// multiplied against to land its sum in column 0.
void fill_column_of_ones(uint32_t tile_addr, uint32_t tile_bytes) {
    auto* words = reinterpret_cast<volatile uint32_t*>(tile_addr);
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
        words[i] = 0;
    }
    // Faces 0 and 2 hold column 0, sixteen rows each, sixteen bf16 values to a face row.
    auto* values = reinterpret_cast<volatile uint16_t*>(tile_addr);
    constexpr uint16_t kOne = 0x3F80;
    for (uint32_t face : {0u, 2u}) {
        for (uint32_t row = 0; row < 16; ++row) {
            values[face * 256 + row * 16] = kOne;
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t kSplitArgs = 2;
    constexpr auto sealed_args = TensorAccessorArgs<kSplitArgs + 3>();
    constexpr auto queries_args = TensorAccessorArgs<sealed_args.next_compile_time_args_offset()>();

    // Common args: the queries, then one (address, first page) pair per candidate.
    constexpr uint32_t kCandidateArgs = 1;

    constexpr uint32_t cb_v = 0;
    constexpr uint32_t cb_queries = 1;
    constexpr uint32_t cb_ones = 2;
    constexpr uint32_t tile_bytes = get_tile_size(cb_v);

    Noc noc;
    CircularBuffer cb_v_obj(cb_v);
    CircularBuffer cb_queries_obj(cb_queries);
    CircularBuffer cb_ones_obj(cb_ones);

    // Resident for every item: the queries and the column of ones.
    auto queries_accessor = TensorAccessor(queries_args, get_common_arg_val<uint32_t>(0));
    cb_queries_obj.reserve_back(Wt);
    for (uint32_t k = 0; k < Wt; ++k) {
        noc.async_read(queries_accessor, cb_queries_obj, tile_bytes, {.page_id = k}, {.offset_bytes = k * tile_bytes});
    }
    cb_ones_obj.reserve_back(1);
    fill_column_of_ones(cb_ones_obj.get_write_ptr(), tile_bytes);
    noc.async_read_barrier();
    cb_queries_obj.push_back(Wt);
    cb_ones_obj.push_back(1);

    for_each_attn_res_block_stats_item<kSplitArgs>([&](uint32_t item) {
        const uint32_t candidate = item / Ht;
        const uint32_t row = item % Ht;
        const uint32_t address = get_common_arg_val<uint32_t>(kCandidateArgs + 2 * candidate);
        const uint32_t first_page = get_common_arg_val<uint32_t>(kCandidateArgs + 2 * candidate + 1) + row * Wt;
        auto sealed_accessor = TensorAccessor(sealed_args, address);

        cb_v_obj.reserve_back(Wt);
        for (uint32_t k = 0; k < Wt; ++k) {
            noc.async_read(
                sealed_accessor, cb_v_obj, tile_bytes, {.page_id = first_page + k}, {.offset_bytes = k * tile_bytes});
        }
        noc.async_read_barrier();
        cb_v_obj.push_back(Wt);
    });
}
