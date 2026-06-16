// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_dataflow_common.hpp"

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#include <cstdint>

void kernel_main() {
    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);        // DRAM address of input tensor
    const uint32_t start_ht = get_arg_val<uint32_t>(1);        // Starting height tile index
    const uint32_t start_wt = get_arg_val<uint32_t>(2);        // Starting width tile index for this core
    const bool is32_bit_data = get_arg_val<uint32_t>(3) == 1;  // Flag indicating if indices data is 32-bit

    // Compile time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);  // Input values circular buffer
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);  // Generated indices circular buffer
    constexpr uint32_t Ht = get_compile_time_arg_val(2);         // Total height tiles in tensor
    constexpr uint32_t Wt_local = get_compile_time_arg_val(3);   // Width tiles assigned to this core
    constexpr uint32_t Wt = get_compile_time_arg_val(4);         // Total width tiles in tensor

    // Constants
    constexpr uint32_t onetile = 1;

    // DRAM tensor accessor configuration
    constexpr auto s_args = TensorAccessorArgs<5>();
    const auto s = TensorAccessor(s_args, src_addr);

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in1(cb_id_in1);
    const uint32_t tile_bytes_in0 = cb_in0.get_tile_size();

#if not GENERATE_INDICES
    // Precomputed indices tensor accessor
    constexpr auto indices_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const uint32_t src_indices_addr = get_arg_val<uint32_t>(4);
    const auto indices_accessor = TensorAccessor(indices_args, src_indices_addr);
    const uint32_t tile_bytes_in1 = cb_in1.get_tile_size();
#endif  // not GENERATE_INDICES

    for (uint32_t i = start_ht; i < Ht; ++i) {
        for (uint32_t j = start_wt; j < start_wt + Wt_local; ++j) {
            // Stream input value tile from DRAM to local circular buffer
            cb_in0.reserve_back(onetile);
            noc.async_read(s, cb_in0, tile_bytes_in0, {.page_id = i * Wt + j}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in0.push_back(onetile);
#if GENERATE_INDICES
            // Generate corresponding index tile for position tracking during sort
            if (is32_bit_data) {
                generate_index_tile<uint32_t>(cb_id_in1, j);  // Generate indices for width position j
            } else {
                generate_index_tile<uint16_t>(cb_id_in1, j);  // Generate indices for width position j
            }
#else
            // Read precomputed indices to circular buffer
            cb_in1.reserve_back(onetile);
            noc.async_read(indices_accessor, cb_in1, tile_bytes_in1, {.page_id = i * Wt + j}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in1.push_back(onetile);
#endif  // GENERATE_INDICES
        }  // j loop
    }  // i loop
}
