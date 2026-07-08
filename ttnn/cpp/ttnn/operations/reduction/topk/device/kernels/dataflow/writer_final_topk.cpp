// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Runtime args
    const uint32_t dst_addr0 = get_arg_val<uint32_t>(0);  // DRAM address for TopK values output tensor
    const uint32_t dst_addr1 = get_arg_val<uint32_t>(1);  // DRAM address for TopK indices output tensor

    // Compile time args
    constexpr uint32_t values_dfb_index = get_compile_time_arg_val(0);      // Final values circular buffer
    constexpr uint32_t output_ind_dfb_index = get_compile_time_arg_val(1);  // Final indices circular buffer
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);

    // DRAM tensor accessor configuration for output writing
    constexpr auto interleaved_accessor0_args = TensorAccessorArgs<4>();
    constexpr auto interleaved_accessor1_args =
        TensorAccessorArgs<interleaved_accessor0_args.next_compile_time_args_offset()>();

    // Memory transfer configuration
    constexpr uint32_t onetile = 1;

    // Initialize DRAM tensor accessors for interleaved output format
    const auto interleaved_accessor0 = TensorAccessor(interleaved_accessor0_args, dst_addr0);
    const auto interleaved_accessor1 = TensorAccessor(interleaved_accessor1_args, dst_addr1);

    Noc noc;
    DataflowBuffer values_dfb(values_dfb_index);
    DataflowBuffer indices_dfb(output_ind_dfb_index);
    const uint32_t tile_bytes_val = values_dfb.get_entry_size();
    const uint32_t tile_bytes_idx = indices_dfb.get_entry_size();

    // Process each height row sequentially, writing Kt tiles of TopK results
    for (uint32_t j = 0; j < Ht; ++j) {
        // Write the final globally optimal TopK values for this height row
        for (uint32_t i = 0; i < Kt; ++i) {
            values_dfb.wait_front(onetile);
            noc.async_write(
                values_dfb, interleaved_accessor0, tile_bytes_val, {.offset_bytes = 0}, {.page_id = j * Kt + i});
            noc.async_write_barrier();
            values_dfb.pop_front(onetile);
        }  // i loop

        // Write the corresponding original indices for the TopK values
        for (uint32_t i = 0; i < Kt; ++i) {
            indices_dfb.wait_front(onetile);
            noc.async_write(
                indices_dfb, interleaved_accessor1, tile_bytes_idx, {.offset_bytes = 0}, {.page_id = j * Kt + i});
            noc.async_write_barrier();
            indices_dfb.pop_front(onetile);
        }  // i loop
    }  // j loop
}
