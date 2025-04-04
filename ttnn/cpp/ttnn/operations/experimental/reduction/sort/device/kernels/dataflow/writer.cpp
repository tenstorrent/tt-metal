// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "Starting WRITER kernel" << ENDL();

    // Runtime args
    const uint32_t value_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);

    // Compile time args
    // TODO: More arguments
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(0);
    constexpr bool value_tensor_is_dram = get_compile_time_arg_val(1);
    constexpr bool index_tensor_is_dram = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);

    // Output tensor config
    constexpr uint32_t one_tile = 1;
    // TODO: Value tensor output

    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_output_cb_index);
    const DataFormat index_tensor_output_data_format = get_dataformat(index_tensor_output_cb_index);
    const InterleavedAddrGenFast<index_tensor_is_dram> interleaved_accessor1 = {
        .bank_base_address = index_tensor_buffer_addr,
        .page_size = index_tensor_output_tile_size_bytes,
        .data_format = index_tensor_output_data_format};

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            // Value tensor
            // TODO: value tensor handling

            // Index tensor
            cb_wait_front(tt::CBIndex::c_1, one_tile);
            uint32_t l1_read_addr = get_read_ptr(tt::CBIndex::c_1);
            uint16_t* l1_read_addr_val = (uint16_t*)l1_read_addr;
            uint32_t pointer_addr = (uint32_t)l1_read_addr_val;
            DPRINT << "Got: " << U32(pointer_addr) << "Value: " << U32(l1_read_addr_val[0]) << ENDL();
            noc_async_write_tile(h * Wt + w, interleaved_accessor1, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(tt::CBIndex::c_1, one_tile);
        }
    }  // Ht loop
}
