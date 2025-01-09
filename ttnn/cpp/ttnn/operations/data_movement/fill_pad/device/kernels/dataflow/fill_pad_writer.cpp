// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // get input tensor DRAM and find starting points for pad iteration
    const std::uint32_t tensor_dram_buffer_src_addr = get_arg_val<uint32_t>(0);
    const std::uint32_t beginning_row = get_arg_val<uint32_t>(1);
    const std::uint32_t beginning_col = get_arg_val<uint32_t>(2);

    // hardware constraints
    constexpr uint32_t face_size = 16;
    constexpr uint32_t tile_height = 32;

    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr bool tensor_in_dram = get_compile_time_arg_val(1) == 1;
    const std::uint32_t fill_value = get_compile_time_arg_val(4);

    const auto tensor = get_interleaved_addr_gen<tensor_in_dram, 16>(tensor_buffer_src_addr);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t last_row = get_arg_val<uint32_t>(2);
    uint32_t last_col = get_arg_val<uint32_t>(3);
    uint32_t last_tile_row = get_arg_val<uint32_t>(4);
    uint32_t last_tile_col = get_arg_val<uint32_t>(5);

#define dst_stick_size_is_pow2 get_compile_time_arg_val(2) == 1
#if (dst_stick_size_is_pow2)
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);
    const InterleavedPow2AddrGen<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = dst_log_base_2_of_page_size  // TODO(AP): refactor
    };
#else
    const InterleavedAddrGen<tensor_in_dram> s0 = {.bank_base_address = dst_addr, .page_size = stick_size};
#endif

    // Reserve and push the fill value into the circular buffer
    cb_reserve_back(cb_id_0, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_id_0);
    volatile tt_l1_ptr uint32_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    *l1_ptr = fill_value;
    cb_push_back(cb_id_0, 1);  // Push the fill value to the circular buffer once

    uint32_t start_col;
    for (uint32_t row = 0; row < last_tile_row; row++) {
        if (row < last_row) {
            start_col = last_col;
        } else {
            start_col = 0;
        }
        for (uint32_t col = start_col; col < last_tile_col; col++) {
            uint64_t dst_noc_addr = get_noc_addr(row, col, s0);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            noc_async_write_barrier();
        }
    }
}
