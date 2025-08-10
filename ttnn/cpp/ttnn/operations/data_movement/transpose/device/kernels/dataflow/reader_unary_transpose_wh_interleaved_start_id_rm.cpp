// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_hw_blocks_per_core = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t H_per_tile = get_compile_time_arg_val(1);
    constexpr uint32_t H_per_tile_last = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t W = get_compile_time_arg_val(4);
    constexpr uint32_t HtWt = get_compile_time_arg_val(5);
    constexpr uint32_t W_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t l1_write_offset_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t page_size = get_compile_time_arg_val(8);
    constexpr auto src_args = TensorAccessorArgs<9>();

    constexpr auto cb_in0 = tt::CBIndex::c_0;

    const uint32_t stick_size_bytes = W_size_bytes;

    const auto s = TensorAccessor(src_args, src_addr, page_size);

    uint32_t i_stick = start_id;

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            cb_reserve_back(cb_in0, Wt);
            uint32_t l1_write_addr = get_write_ptr(cb_in0);
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                uint64_t read_noc_addr = get_noc_addr(i_stick, s);
                noc_async_read(read_noc_addr, l1_write_addr, stick_size_bytes);
                l1_write_addr += l1_write_offset_bytes;
                i_stick += 1;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in0, Wt);
        }
    }
}
