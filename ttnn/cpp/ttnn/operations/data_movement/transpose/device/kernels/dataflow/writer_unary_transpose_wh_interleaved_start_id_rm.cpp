// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_hw_blocks_per_core = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();
    constexpr uint32_t Ht = get_compile_time_arg_val(1 + tensor_args.compile_time_args_skip());
    constexpr uint32_t H = get_compile_time_arg_val(2 + tensor_args.compile_time_args_skip());
    constexpr uint32_t Wt = get_compile_time_arg_val(3 + tensor_args.compile_time_args_skip());
    constexpr uint32_t W_per_tile = get_compile_time_arg_val(4 + tensor_args.compile_time_args_skip());
    constexpr uint32_t W_per_tile_last = get_compile_time_arg_val(5 + tensor_args.compile_time_args_skip());
    constexpr uint32_t HtWt = get_compile_time_arg_val(6 + tensor_args.compile_time_args_skip());
    constexpr uint32_t H_size_bytes = get_compile_time_arg_val(7 + tensor_args.compile_time_args_skip());
    constexpr uint32_t l1_read_offset_bytes = get_compile_time_arg_val(8 + tensor_args.compile_time_args_skip());

    // single-tile ublocks
    const uint32_t stick_size_bytes = H_size_bytes;

    constexpr bool stick_size_is_pow2 = get_compile_time_arg_val(9 + tensor_args.compile_time_args_skip()) == 1;
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(10 + tensor_args.compile_time_args_skip());
    const auto s = TensorAccessor(tensor_args, dst_addr, log_base_2_of_page_size, true);
#else
    constexpr uint32_t page_size = get_compile_time_arg_val(10 + tensor_args.compile_time_args_skip());
    const auto s = TensorAccessor(tensor_args, dst_addr, page_size);
#endif

    uint32_t i_stick = start_id;

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t w = 0; w < Wt; ++w) {
            cb_wait_front(cb_out0, Ht);
            uint32_t l1_read_addr = get_read_ptr(cb_out0);
            uint32_t W_curr = w == Wt - 1 ? W_per_tile_last : W_per_tile;
            for (uint32_t w_datum = 0; w_datum < W_curr; ++w_datum) {
                uint64_t write_noc_addr = get_noc_addr(i_stick, s);
                noc_async_write(l1_read_addr, write_noc_addr, stick_size_bytes);
                l1_read_addr += l1_read_offset_bytes;
                i_stick += 1;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out0, Ht);
        }
    }
}
