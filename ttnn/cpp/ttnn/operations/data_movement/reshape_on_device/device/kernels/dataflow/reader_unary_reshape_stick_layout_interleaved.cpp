// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);  // num rows or H of input tensor
    uint32_t stick_size = get_arg_val<uint32_t>(2);
    uint32_t num_tiles_c = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr auto tensor_args = TensorAccessorArgs<0>();
    constexpr bool stick_size_is_power_of_two = get_compile_time_arg_val(tensor_args.compile_time_args_skip()) == 1;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    uint32_t stick_id = start_id;

#if (stick_size_is_power_of_two)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(tensor_args.compile_time_args_skip() + 1);
    const auto s = TensorAccessor(tensor_args, src_addr, 1 << log_base_2_of_page_size);
#else
    const auto s = TensorAccessor(tensor_args, src_addr, stick_size);
#endif

    for (uint32_t i = 0; i < num_sticks / 32; i++) {
        // We reserve back an entire tile row and issue a bunch of reads
        cb_reserve_back(cb_id_in0, num_tiles_c);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < 32; j++) {
            uint64_t src_noc_addr = get_noc_addr(stick_id, s);
            noc_async_read(src_noc_addr, l1_write_addr, stick_size);
            l1_write_addr += tile_bytes;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_tiles_c);
    }
}
