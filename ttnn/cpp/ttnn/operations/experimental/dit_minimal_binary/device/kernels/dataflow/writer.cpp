// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    constexpr uint32_t cb_in1_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(1);

    constexpr uint32_t stick_bytes = get_compile_time_arg_val(2);
    constexpr auto in1_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(out_args.next_compile_time_args_offset());

    const uint32_t src1_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);

    const auto dst = TensorAccessor(out_args, dst_addr, stick_bytes);
    const auto src1 = TensorAccessor(in1_args, src1_addr, stick_bytes);

    const uint32_t end_stick = start_stick_id + num_sticks;

    for (uint32_t stick_id = start_stick_id; stick_id < end_stick; stick_id += TILE_HEIGHT) {
        uint32_t nrows = std::min(TILE_HEIGHT, end_stick - stick_id);

        cb_reserve_back(cb_in1_id, ntiles_per_row);

        uint32_t l1_ptr_in1 = get_write_ptr(cb_in1_id);
        for (uint32_t k = 0; k < nrows; ++k) {
            noc_async_read_page(stick_id + k, src1, l1_ptr_in1);
            l1_ptr_in1 += stick_bytes;
        }
        noc_async_read_barrier();

        cb_push_back(cb_in1_id, ntiles_per_row);

        cb_wait_front(cb_out_id, ntiles_per_row);

        uint32_t l1_ptr_out = get_read_ptr(cb_out_id);
        for (uint32_t k = 0; k < nrows; ++k) {
            noc_async_write_page(stick_id + k, dst, l1_ptr_out);
            l1_ptr_out += stick_bytes;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_out_id, ntiles_per_row);
    }
}
