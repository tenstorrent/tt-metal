// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto a_args = TensorAccessorArgs<1>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(b_args.next_compile_time_args_offset());
    constexpr uint32_t tile_width_bytes = get_compile_time_arg_val(b_args.next_compile_time_args_offset() + 1);

    const uint32_t src_a_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_b_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);      // number of sticks ~ number of rows
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);  // first row for this core

    const auto a = TensorAccessor(a_args, src_a_addr, stick_size);
    const auto b = TensorAccessor(b_args, src_b_addr, stick_size);

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;

    uint32_t end_stick = start_stick_id + num_sticks;

    for (uint32_t stick_id = start_stick_id; stick_id < end_stick; stick_id += TILE_HEIGHT) {
        uint32_t nrows = std::min(TILE_HEIGHT, end_stick - stick_id);

        cb_reserve_back(cb_a, ntiles_per_row);
        uint32_t l1_a = get_write_ptr(cb_a);
        for (uint32_t k = 0; k < nrows; ++k) {
            noc_async_read_page(stick_id + k, a, l1_a);
            l1_a += stick_size;  // each page/stick is stick_size bytes (= ntiles_per_row * tile_width_bytes)
        }
        noc_async_read_barrier();

        cb_push_back(cb_a, ntiles_per_row);
    }
}
