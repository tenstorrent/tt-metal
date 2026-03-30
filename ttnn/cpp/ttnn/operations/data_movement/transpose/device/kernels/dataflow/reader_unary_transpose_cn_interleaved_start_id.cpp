// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t N = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);
    const uint32_t HtWt = get_arg_val<uint32_t>(3);
    const uint32_t batch_step = get_arg_val<uint32_t>(4);    // CHtWt - HtWt
    const uint32_t channel_step = get_arg_val<uint32_t>(5);  // NCHtWt - HtWt
    const uint32_t num_pages = get_arg_val<uint32_t>(6);
    const uint32_t start_id = get_arg_val<uint32_t>(7);
    uint32_t hw = get_arg_val<uint32_t>(8);
    uint32_t n = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t read_size = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    // ublocks size defined in tiles
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(src_args, src_addr, page_size);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t page_idx = start_id;
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id_in0, onepage);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
#ifdef CN_RM
        tt::data_movement::common::noc_async_read_sharded(l1_write_addr, s, page_idx, 0, read_size);
#else
        noc_async_read_page(page_idx, s, l1_write_addr);
#endif
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onepage);
        page_idx++;
        hw++;
        if (hw == HtWt) {
            hw = 0;
            n++;
            page_idx += batch_step;
            if (n == N) {
                n = 0;
                page_idx -= channel_step;
            }
        }
    }
}
