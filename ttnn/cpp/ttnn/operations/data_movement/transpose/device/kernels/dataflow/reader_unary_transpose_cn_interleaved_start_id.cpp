// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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

    const auto s = TensorAccessor(src_args, src_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t page_idx = start_id;
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb.reserve_back(onepage);
#ifdef CN_RM
        noc.async_read(s, cb, read_size, {.page_id = page_idx, .offset_bytes = 0}, {.offset_bytes = 0});
#else
        noc.async_read(s, cb, page_size, {.page_id = page_idx}, {.offset_bytes = 0});
#endif
        noc.async_read_barrier();
        cb.push_back(onepage);
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
