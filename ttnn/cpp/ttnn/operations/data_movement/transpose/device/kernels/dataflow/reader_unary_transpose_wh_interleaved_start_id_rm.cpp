// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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
    constexpr auto src_args = TensorAccessorArgs<9>();

    constexpr auto cb_in0 = tt::CBIndex::c_0;

    const uint32_t stick_size_bytes = W_size_bytes;

    const auto s = TensorAccessor(src_args, src_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_in0);

    uint32_t i_stick = start_id;

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            cb.reserve_back(Wt);
            uint32_t l1_write_offset = 0;
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                noc.async_read(
                    s,
                    cb,
                    stick_size_bytes,
                    {.page_id = i_stick, .offset_bytes = 0},
                    {.offset_bytes = l1_write_offset});
                l1_write_offset += l1_write_offset_bytes;
                i_stick += 1;
            }
            noc.async_read_barrier();
            cb.push_back(Wt);
        }
    }
}
