// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_hw_blocks_per_core = get_arg_val<uint32_t>(2);

    constexpr uint32_t dfb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t W_per_tile = get_compile_time_arg_val(4);
    constexpr uint32_t W_per_tile_last = get_compile_time_arg_val(5);
    constexpr uint32_t HtWt = get_compile_time_arg_val(6);
    constexpr uint32_t H_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t l1_read_offset_bytes = get_compile_time_arg_val(8);

    // single-tile ublocks
    const uint32_t stick_size_bytes = H_size_bytes;

    constexpr auto dst_args = TensorAccessorArgs<10>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb(dfb_out0);

    uint32_t i_stick = start_id;

    // Uses tt::data_movement::common::noc_async_write_sharded for BLOCK/WIDTH-sharded
    // RM outputs (see reader kernel comment). PR #42130 dropped this on the writer side
    // too, silently regressing the same 24+ test cases for output sharding.
    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t w = 0; w < Wt; ++w) {
            dfb.wait_front(Ht);
            const uint32_t cb_read_ptr = dfb.get_read_ptr();
            uint32_t l1_read_offset = 0;
            uint32_t W_curr = w == Wt - 1 ? W_per_tile_last : W_per_tile;
            for (uint32_t w_datum = 0; w_datum < W_curr; ++w_datum) {
                tt::data_movement::common::noc_async_write_sharded(
                    noc, cb_read_ptr + l1_read_offset, s, i_stick, 0, stick_size_bytes);
                l1_read_offset += l1_read_offset_bytes;
                i_stick += 1;
            }
            noc.async_write_barrier();
            dfb.pop_front(Ht);
        }
    }
}
