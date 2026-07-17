// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t src0_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    uint32_t src2_addr = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);
    uint32_t start_id = get_arg_val<uint32_t>(6);

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto src2_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;
    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(src0_args, src0_addr);
    const auto s1 = TensorAccessor(src1_args, src1_addr);
    const auto s2 = TensorAccessor(src2_args, src2_addr);

    Noc noc;
    DataflowBuffer dfb_in0(cb_id_in0);
    DataflowBuffer dfb_in1(cb_id_in1);
    DataflowBuffer dfb_in2(cb_id_in2);
    const auto in0_tile_bytes = get_tile_size(cb_id_in0);
    const auto in1_tile_bytes = get_tile_size(cb_id_in1);
    const auto in2_tile_bytes = get_tile_size(cb_id_in2);

    dfb_in0.reserve_back(onetile);
    noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb_in0.push_back(onetile);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        if (has_input_grad) {
            dfb_in2.reserve_back(onetile);
            noc.async_read(s2, dfb_in2, in2_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in2.push_back(onetile);
        }

        if (has_other_grad) {
            dfb_in1.reserve_back(onetile);
            noc.async_read(s1, dfb_in1, in1_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in1.push_back(onetile);
        }
    }
}
