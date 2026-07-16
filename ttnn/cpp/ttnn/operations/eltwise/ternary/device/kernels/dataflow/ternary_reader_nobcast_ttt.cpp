// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(2);

    constexpr auto src0_args = TensorAccessorArgs<3, 0>();
    constexpr auto src1_args =
        TensorAccessorArgs<src0_args.next_compile_time_args_offset(), src0_args.next_common_runtime_args_offset()>();
    constexpr auto src2_args =
        TensorAccessorArgs<src1_args.next_compile_time_args_offset(), src1_args.next_common_runtime_args_offset()>();

    Noc noc;
    DataflowBuffer dfb0(cb_id_in0);
    DataflowBuffer dfb1(cb_id_in1);
    DataflowBuffer dfb2(cb_id_in2);

    const uint32_t tile_bytes_0 = dfb0.get_entry_size();
    const uint32_t tile_bytes_1 = dfb1.get_entry_size();
    const uint32_t tile_bytes_2 = dfb2.get_entry_size();
    const auto s0 = TensorAccessor(src0_args, src0_addr);
    const auto s1 = TensorAccessor(src1_args, src1_addr);
    const auto s2 = TensorAccessor(src2_args, src2_addr);
    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
        dfb0.reserve_back(onetile);
        noc.async_read(s0, dfb0, tile_bytes_0, {.page_id = tile_id}, {.offset_bytes = 0});

        dfb1.reserve_back(onetile);
        noc.async_read(s1, dfb1, tile_bytes_1, {.page_id = tile_id}, {.offset_bytes = 0});

        dfb2.reserve_back(onetile);
        noc.async_read(s2, dfb2, tile_bytes_2, {.page_id = tile_id}, {.offset_bytes = 0});

        noc.async_read_barrier();

        dfb0.push_back(onetile);
        dfb1.push_back(onetile);
        dfb2.push_back(onetile);
    }
}
