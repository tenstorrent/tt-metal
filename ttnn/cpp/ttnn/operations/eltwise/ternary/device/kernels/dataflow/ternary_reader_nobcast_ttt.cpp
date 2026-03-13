// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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
    const uint32_t tile_bytes_0 = get_tile_size(cb_id_in0);
    const uint32_t tile_bytes_1 = get_tile_size(cb_id_in1);
    const uint32_t tile_bytes_2 = get_tile_size(cb_id_in2);
    const auto s0 = TensorAccessor(src0_args, src0_addr, tile_bytes_0);
    const auto s1 = TensorAccessor(src1_args, src1_addr, tile_bytes_1);
    const auto s2 = TensorAccessor(src2_args, src2_addr, tile_bytes_2);
    constexpr uint32_t onetile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_id_in0);
    experimental::CircularBuffer cb1(cb_id_in1);
    experimental::CircularBuffer cb2(cb_id_in2);

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
        cb0.reserve_back(onetile);
        noc.async_read(s0, cb0, tile_bytes_0, {.page_id = tile_id}, {.offset_bytes = 0});

        cb1.reserve_back(onetile);
        noc.async_read(s1, cb1, tile_bytes_1, {.page_id = tile_id}, {.offset_bytes = 0});

        cb2.reserve_back(onetile);
        noc.async_read(s2, cb2, tile_bytes_2, {.page_id = tile_id}, {.offset_bytes = 0});

        noc.async_read_barrier();

        cb0.push_back(onetile);
        cb1.push_back(onetile);
        cb2.push_back(onetile);
    }
}
