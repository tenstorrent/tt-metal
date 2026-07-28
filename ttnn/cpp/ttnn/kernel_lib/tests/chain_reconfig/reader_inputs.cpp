// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams the same tile range from one to four tensors into CBs c_0..c_3.
// The input count is a compile-time argument; TensorAccessorArgs follow it.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

template <uint32_t NumInputs>
void read_inputs() {
    static_assert(NumInputs >= 1 && NumInputs <= 4);
    constexpr uint32_t one_tile = 1;
    Noc noc;

    if constexpr (NumInputs == 1) {
        const uint32_t src0_addr = get_arg_val<uint32_t>(0);
        const uint32_t num_tiles = get_arg_val<uint32_t>(1);
        const uint32_t start_id = get_arg_val<uint32_t>(2);
        constexpr auto src0_args = TensorAccessorArgs<1>();
        const auto src0 = TensorAccessor(src0_args, src0_addr);
        const uint32_t bytes0 = get_local_cb_interface(0).fifo_page_size;
        CircularBuffer cb0(0);
        for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
            cb0.reserve_back(one_tile);
            noc.async_read(src0, cb0, bytes0, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb0.push_back(one_tile);
        }
    } else if constexpr (NumInputs == 2) {
        const uint32_t src0_addr = get_arg_val<uint32_t>(0);
        const uint32_t src1_addr = get_arg_val<uint32_t>(1);
        const uint32_t num_tiles = get_arg_val<uint32_t>(2);
        const uint32_t start_id = get_arg_val<uint32_t>(3);
        constexpr auto src0_args = TensorAccessorArgs<1>();
        constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
        const auto src0 = TensorAccessor(src0_args, src0_addr);
        const auto src1 = TensorAccessor(src1_args, src1_addr);
        const uint32_t bytes0 = get_local_cb_interface(0).fifo_page_size;
        const uint32_t bytes1 = get_local_cb_interface(1).fifo_page_size;
        CircularBuffer cb0(0), cb1(1);
        for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
            cb0.reserve_back(one_tile);
            cb1.reserve_back(one_tile);
            noc.async_read(src0, cb0, bytes0, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read(src1, cb1, bytes1, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb0.push_back(one_tile);
            cb1.push_back(one_tile);
        }
    } else if constexpr (NumInputs == 3) {
        const uint32_t src0_addr = get_arg_val<uint32_t>(0);
        const uint32_t src1_addr = get_arg_val<uint32_t>(1);
        const uint32_t src2_addr = get_arg_val<uint32_t>(2);
        const uint32_t num_tiles = get_arg_val<uint32_t>(3);
        const uint32_t start_id = get_arg_val<uint32_t>(4);
        constexpr auto src0_args = TensorAccessorArgs<1>();
        constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
        constexpr auto src2_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
        const auto src0 = TensorAccessor(src0_args, src0_addr);
        const auto src1 = TensorAccessor(src1_args, src1_addr);
        const auto src2 = TensorAccessor(src2_args, src2_addr);
        const uint32_t bytes0 = get_local_cb_interface(0).fifo_page_size;
        const uint32_t bytes1 = get_local_cb_interface(1).fifo_page_size;
        const uint32_t bytes2 = get_local_cb_interface(2).fifo_page_size;
        CircularBuffer cb0(0), cb1(1), cb2(2);
        for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
            cb0.reserve_back(one_tile);
            cb1.reserve_back(one_tile);
            cb2.reserve_back(one_tile);
            noc.async_read(src0, cb0, bytes0, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read(src1, cb1, bytes1, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read(src2, cb2, bytes2, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb0.push_back(one_tile);
            cb1.push_back(one_tile);
            cb2.push_back(one_tile);
        }
    } else {
        const uint32_t src0_addr = get_arg_val<uint32_t>(0);
        const uint32_t src1_addr = get_arg_val<uint32_t>(1);
        const uint32_t src2_addr = get_arg_val<uint32_t>(2);
        const uint32_t src3_addr = get_arg_val<uint32_t>(3);
        const uint32_t num_tiles = get_arg_val<uint32_t>(4);
        const uint32_t start_id = get_arg_val<uint32_t>(5);
        constexpr auto src0_args = TensorAccessorArgs<1>();
        constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
        constexpr auto src2_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
        constexpr auto src3_args = TensorAccessorArgs<src2_args.next_compile_time_args_offset()>();
        const auto src0 = TensorAccessor(src0_args, src0_addr);
        const auto src1 = TensorAccessor(src1_args, src1_addr);
        const auto src2 = TensorAccessor(src2_args, src2_addr);
        const auto src3 = TensorAccessor(src3_args, src3_addr);
        const uint32_t bytes0 = get_local_cb_interface(0).fifo_page_size;
        const uint32_t bytes1 = get_local_cb_interface(1).fifo_page_size;
        const uint32_t bytes2 = get_local_cb_interface(2).fifo_page_size;
        const uint32_t bytes3 = get_local_cb_interface(3).fifo_page_size;
        CircularBuffer cb0(0), cb1(1), cb2(2), cb3(3);
        for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
            cb0.reserve_back(one_tile);
            cb1.reserve_back(one_tile);
            cb2.reserve_back(one_tile);
            cb3.reserve_back(one_tile);
            noc.async_read(src0, cb0, bytes0, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read(src1, cb1, bytes1, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read(src2, cb2, bytes2, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read(src3, cb3, bytes3, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb0.push_back(one_tile);
            cb1.push_back(one_tile);
            cb2.push_back(one_tile);
            cb3.push_back(one_tile);
        }
    }
}

void kernel_main() {
    constexpr uint32_t num_inputs = get_compile_time_arg_val(0);
    read_inputs<num_inputs>();
}
