// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void mask_tile_in_reader(uint32_t l1_addr, uint32_t mask_w = 32, uint32_t mask_h = 32) {
    union {
        float f;
        uint32_t u;
    } zero;
    zero.f = 0.0f;
    experimental::CoreLocalMem<uint16_t> ptr(l1_addr);
    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 2
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }
}

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t mask_h = get_arg_val<uint32_t>(4);
    uint32_t mask_w = get_arg_val<uint32_t>(5);

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_id_in2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    uint32_t l1_write_addr_in0;
    const auto s0 = TensorAccessor(src0_args, src0_addr);
    uint32_t l1_write_addr_in1;
    const auto s1 = TensorAccessor(src1_args, src1_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in1(cb_id_in1);
    const auto in0_tile_bytes = get_tile_size(cb_id_in0);
    const auto in1_tile_bytes = get_tile_size(cb_id_in1);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        bool last_tile = i == (start_id + num_tiles - 1);
        cb_in0.reserve_back(onetile);
        l1_write_addr_in0 = cb_in0.get_write_ptr();
        noc.async_read(s0, cb_in0, in0_tile_bytes, {.page_id = i}, {.offset_bytes = 0});

        cb_in1.reserve_back(onetile);
        l1_write_addr_in1 = cb_in1.get_write_ptr();
        noc.async_read(s1, cb_in1, in1_tile_bytes, {.page_id = i}, {.offset_bytes = 0});

        noc.async_read_barrier();

        if (last_tile) {
            mask_tile_in_reader(l1_write_addr_in0, mask_w, mask_h);
            mask_tile_in_reader(l1_write_addr_in1, mask_w, mask_h);
        }

        cb_in0.push_back(onetile);
        cb_in1.push_back(onetile);
    }
}
