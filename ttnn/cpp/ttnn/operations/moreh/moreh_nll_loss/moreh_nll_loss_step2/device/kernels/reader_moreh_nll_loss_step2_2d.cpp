// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

void kernel_main() {
    using namespace tt::constants;
    uint32_t i = 0;
    auto input_addr = get_arg_val<uint32_t>(i++);
    auto target_addr = get_arg_val<uint32_t>(i++);
    auto weight_addr = get_arg_val<uint32_t>(i++);
    auto divisor_addr = get_arg_val<uint32_t>(i++);
    auto ignore_index = static_cast<int32_t>(get_arg_val<uint32_t>(i++));
    auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);
    auto N = get_arg_val<uint32_t>(i++);
    auto C = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_target = tt::CBIndex::c_1;
    constexpr uint32_t cb_weight = tt::CBIndex::c_2;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;

    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    // ublocks size defined in tiles
    const auto input_data_format = get_dataformat(cb_input);

    const auto weight_data_format = get_dataformat(cb_weight);

    const auto divisor_data_format = get_dataformat(cb_divisor);

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto target_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<target_args.next_compile_time_args_offset()>();
    constexpr auto divisor_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    const auto addrg_input = TensorAccessor(input_args, input_addr);
    const auto addrg_target = TensorAccessor(target_args, target_addr);
    const auto addrg_weight = TensorAccessor(weight_args, weight_addr);

    constexpr uint32_t onetile = 1;

#if defined(DIVISOR)
    const auto addrg_divisor = TensorAccessor(divisor_args, divisor_addr);

    read_tile(cb_divisor, addrg_divisor, 0);
#endif

    uint32_t Ct = (C + TILE_HEIGHT - 1) / TILE_HEIGHT;

    experimental::CircularBuffer cb_input_obj(cb_input);
    experimental::CircularBuffer cb_target_obj(cb_target);
    experimental::CircularBuffer cb_tmp_input_obj(cb_tmp_input);
#if defined(WEIGHT)
    experimental::CircularBuffer cb_weight_obj(cb_weight);
    experimental::CircularBuffer cb_tmp_weight_obj(cb_tmp_weight);
#endif

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t n_start = i * TILE_HEIGHT;
        uint32_t n_end = std::min(i * TILE_HEIGHT + TILE_HEIGHT, N);
        uint32_t nt = i;

        auto target_noc_id = nt;
        read_tile(cb_target, addrg_target, target_noc_id);

#if defined(WEIGHT)
        cb_tmp_weight_obj.reserve_back(onetile);
        experimental::CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_weight_l1_ptr(cb_tmp_weight_obj.get_write_ptr());
#endif

        cb_tmp_input_obj.reserve_back(onetile);
        cb_target_obj.wait_front(onetile);

        experimental::CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_input_l1_ptr(cb_tmp_input_obj.get_write_ptr());
        experimental::CoreLocalMem<volatile int32_t> target_l1_ptr(cb_target_obj.get_read_ptr());

        uint32_t w = 0;
        for (uint32_t n = n_start; n < n_end; n++, w++) {
            uint32_t tilized_idx = get_tilized_idx(0, w);
            int32_t target_val = target_l1_ptr[tilized_idx];

            if (target_val != ignore_index && (0 <= target_val && target_val < static_cast<int32_t>(C))) {
                uint32_t noc_id = (nt * Ct) + (target_val / TILE_WIDTH);
                uint32_t input_tilized_idx = get_tilized_idx(n, target_val);
                read_value(cb_input, addrg_input, noc_id, input_tilized_idx);

                cb_input_obj.wait_front(onetile);
                experimental::CoreLocalMem<volatile uint16_t> input_l1_ptr(cb_input_obj.get_read_ptr());
                tmp_input_l1_ptr[tilized_idx] = fp32_dest_acc_cast(input_l1_ptr[input_tilized_idx]);

                cb_input_obj.pop_front(onetile);
            } else {
                tmp_input_l1_ptr[tilized_idx] = fp32_dest_acc_cast(0.0f);
            }

#if defined(WEIGHT)
            uint32_t noc_id = target_val / TILE_WIDTH;
            uint32_t weight_tilized_idx = get_tilized_idx(0, target_val);
            read_value(cb_weight, addrg_weight, noc_id, weight_tilized_idx);

            cb_weight_obj.wait_front(onetile);
            experimental::CoreLocalMem<volatile uint16_t> weight_l1_ptr(cb_weight_obj.get_read_ptr());
            tmp_weight_l1_ptr[tilized_idx] = fp32_dest_acc_cast(weight_l1_ptr[weight_tilized_idx]);
            cb_weight_obj.pop_front(onetile);
#endif
        }
        cb_tmp_input_obj.push_back(onetile);
#if defined(WEIGHT)
        cb_tmp_weight_obj.push_back(onetile);
#endif
        cb_target_obj.pop_front(onetile);
    }
}
