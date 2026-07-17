// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

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
    auto Wt = get_arg_val<uint32_t>(i++);
    auto num_inner_tile = get_arg_val<uint32_t>(i++);
    auto weight_num_tile = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_target = tt::CBIndex::c_1;
    constexpr uint32_t cb_weight = tt::CBIndex::c_2;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;

    constexpr uint32_t cb_weight_scratch = tt::CBIndex::c_7;

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

    DataflowBuffer dfb_divisor_obj(cb_divisor);
    read_tile(dfb_divisor_obj, addrg_divisor, 0);
#endif

    DataflowBuffer dfb_input_obj(cb_input);
    DataflowBuffer dfb_target_obj(cb_target);
    DataflowBuffer dfb_tmp_input_obj(cb_tmp_input);
#if defined(WEIGHT)
    DataflowBuffer dfb_weight_obj(cb_weight);
    DataflowBuffer dfb_tmp_weight_obj(cb_tmp_weight);

    dfb_weight_obj.reserve_back(weight_num_tile);

    DataflowBuffer dfb_weight_scratch_obj(cb_weight_scratch);
    read_line(dfb_weight_obj, dfb_weight_scratch_obj, addrg_weight, weight_num_tile);

    dfb_weight_obj.wait_front(weight_num_tile);
    CoreLocalMem<volatile uint16_t> weight_l1_ptr(dfb_weight_obj.get_read_ptr());
#endif

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t target_noc_id = i;
        read_tile(dfb_target_obj, addrg_target, target_noc_id);

        dfb_target_obj.wait_front(onetile);
        CoreLocalMem<volatile int32_t> target_l1_ptr(dfb_target_obj.get_read_ptr());

#if defined(WEIGHT)
        dfb_tmp_weight_obj.reserve_back(onetile);
        CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_weight_l1_ptr(dfb_tmp_weight_obj.get_write_ptr());

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t tilized_idx = get_tilized_idx(h, w);
                int32_t target_val = target_l1_ptr[tilized_idx];
                if (target_val != ignore_index) {
                    if (0 <= target_val && target_val < static_cast<int32_t>(C)) {
                        tmp_weight_l1_ptr[tilized_idx] = fp32_dest_acc_cast(weight_l1_ptr[target_val]);
                        continue;
                    }
                }
                tmp_weight_l1_ptr[tilized_idx] = fp32_dest_acc_cast(0.0f);
            }
        }
        dfb_tmp_weight_obj.push_back(onetile);
#endif

        dfb_tmp_input_obj.reserve_back(onetile);
        CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_input_l1_ptr(dfb_tmp_input_obj.get_write_ptr());

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t tilized_idx = get_tilized_idx(h, w);
                int32_t target_val = target_l1_ptr[tilized_idx];

                if (target_val != ignore_index) {
                    if (0 <= target_val && target_val < static_cast<int32_t>(C)) {
                        uint32_t n = i / num_inner_tile;
                        uint32_t inner = i % num_inner_tile;

                        uint32_t noc_id = (n * C * num_inner_tile) + target_val * num_inner_tile + inner;
                        uint32_t tilized_idx = get_tilized_idx(h, w);
                        read_value(dfb_input_obj, addrg_input, noc_id, tilized_idx);

                        dfb_input_obj.wait_front(onetile);
                        CoreLocalMem<volatile uint16_t> input_l1_ptr(dfb_input_obj.get_read_ptr());

                        tmp_input_l1_ptr[tilized_idx] = fp32_dest_acc_cast(input_l1_ptr[tilized_idx]);

                        dfb_input_obj.pop_front(onetile);
                        continue;
                    }
                }

                tmp_input_l1_ptr[tilized_idx] = fp32_dest_acc_cast(0.0f);
            }
        }

        dfb_tmp_input_obj.push_back(onetile);
        dfb_target_obj.pop_front(onetile);
    }
#if defined(WEIGHT)
    dfb_weight_obj.pop_front(weight_num_tile);
#endif
}
