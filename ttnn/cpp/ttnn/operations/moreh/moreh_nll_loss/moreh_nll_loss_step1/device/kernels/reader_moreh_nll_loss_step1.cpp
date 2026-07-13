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
    auto target_addr = get_arg_val<uint32_t>(i++);
    auto weight_addr = get_arg_val<uint32_t>(i++);
    auto ignore_index = static_cast<int32_t>(get_arg_val<uint32_t>(i++));
    auto num_units_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);
    auto C = get_arg_val<uint32_t>(i++);
    auto weight_num_tile = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);
    auto target_element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_target = tt::CBIndex::c_0;
    constexpr uint32_t cb_weight = tt::CBIndex::c_1;
    constexpr uint32_t cb_weight_scratch = tt::CBIndex::c_7;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    // ublocks size defined in tiles

    constexpr bool weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr auto target_args = TensorAccessorArgs<1>();
    constexpr auto weight_args = TensorAccessorArgs<target_args.next_compile_time_args_offset()>();

    const auto addrg_target = TensorAccessor(target_args, target_addr);

#if defined(WEIGHT)
    const auto addrg_weight = TensorAccessor(weight_args, weight_addr);
#endif

    constexpr uint32_t onetile = 1;

    Scalar one, zero;
    one.f = 1.0f;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

    DataflowBuffer dfb_target_obj(cb_target);
    DataflowBuffer dfb_output_obj(cb_output);
#if defined(WEIGHT)
    DataflowBuffer dfb_weight_obj(cb_weight);
#endif

#if defined(WEIGHT)
    // weight: (1, C)
    DataflowBuffer dfb_weight_scratch_obj(cb_weight_scratch);
    read_line(dfb_weight_obj, dfb_weight_scratch_obj, addrg_weight, weight_num_tile);

    dfb_weight_obj.wait_front(weight_num_tile);
    CoreLocalMem<volatile uint16_t> weight_l1_ptr(dfb_weight_obj.get_read_ptr());
#endif

    uint32_t end_id = start_id + num_units_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t target_noc_id = i;
        read_tile(dfb_target_obj, addrg_target, target_noc_id);

        dfb_output_obj.reserve_back(onetile);
        dfb_target_obj.wait_front(onetile);

        CoreLocalMem<volatile uint16_t> output_l1_ptr(dfb_output_obj.get_write_ptr());
        CoreLocalMem<volatile int32_t> target_l1_ptr(dfb_target_obj.get_read_ptr());

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t inout_idx = h * TILE_WIDTH + w;
                int32_t target_val = target_l1_ptr[inout_idx];
                if (target_val != ignore_index) {
                    if (0 <= target_val && target_val < static_cast<int32_t>(C)) {
#if defined(WEIGHT)
                        uint32_t target_idx = target_val;
                        output_l1_ptr[inout_idx] = weight_l1_ptr[target_idx];
#else
                        output_l1_ptr[inout_idx] = u16_one;
#endif
                    } else {
                        output_l1_ptr[inout_idx] = u16_zero;
                    }
                } else {
                    output_l1_ptr[inout_idx] = u16_zero;
                }
            }
        }
        dfb_output_obj.push_back(onetile);

        dfb_target_obj.pop_front(onetile);
    }
}
