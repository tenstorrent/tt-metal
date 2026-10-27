// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    using namespace tt::constants;
    auto ignore_index = static_cast<int32_t>(get_arg(args::ignore_index));
    auto num_units_per_core = get_arg(args::num_units_per_core);
    auto start_id = get_arg(args::start_id);
    auto C = get_arg(args::C);
    auto weight_num_tile = get_arg(args::weight_num_tile);

    // ublocks size defined in tiles

    constexpr bool weight_has_value = get_arg(args::weight_has_value) == 1;

    const auto addrg_target = TensorAccessor(tensor::target);

    DataflowBuffer dfb_target_obj(dfb::target);
    DataflowBuffer dfb_output_obj(dfb::output);
#if defined(WEIGHT)
    DataflowBuffer dfb_weight_obj(dfb::weight);
    const uint32_t weight_tile_bytes = dfb_weight_obj.get_tile_size();
    auto weight_element_size = weight_tile_bytes / 1024;
    const auto addrg_weight = TensorAccessor(tensor::weight);
#endif

    constexpr uint32_t onetile = 1;

    Scalar one, zero;
    one.f = 1.0f;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

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

                        uint32_t noc_id = target_idx / TILE_WIDTH;
                        uint32_t weight_tilized_idx = get_tilized_idx(0, target_idx);
                        read_value(dfb_weight_obj, addrg_weight, noc_id, weight_tilized_idx);

                        dfb_weight_obj.wait_front(onetile);
                        CoreLocalMem<volatile uint16_t> weight_l1_ptr(dfb_weight_obj.get_read_ptr());

                        output_l1_ptr[inout_idx] = weight_l1_ptr[weight_tilized_idx];

                        dfb_weight_obj.pop_front(onetile);
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
