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
    auto num_tiles_per_core = get_arg(args::num_tiles_per_core);
    auto start_id = get_arg(args::start_id);
    auto C = get_arg(args::C);
    auto num_inner_tile = get_arg(args::num_inner_tile);
    auto weight_num_tile = get_arg(args::weight_num_tile);

    const auto addrg_target = TensorAccessor(tensor::target);
    const auto addrg_output_grad = TensorAccessor(tensor::output_grad);
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_target_obj(dfb::target);
    DataflowBuffer dfb_tmp_weight_obj(dfb::tmp_weight);
#if defined(WEIGHT)
    DataflowBuffer dfb_weight_obj(dfb::weight);
    const auto addrg_weight = TensorAccessor(tensor::weight);

    DataflowBuffer dfb_weight_scratch_obj(dfb::weight_scratch);
    read_line(dfb_weight_obj, dfb_weight_scratch_obj, addrg_weight, weight_num_tile);

    dfb_weight_obj.wait_front(weight_num_tile);
    CoreLocalMem<volatile uint16_t> weight_l1_ptr(dfb_weight_obj.get_read_ptr());
#endif

#if defined(DIVISOR)
    const auto addrg_divisor = TensorAccessor(tensor::divisor);

    DataflowBuffer dfb_divisor_obj(dfb::divisor);
    read_tile(dfb_divisor_obj, addrg_divisor, 0);
#endif

    DataflowBuffer dfb_output_grad_obj(dfb::output_grad);
    read_tile(dfb_output_grad_obj, addrg_output_grad, 0);

    uint32_t Ct = (C + TILE_HEIGHT - 1) / TILE_HEIGHT;

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t inner = i % num_inner_tile;
        uint32_t nct = i / num_inner_tile;
        uint32_t n = nct / Ct;
        uint32_t ct = nct % Ct;

        uint32_t wt = inner;
        uint32_t Wt = num_inner_tile;
        uint32_t nt = n / TILE_HEIGHT;
        uint32_t target_noc_id = nt * Wt + wt;
        read_tile(dfb_target_obj, addrg_target, target_noc_id);

        dfb_tmp_weight_obj.reserve_back(onetile);
        dfb_target_obj.wait_front(onetile);

        CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_weight_l1_ptr(dfb_tmp_weight_obj.get_write_ptr());
        CoreLocalMem<volatile int32_t> target_l1_ptr(dfb_target_obj.get_read_ptr());

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t target_tilized_idx = get_tilized_idx(n % TILE_HEIGHT, w);
                int32_t target_val = target_l1_ptr[target_tilized_idx];

                uint32_t c = ct * TILE_HEIGHT + h;
                uint32_t tmp_weight_tilized_idx = get_tilized_idx(h, w);

                if (target_val != ignore_index && target_val == static_cast<int32_t>(c)) {
#if defined(WEIGHT)
                    tmp_weight_l1_ptr[tmp_weight_tilized_idx] = fp32_dest_acc_cast(weight_l1_ptr[target_val]);
#else
                    tmp_weight_l1_ptr[tmp_weight_tilized_idx] = fp32_dest_acc_cast(1.0f);
#endif
                    continue;
                }
                tmp_weight_l1_ptr[tmp_weight_tilized_idx] = fp32_dest_acc_cast(0.0f);
            }
        }

        dfb_tmp_weight_obj.push_back(onetile);

        dfb_target_obj.pop_front(onetile);
    }
}
