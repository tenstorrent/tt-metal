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
    CoreLocalMem<volatile uint16_t> weight_l1_ptr(static_cast<uint32_t>(0));
    with_nullable_token(dfb::weight, [&](const DFBBindingToken& weight_tok) {
        with_nullable_token(dfb::weight_scratch, [&](const DFBBindingToken& scratch_tok) {
            with_nullable_token(tensor::weight, [&](const auto& weight_tensor_tok) {
                DataflowBuffer dfb_weight_obj(weight_tok);
                DataflowBuffer dfb_weight_scratch_obj(scratch_tok);
                const auto addrg_weight = TensorAccessor(weight_tensor_tok);
                read_line(dfb_weight_obj, dfb_weight_scratch_obj, addrg_weight, weight_num_tile);

                dfb_weight_obj.wait_front(weight_num_tile);
                weight_l1_ptr = CoreLocalMem<volatile uint16_t>(dfb_weight_obj.get_read_ptr());
            });
        });
    });

    with_nullable_token(dfb::divisor, [&](const DFBBindingToken& divisor_tok) {
        with_nullable_token(tensor::divisor, [&](const auto& divisor_tensor_tok) {
            const auto addrg_divisor = TensorAccessor(divisor_tensor_tok);
            DataflowBuffer dfb_divisor_obj(divisor_tok);
            read_tile(dfb_divisor_obj, addrg_divisor, 0);
        });
    });

    DataflowBuffer dfb_output_grad_obj(dfb::output_grad);
    read_tile(dfb_output_grad_obj, addrg_output_grad, 0);

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t inner = i % num_inner_tile;
        uint32_t nc = i / num_inner_tile;
        uint32_t n = nc / C;
        uint32_t c = nc % C;

        uint32_t target_noc_id = n * num_inner_tile + inner;
        read_tile(dfb_target_obj, addrg_target, target_noc_id);

        dfb_tmp_weight_obj.reserve_back(onetile);
        dfb_target_obj.wait_front(onetile);

        CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_weight_l1_ptr(dfb_tmp_weight_obj.get_write_ptr());
        CoreLocalMem<volatile int32_t> target_l1_ptr(dfb_target_obj.get_read_ptr());

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t idx = h * TILE_WIDTH + w;
                int32_t target_val = target_l1_ptr[idx];
                FP32_DEST_ACC_FTYPE val;

                if (target_val != ignore_index && target_val == static_cast<int32_t>(c)) {
                    val = fp32_dest_acc_cast(1.0f);
                    with_nullable_token(dfb::weight, [&](const DFBBindingToken&) {
                        val = fp32_dest_acc_cast(weight_l1_ptr[target_val]);
                    });
                } else {
                    val = fp32_dest_acc_cast(0.0f);
                }

                tmp_weight_l1_ptr[idx] = val;
            }
        }

        dfb_tmp_weight_obj.push_back(onetile);

        dfb_target_obj.pop_front(onetile);
    }
}
