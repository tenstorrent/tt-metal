// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
    auto Nt = get_arg(args::Nt);
    auto Ct = get_arg(args::Ct);

    // ublocks size defined in tiles

    const auto addrg_target = TensorAccessor(tensor::target);
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_target_obj(dfb::target);
    DataflowBuffer dfb_output_grad_obj(dfb::output_grad);
    DataflowBuffer dfb_input_grad_obj(dfb::input_grad);
#if defined(WEIGHT)
    DataflowBuffer dfb_weight_obj(dfb::weight);
    const auto addrg_weight = TensorAccessor(tensor::weight);

    DataflowBuffer dfb_weight_scratch_obj(dfb::weight_scratch);
    read_line(dfb_weight_obj, dfb_weight_scratch_obj, addrg_weight, Ct);

    dfb_weight_obj.wait_front(Ct);
    CoreLocalMem<volatile uint16_t> weight_l1_ptr(dfb_weight_obj.get_read_ptr());
#endif

    const auto addrg_output_grad = TensorAccessor(tensor::output_grad);

    DataflowBuffer dfb_output_grad_scratch_obj(dfb::output_grad_scratch);
    read_line(dfb_output_grad_obj, dfb_output_grad_scratch_obj, addrg_output_grad, Nt);

    dfb_output_grad_obj.wait_front(Nt);

    auto zero = fp32_to_bf16_truncate(0.0f);

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t nt = i / Ct;
        uint32_t ct = i % Ct;

        auto target_noc_id = nt;
        read_tile(dfb_target_obj, addrg_target, target_noc_id);

        dfb_input_grad_obj.reserve_back(onetile);
        dfb_target_obj.wait_front(onetile);

        CoreLocalMem<volatile uint16_t> input_grad_l1_ptr(dfb_input_grad_obj.get_write_ptr());
        CoreLocalMem<volatile int32_t> target_l1_ptr(dfb_target_obj.get_read_ptr());
        CoreLocalMem<volatile uint16_t> output_grad_l1_ptr(dfb_output_grad_obj.get_read_ptr());

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t n = nt * TILE_HEIGHT + h;
                uint32_t c = ct * TILE_WIDTH + w;

                uint32_t target_tilized_idx = get_tilized_idx(0, h);
                int32_t target_val = target_l1_ptr[target_tilized_idx];

                uint32_t input_grad_idx = get_tilized_idx(h, w);

                uint16_t input_grad_val;

                if (target_val != ignore_index && target_val == static_cast<int32_t>(c)) {
                    float output_grad_val = bf16_to_fp32(output_grad_l1_ptr[n]);
#if defined(WEIGHT)
                    float weight_val = bf16_to_fp32(weight_l1_ptr[target_val]);

                    input_grad_val = fp32_to_bf16_truncate(-output_grad_val * weight_val);
#else
                    input_grad_val = fp32_to_bf16_truncate(-output_grad_val);
#endif
                } else {
                    input_grad_val = zero;
                }
                input_grad_l1_ptr[input_grad_idx] = input_grad_val;
            }
        }

        dfb_input_grad_obj.push_back(onetile);

        dfb_target_obj.pop_front(onetile);
    }
}
