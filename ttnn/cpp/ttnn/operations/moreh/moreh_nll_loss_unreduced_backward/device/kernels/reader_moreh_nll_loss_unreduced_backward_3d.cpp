// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    using namespace tt::constants;
    uint32_t i = 0;
    auto target_addr = get_arg_val<uint32_t>(i++);
    auto output_grad_addr = get_arg_val<uint32_t>(i++);
    auto weight_addr = get_arg_val<uint32_t>(i++);
    auto ignore_index = static_cast<int32_t>(get_arg_val<uint32_t>(i++));
    auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);
    auto C = get_arg_val<uint32_t>(i++);
    auto Ct = get_arg_val<uint32_t>(i++);
    auto Wt = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_target = tt::CBIndex::c_0;
    constexpr uint32_t cb_output_grad = tt::CBIndex::c_1;
    constexpr uint32_t cb_weight = tt::CBIndex::c_2;

    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;

    constexpr uint32_t cb_weight_scratch = tt::CBIndex::c_7;

    // ublocks size defined in tiles

    const DataFormat weight_data_format = get_dataformat(cb_weight);

    const DataFormat output_grad_data_format = get_dataformat(cb_output_grad);

    constexpr auto target_args = TensorAccessorArgs<0>();
    constexpr auto output_grad_args = TensorAccessorArgs<target_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<output_grad_args.next_compile_time_args_offset()>();

    const auto addrg_target = TensorAccessor(target_args, target_addr);
    constexpr uint32_t onetile = 1;

    CircularBuffer cb_target_obj(cb_target);
    CircularBuffer cb_output_grad_obj(cb_output_grad);
    CircularBuffer cb_input_grad_obj(cb_input_grad);
#if defined(WEIGHT)
    CircularBuffer cb_weight_obj(cb_weight);
    const auto addrg_weight = TensorAccessor(weight_args, weight_addr);

    read_line(cb_weight, cb_weight_scratch, addrg_weight, Ct);

    cb_weight_obj.wait_front(Ct);
    CoreLocalMem<volatile uint16_t> weight_l1_ptr(cb_weight_obj.get_read_ptr());
#endif

    const auto addrg_output_grad = TensorAccessor(output_grad_args, output_grad_addr);

    auto zero = fp32_to_bf16_truncate(0.0f);

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t wt = i % Wt;
        uint32_t nct = i / Wt;
        uint32_t n = nct / Ct;
        uint32_t nt = n / TILE_HEIGHT;
        uint32_t ct = nct % Ct;

        auto target_noc_id = nt * Wt + wt;
        read_tile(cb_target, addrg_target, target_noc_id);

        auto output_grad_noc_id = nt * Wt + wt;
        read_tile(cb_output_grad, addrg_output_grad, output_grad_noc_id);

        cb_input_grad_obj.reserve_back(onetile);
        cb_target_obj.wait_front(onetile);
        cb_output_grad_obj.wait_front(onetile);

        CoreLocalMem<volatile uint16_t> input_grad_l1_ptr(cb_input_grad_obj.get_write_ptr());
        CoreLocalMem<volatile int32_t> target_l1_ptr(cb_target_obj.get_read_ptr());
        CoreLocalMem<volatile uint16_t> output_grad_l1_ptr(cb_output_grad_obj.get_read_ptr());

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t nw_tilized_idx = get_tilized_idx(n % TILE_HEIGHT, w);
                int32_t target_val = target_l1_ptr[nw_tilized_idx];

                uint32_t c = ct * TILE_HEIGHT + h;
                uint32_t input_grad_idx = get_tilized_idx(h, w);

                uint16_t input_grad_val;

                if (target_val != ignore_index && target_val == static_cast<int32_t>(c)) {
                    float output_grad_val = bf16_to_fp32(output_grad_l1_ptr[nw_tilized_idx]);

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

        cb_input_grad_obj.push_back(onetile);

        cb_target_obj.pop_front(onetile);

        cb_output_grad_obj.pop_front(onetile);
    }
}
