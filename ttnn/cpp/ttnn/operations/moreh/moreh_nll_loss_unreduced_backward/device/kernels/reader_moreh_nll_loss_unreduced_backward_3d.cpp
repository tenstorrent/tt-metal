// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
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

    constexpr uint32_t cb_target = tt::CB::c_in0;
    constexpr uint32_t cb_output_grad = tt::CB::c_in1;
    constexpr uint32_t cb_weight = tt::CB::c_in2;

    constexpr uint32_t cb_input_grad = tt::CB::c_out0;

    // ublocks size defined in tiles
    const uint32_t target_tile_bytes = get_tile_size(cb_target);

    const uint32_t weight_tile_bytes = get_tile_size(cb_weight);
    const DataFormat weight_data_format = get_dataformat(cb_weight);

    const uint32_t output_grad_tile_bytes = get_tile_size(cb_output_grad);
    const DataFormat output_grad_data_format = get_dataformat(cb_output_grad);

    constexpr bool target_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool output_grad_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool weight_is_dram = get_compile_time_arg_val(2) == 1;

    const InterleavedAddrGen<target_is_dram> addrg_target = {.bank_base_address = target_addr,
                                                             .page_size = target_tile_bytes};
    constexpr uint32_t onetile = 1;

#if defined(WEIGHT)
    const InterleavedAddrGen<weight_is_dram> addrg_weight = {
        .bank_base_address = weight_addr,
        .page_size = weight_tile_bytes,
    };

    // weight: (1, C)
    read_line(cb_weight, addrg_weight, Ct);

    cb_wait_front(cb_weight, Ct);
    auto weight_l1_ptr = get_read_ptr<uint16_t>(cb_weight);
#endif

    const InterleavedAddrGenFast<output_grad_is_dram> addrg_output_grad = {.bank_base_address = output_grad_addr,
                                                                           .page_size = output_grad_tile_bytes,
                                                                           .data_format = output_grad_data_format};

    auto zero = float_to_bfloat16(0.0f);

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t wt = i % Wt;
        uint32_t nct = i / Wt;
        uint32_t n = nct / Ct;
        uint32_t nt = n / TILE_HEIGHT;
        uint32_t ct = nct % Ct;

        // target: (N, W)
        auto target_noc_id = nt * Wt + wt;
        read_tile(cb_target, addrg_target, target_noc_id);

        // output_grad: (N, W)
        auto output_grad_noc_id = nt * Wt + wt;
        read_tile(cb_output_grad, addrg_output_grad, output_grad_noc_id);

        cb_reserve_back(cb_input_grad, onetile);
        cb_wait_front(cb_target, onetile);
        cb_wait_front(cb_output_grad, onetile);

        auto input_grad_l1_ptr = get_write_ptr<uint16_t>(cb_input_grad);
        auto target_l1_ptr = get_read_ptr<int32_t>(cb_target);
        auto output_grad_l1_ptr = get_read_ptr<uint16_t>(cb_output_grad);

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t nw_tilized_idx = get_tilized_idx(n % TILE_HEIGHT, w);  // target(n, w)
                int32_t target_val = target_l1_ptr[nw_tilized_idx];

                uint32_t c = ct * TILE_HEIGHT + h;
                uint32_t input_grad_idx = get_tilized_idx(h, w);  // input_grad(c, w)

                uint16_t input_grad_val;

                if (target_val != ignore_index && target_val == static_cast<int32_t>(c)) {
                    float output_grad_val = bfloat16_to_float(output_grad_l1_ptr[nw_tilized_idx]);

#if defined(WEIGHT)
                    float weight_val = bfloat16_to_float(weight_l1_ptr[target_val]);

                    input_grad_val = float_to_bfloat16(-output_grad_val * weight_val);
#else
                    input_grad_val = float_to_bfloat16(-output_grad_val);
#endif
                } else {
                    input_grad_val = zero;
                }
                input_grad_l1_ptr[input_grad_idx] = input_grad_val;
            }
        }

        cb_push_back(cb_input_grad, onetile);

        cb_pop_front(cb_target, onetile);

        cb_pop_front(cb_output_grad, onetile);
    }
}
