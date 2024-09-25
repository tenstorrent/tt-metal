// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
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

    constexpr uint32_t cb_input = tt::CB::c_in0;
    constexpr uint32_t cb_target = tt::CB::c_in1;
    constexpr uint32_t cb_weight = tt::CB::c_in2;
    constexpr uint32_t cb_divisor = tt::CB::c_in3;

    constexpr uint32_t cb_tmp_weight = tt::CB::c_intermed0;
    constexpr uint32_t cb_tmp_input = tt::CB::c_intermed1;

    constexpr uint32_t cb_output = tt::CB::c_out0;

    // ublocks size defined in tiles
    const uint32_t input_tile_bytes = get_tile_size(cb_input);
    const auto input_data_format = get_dataformat(cb_input);

    const uint32_t target_tile_bytes = get_tile_size(cb_target);

    const uint32_t weight_tile_bytes = get_tile_size(cb_weight);
    const auto weight_data_format = get_dataformat(cb_weight);

    const uint32_t divisor_tile_bytes = get_tile_size(cb_divisor);
    const auto divisor_data_format = get_dataformat(cb_divisor);

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool target_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool weight_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool divisor_is_dram = get_compile_time_arg_val(3) == 1;

    const InterleavedAddrGen<input_is_dram> addrg_input = {
        .bank_base_address = input_addr,
        .page_size = input_tile_bytes,
    };

    const InterleavedAddrGen<target_is_dram> addrg_target = {
        .bank_base_address = target_addr, .page_size = target_tile_bytes};

    const InterleavedAddrGen<weight_is_dram> addrg_weight = {
        .bank_base_address = weight_addr,
        .page_size = weight_tile_bytes,
    };

    constexpr uint32_t onetile = 1;

#if defined(DIVISOR)
    const InterleavedAddrGenFast<divisor_is_dram> addrg_divisor = {
        .bank_base_address = divisor_addr, .page_size = divisor_tile_bytes, .data_format = divisor_data_format};

    read_tile(cb_divisor, addrg_divisor, 0);
#endif

    uint32_t Ct = (C + TILE_HEIGHT - 1) / TILE_HEIGHT;

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        // loop from n_start to n_end
        uint32_t n_start = i * TILE_HEIGHT;
        uint32_t n_end = min(i * TILE_HEIGHT + TILE_HEIGHT, N);
        uint32_t nt = i;

        // target: (1, N)
        auto target_noc_id = nt;
        read_tile(cb_target, addrg_target, target_noc_id);

#if defined(WEIGHT)
        cb_reserve_back(cb_tmp_weight, onetile);

        auto tmp_weight_l1_ptr = get_write_ptr<FP32_DEST_ACC_FTYPE>(cb_tmp_weight);
#endif

        cb_reserve_back(cb_tmp_input, onetile);
        cb_wait_front(cb_target, onetile);

        auto tmp_input_l1_ptr = get_write_ptr<FP32_DEST_ACC_FTYPE>(cb_tmp_input);
        auto target_l1_ptr = get_read_ptr<int32_t>(cb_target);

        uint32_t w = 0;
        for (uint32_t n = n_start; n < n_end; n++, w++) {
            uint32_t tilized_idx = get_tilized_idx(0, w);
            int32_t target_val = target_l1_ptr[tilized_idx];

            if (target_val != ignore_index && (0 <= target_val && target_val < static_cast<int32_t>(C))) {
                // input: (N, C)
                // noc_id: nt * Ct + ct
                uint32_t noc_id = (nt * Ct) + (target_val / TILE_WIDTH);
                uint32_t input_tilized_idx = get_tilized_idx(n, target_val);
                read_value(cb_input, addrg_input, noc_id, input_tilized_idx);

                cb_wait_front(cb_input, onetile);
                auto input_l1_ptr = get_read_ptr<uint16_t>(cb_input);
                tmp_input_l1_ptr[tilized_idx] = fp32_dest_acc_cast(input_l1_ptr[input_tilized_idx]);

                cb_pop_front(cb_input, onetile);
            } else {
                tmp_input_l1_ptr[tilized_idx] = fp32_dest_acc_cast(0.0f);
            }

#if defined(WEIGHT)
            // read weight
            // weight: (1, C)
            // noc_id: target_val / TILE_WIDTH
            uint32_t noc_id = target_val / TILE_WIDTH;
            uint32_t weight_tilized_idx = get_tilized_idx(0, target_val);
            read_value(cb_weight, addrg_weight, noc_id, weight_tilized_idx);

            cb_wait_front(cb_weight, onetile);
            auto weight_l1_ptr = get_read_ptr<uint16_t>(cb_weight);
            tmp_weight_l1_ptr[tilized_idx] = fp32_dest_acc_cast(weight_l1_ptr[weight_tilized_idx]);
            cb_pop_front(cb_weight, onetile);
#endif
        }
        cb_push_back(cb_tmp_input, onetile);
#if defined(WEIGHT)
        cb_push_back(cb_tmp_weight, onetile);
#endif
        cb_pop_front(cb_target, onetile);
    }
}
