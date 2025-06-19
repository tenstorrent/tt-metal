// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

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
    auto Wt = get_arg_val<uint32_t>(i++);
    auto num_inner_tile = get_arg_val<uint32_t>(i++);
    auto weight_num_tile = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_target = tt::CBIndex::c_1;
    constexpr uint32_t cb_weight = tt::CBIndex::c_2;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;

    constexpr uint32_t cb_output = tt::CBIndex::c_16;

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

#if defined(WEIGHT)
    cb_reserve_back(cb_weight, weight_num_tile);

    // weight: (1, C)
    read_line(cb_weight, addrg_weight, weight_num_tile);

    cb_wait_front(cb_weight, weight_num_tile);
    auto weight_l1_ptr = get_read_ptr<uint16_t>(cb_weight);
#endif

    // iterate from start_id to end_id
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t target_noc_id = i;
        read_tile(cb_target, addrg_target, target_noc_id);

        cb_wait_front(cb_target, onetile);
        auto target_l1_ptr = get_read_ptr<int32_t>(cb_target);

#if defined(WEIGHT)
        cb_reserve_back(cb_tmp_weight, onetile);
        auto tmp_weight_l1_ptr = get_write_ptr<FP32_DEST_ACC_FTYPE>(cb_tmp_weight);

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
        cb_push_back(cb_tmp_weight, onetile);
#endif

        cb_reserve_back(cb_tmp_input, onetile);
        auto tmp_input_l1_ptr = get_write_ptr<FP32_DEST_ACC_FTYPE>(cb_tmp_input);

        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            for (uint32_t w = 0; w < TILE_WIDTH; w++) {
                uint32_t tilized_idx = get_tilized_idx(h, w);
                int32_t target_val = target_l1_ptr[tilized_idx];

                if (target_val != ignore_index) {
                    if (0 <= target_val && target_val < static_cast<int32_t>(C)) {
                        uint32_t n = i / num_inner_tile;
                        uint32_t inner = i % num_inner_tile;

                        // read input
                        // input: (N, C, d1,d2,... dn)
                        // noc_id: n * C * num_inner_tile + c * num_inner_tile + inner
                        uint32_t noc_id = (n * C * num_inner_tile) + target_val * num_inner_tile + inner;
                        uint32_t tilized_idx = get_tilized_idx(h, w);
                        read_value(cb_input, addrg_input, noc_id, tilized_idx);

                        cb_wait_front(cb_input, onetile);
                        auto input_l1_ptr = get_read_ptr<uint16_t>(cb_input);

                        tmp_input_l1_ptr[tilized_idx] = fp32_dest_acc_cast(input_l1_ptr[tilized_idx]);

                        cb_pop_front(cb_input, onetile);
                        continue;
                    }
                }

                tmp_input_l1_ptr[tilized_idx] = fp32_dest_acc_cast(0.0f);
            }
        }

        cb_push_back(cb_tmp_input, onetile);
        cb_pop_front(cb_target, onetile);
    }
#if defined(WEIGHT)
    cb_pop_front(cb_weight, weight_num_tile);
#endif
}
