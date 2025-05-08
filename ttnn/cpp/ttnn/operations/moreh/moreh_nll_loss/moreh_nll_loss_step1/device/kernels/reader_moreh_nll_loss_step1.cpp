// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
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

    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    // ublocks size defined in tiles
    const uint32_t target_tile_bytes = get_tile_size(cb_target);

    constexpr bool target_is_dram = get_compile_time_arg_val(0) == 1;
#if defined(WEIGHT)
    constexpr bool weight_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool weight_has_value = get_compile_time_arg_val(2) == 1;
#endif

    const InterleavedAddrGen<target_is_dram> addrg_target = {
        .bank_base_address = target_addr, .page_size = target_tile_bytes};

#if defined(WEIGHT)
    const uint32_t weight_tile_bytes = get_tile_size(cb_weight);
    const DataFormat weight_data_format = get_dataformat(cb_weight);
    const InterleavedAddrGen<weight_is_dram> addrg_weight = {
        .bank_base_address = weight_addr,
        .page_size = weight_tile_bytes,
    };
#endif

    constexpr uint32_t onetile = 1;

    Scalar one, zero;
    one.f = 1.0f;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

#if defined(WEIGHT)
    // weight: (1, C)
    read_line(cb_weight, addrg_weight, weight_num_tile);

    cb_wait_front(cb_weight, weight_num_tile);
    auto weight_l1_ptr = get_read_ptr<uint16_t>(cb_weight);
#endif

    uint32_t end_id = start_id + num_units_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        // target: (N, d1, d2, .. dk)
        uint32_t target_noc_id = i;
        read_tile(cb_target, addrg_target, target_noc_id);

        cb_reserve_back(cb_output, onetile);
        cb_wait_front(cb_target, onetile);

        auto output_l1_ptr = get_write_ptr<uint16_t>(cb_output);
        auto target_l1_ptr = get_read_ptr<int32_t>(cb_target);

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
        cb_push_back(cb_output, onetile);

        cb_pop_front(cb_target, onetile);
    }
}
