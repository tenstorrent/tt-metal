// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t i = 0;
    auto target_addr = get_arg_val<uint32_t>(i++);
    auto output_grad_addr = get_arg_val<uint32_t>(i++);
    auto weight_addr = get_arg_val<uint32_t>(i++);
    auto divisor_addr = get_arg_val<uint32_t>(i++);
    auto ignore_index = static_cast<int32_t>(get_arg_val<uint32_t>(i++));
    auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);
    auto C = get_arg_val<uint32_t>(i++);
    auto num_inner_tile = get_arg_val<uint32_t>(i++);
    auto weight_num_tile = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_output_grad = tt::CB::c_in0;
    constexpr uint32_t cb_target = tt::CB::c_in1;
    constexpr uint32_t cb_weight = tt::CB::c_in2;
    constexpr uint32_t cb_divisor = tt::CB::c_in3;

    constexpr uint32_t cb_tmp_weight = tt::CB::c_intermed0;

    // ublocks size defined in tiles
    const uint32_t weight_tile_bytes = get_tile_size(cb_weight);
    const DataFormat weight_data_format = get_dataformat(cb_weight);

    const uint32_t divisor_tile_bytes = get_tile_size(cb_divisor);
    const DataFormat divisor_data_format = get_dataformat(cb_divisor);

    const uint32_t output_grad_tile_bytes = get_tile_size(cb_output_grad);
    const DataFormat output_grad_data_format = get_dataformat(cb_output_grad);

    const uint32_t target_tile_bytes = get_tile_size(cb_target);

    constexpr bool target_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool weight_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool divisor_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool output_grad_is_dram = get_compile_time_arg_val(3) == 1;

    const InterleavedAddrGen<target_is_dram> addrg_target = {.bank_base_address = target_addr,
                                                             .page_size = target_tile_bytes};
    const InterleavedAddrGenFast<output_grad_is_dram> addrg_output_grad = {.bank_base_address = output_grad_addr,
                                                                           .page_size = output_grad_tile_bytes,
                                                                           .data_format = output_grad_data_format};
    constexpr uint32_t onetile = 1;

#if defined(WEIGHT)
    const InterleavedAddrGen<weight_is_dram> addrg_weight = {
        .bank_base_address = weight_addr,
        .page_size = weight_tile_bytes,
    };

    // weight: (1, C)
    read_line(cb_weight, addrg_weight, weight_num_tile);

    cb_wait_front(cb_weight, weight_num_tile);
    auto weight_l1_ptr = get_read_ptr<uint16_t>(cb_weight);
#endif

#if defined(DIVISOR)
    const InterleavedAddrGenFast<divisor_is_dram> addrg_divisor = {
        .bank_base_address = divisor_addr, .page_size = divisor_tile_bytes, .data_format = divisor_data_format};

    read_tile(cb_divisor, addrg_divisor, 0);
#endif

    read_tile(cb_output_grad, addrg_output_grad, 0);

    uint32_t Ct = (C + TILE_HEIGHT - 1) / TILE_HEIGHT;

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t inner = i % num_inner_tile;
        uint32_t nct = i / num_inner_tile;
        uint32_t n = nct / Ct;
        uint32_t ct = nct % Ct;

        // target: (N, W)
        // noc_id: nt * Wt + wt
        uint32_t wt = inner;
        uint32_t Wt = num_inner_tile;
        uint32_t nt = n / TILE_HEIGHT;
        uint32_t target_noc_id = nt * Wt + wt;
        read_tile(cb_target, addrg_target, target_noc_id);

        cb_reserve_back(cb_tmp_weight, onetile);
        cb_wait_front(cb_target, onetile);

        auto tmp_weight_l1_ptr = get_write_ptr<FP32_DEST_ACC_FTYPE>(cb_tmp_weight);
        auto target_l1_ptr = get_read_ptr<int32_t>(cb_target);

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

        cb_push_back(cb_tmp_weight, onetile);

        cb_pop_front(cb_target, onetile);
    }
}
