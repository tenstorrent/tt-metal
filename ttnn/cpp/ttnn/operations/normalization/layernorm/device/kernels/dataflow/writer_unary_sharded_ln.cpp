// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    constexpr bool is_all_to_all_worker              = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma                       = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta                        = get_compile_time_arg_val(2) == 1;
    constexpr bool gamma_is_dram                    = get_compile_time_arg_val(3) == 1;
    constexpr bool beta_is_dram                     = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t block_w                      = get_compile_time_arg_val(5);

    const uint32_t gamma_addr                     = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr                      = get_arg_val<uint32_t>(4);
    const uint32_t gamma_tile_start_id            = get_arg_val<uint32_t>(5);
    const uint32_t beta_tile_start_id             = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;


    {
        constexpr uint32_t cb_in_2 = tt::CB::c_in2;
        const uint32_t scalar_w = get_arg_val<uint32_t>(1);
        generate_reduce_scaler(cb_in_2, scalar_w);
    }
    if constexpr(is_all_to_all_worker) {
        constexpr uint32_t cb_in_4 = tt::CB::c_in4;
        const uint32_t scalar_c = get_arg_val<uint32_t>(0);
        generate_reduce_scaler(cb_in_4, scalar_c);
    }
    constexpr uint32_t eps_cb_id = 3;
    const uint32_t eps = get_arg_val<uint32_t>(2);
    generate_bcast_col_scalar(eps_cb_id, eps);

    if constexpr(fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const DataFormat gamma_data_format = get_dataformat(cb_gamma);
        const InterleavedAddrGenFast<gamma_is_dram> gamma = {
            .bank_base_address = gamma_addr,
            .page_size = gamma_tile_bytes,
            .data_format = gamma_data_format
        };

        uint32_t l1_write_addr_gamma = get_write_ptr(cb_gamma);
        cb_reserve_back(cb_gamma, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            noc_async_read_tile(tile_id, gamma, l1_write_addr_gamma);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, block_w);
    }

    if constexpr(fuse_beta) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
        const DataFormat beta_data_format = get_dataformat(cb_beta);
        const InterleavedAddrGenFast<beta_is_dram> beta = {
            .bank_base_address = beta_addr,
            .page_size = beta_tile_bytes,
            .data_format = beta_data_format
        };

        uint32_t l1_write_addr_beta = get_write_ptr(cb_beta);
        cb_reserve_back(cb_beta, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            noc_async_read_tile(tile_id, beta, l1_write_addr_beta);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, block_w);
    }

}
