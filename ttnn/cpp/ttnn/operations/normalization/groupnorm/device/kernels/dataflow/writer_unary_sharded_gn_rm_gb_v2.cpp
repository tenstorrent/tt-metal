// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void generate_tile_with_packed_bfloat16_values(uint32_t cb_id, uint32_t packed_bf16_value) {
    cb_reserve_back(cb_id, 1);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 512U; ++i) {
        *ptr++ = packed_bf16_value;
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    constexpr bool is_mcast_sender = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta = get_compile_time_arg_val(2) == 1;

    // Used only if negative mask is passed in kernel, i.e. if define FUSE_NEGATIVE_MASK is defined

    constexpr uint32_t num_cols_tile_gamma_beta = get_compile_time_arg_val(3);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    constexpr uint32_t per_core_N_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(6);

    constexpr uint32_t num_groups_per_core = get_compile_time_arg_val(7);
    constexpr uint32_t num_batches_per_core = get_compile_time_arg_val(8);
    constexpr uint32_t block_w = get_compile_time_arg_val(9);

    constexpr uint32_t size = get_compile_time_arg_val(10);

    constexpr auto gamma_args = TensorAccessorArgs<11>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto input_mask_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();

    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t input_mask_addr = get_arg_val<uint32_t>(5);

    // Used only if negative mask is passed in kernel, i.e. if define FUSE_NEGATIVE_MASK is defined
    const uint32_t input_negative_mask_addr = get_arg_val<uint32_t>(6);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(7);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(8);
    const uint32_t input_mask_tile_start_id = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_input_mask = tt::CBIndex::c_7;
    constexpr uint32_t cb_ones = tt::CBIndex::c_26;

    // constexpr uint32_t block_w = 4;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_gamma);
    const uint32_t input_mask_single_tile_size_bytes = get_tile_size(cb_input_mask);

    // input mask
    const auto mask = TensorAccessor(input_mask_args, input_mask_addr, input_mask_single_tile_size_bytes);

#if defined(FUSE_NEGATIVE_MASK)
    constexpr uint32_t cb_input_negative_mask = tt::CBIndex::c_14;
    const uint32_t input_negative_mask_single_tile_size_bytes = get_tile_size(cb_input_negative_mask);

    constexpr auto negative_mask_args = TensorAccessorArgs<input_mask_args.next_compile_time_args_offset()>();
    const auto negative_mask_tensor_accessor =
        TensorAccessor(negative_mask_args, input_negative_mask_addr, input_negative_mask_single_tile_size_bytes);

#endif

    for (uint32_t b = 0; b < num_batches_per_core; ++b) {
        uint32_t input_mask_tile_id = input_mask_tile_start_id;
#if defined(FUSE_NEGATIVE_MASK)
        uint32_t input_negative_mask_tile_id = input_mask_tile_start_id;
#endif
        for (uint32_t i = 0; i < num_groups_per_core; ++i) {
            cb_reserve_back(cb_input_mask, block_w);
            uint32_t l1_write_addr_input_mask = get_write_ptr(cb_input_mask);
            for (uint32_t j = 0; j < block_w; ++j) {
                noc_async_read_tile(input_mask_tile_id, mask, l1_write_addr_input_mask);
                l1_write_addr_input_mask += input_mask_single_tile_size_bytes;
                input_mask_tile_id += 1;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_mask, block_w);

#if defined(FUSE_NEGATIVE_MASK)
            cb_reserve_back(cb_input_negative_mask, block_w);
            uint32_t l1_write_addr_input_negative_mask = get_write_ptr(cb_input_negative_mask);
            for (uint32_t j = 0; j < block_w; ++j) {
                noc_async_read_tile(
                    input_negative_mask_tile_id, negative_mask_tensor_accessor, l1_write_addr_input_negative_mask);
                l1_write_addr_input_negative_mask += input_negative_mask_single_tile_size_bytes;
                input_negative_mask_tile_id += 1;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_negative_mask, block_w);
#endif

            if (i == 0 and b == 0) {
                constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
                const uint32_t scalar_w = get_arg_val<uint32_t>(1);
                generate_reduce_scaler(cb_in_2, scalar_w);

                constexpr uint32_t ones = 0x3F803F80;  // 2 packed bfloat16 into 1 uint32_t of value 1.0
                generate_tile_with_packed_bfloat16_values(cb_ones, ones);

                if constexpr (is_mcast_sender) {
                    constexpr uint32_t cb_in_4 = tt::CBIndex::c_4;
                    const uint32_t scalar_c = get_arg_val<uint32_t>(0);
                    generate_reduce_scaler(cb_in_4, scalar_c);
                }

                constexpr uint32_t eps_cb_id = tt::CBIndex::c_3;
                const uint32_t eps = get_arg_val<uint32_t>(2);
                generate_bcast_col_scalar(eps_cb_id, eps);

                if constexpr (fuse_gamma) {
                    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
                    const auto gamma = TensorAccessor(gamma_args, gamma_addr, size);

                    cb_reserve_back(cb_gamma, num_cols_tile_gamma_beta);
                    uint32_t l1_write_addr_gamma = get_write_ptr(cb_gamma);
                    for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
                        uint32_t tile_id = gamma_tile_start_id + w;
                        uint64_t gamma_noc_addr = get_noc_addr(tile_id, gamma);
#ifdef ARCH_BLACKHOLE
                        noc_async_read(gamma_noc_addr, l1_write_addr_gamma, 32 * 2);
                        gamma_noc_addr = get_noc_addr(l1_write_addr_gamma + 32);
                        noc_async_read_barrier();
#else
                        noc_async_read(gamma_noc_addr, l1_write_addr_gamma, 32);
                        gamma_noc_addr += 32;
#endif
                        noc_async_read(gamma_noc_addr, l1_write_addr_gamma + 512, 32);
                        l1_write_addr_gamma += gamma_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_gamma, num_cols_tile_gamma_beta);
                }

                if constexpr (fuse_beta) {
                    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
                    const auto beta = TensorAccessor(beta_args, beta_addr, size);

                    uint32_t l1_write_addr_beta = get_write_ptr(cb_beta);
                    cb_reserve_back(cb_beta, num_cols_tile_gamma_beta);
                    for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
                        uint32_t tile_id = beta_tile_start_id + w;
                        uint64_t beta_noc_addr = get_noc_addr(tile_id, beta);
#ifdef ARCH_BLACKHOLE
                        noc_async_read(beta_noc_addr, l1_write_addr_beta, 32 * 2);
                        beta_noc_addr = get_noc_addr(l1_write_addr_beta + 32);
                        noc_async_read_barrier();
#else
                        noc_async_read(beta_noc_addr, l1_write_addr_beta, 32);
                        beta_noc_addr += 32;
#endif
                        noc_async_read(beta_noc_addr, l1_write_addr_beta + 512, 32);
                        l1_write_addr_beta += beta_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_beta, num_cols_tile_gamma_beta);
                }
            }
        }
    }
}
