// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/constants.hpp"
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    constexpr bool is_mcast_sender = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta = get_compile_time_arg_val(2) == 1;

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

    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(7);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(8);
    const uint32_t input_mask_tile_start_id = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_input_mask = tt::CBIndex::c_7;
    constexpr uint32_t cb_ones = tt::CBIndex::c_26;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_gamma);
    const uint32_t input_mask_single_tile_size_bytes = get_tile_size(cb_input_mask);

    // input mask
    const auto mask = TensorAccessor(input_mask_args, input_mask_addr, input_mask_single_tile_size_bytes);

    constexpr uint32_t eps_cb_id = tt::CBIndex::c_3;
    const uint32_t eps = get_arg_val<uint32_t>(2);
    generate_bcast_col_scalar(eps_cb_id, eps);

    uint32_t input_mask_tile_id = input_mask_tile_start_id;
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
    }

    if constexpr (fuse_gamma) {
        constexpr uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        constexpr uint32_t gamma_element_bytes = gamma_tile_bytes / tt::constants::TILE_HW;
        constexpr uint32_t gamma_face_bytes = gamma_element_bytes * tt::constants::FACE_HW;
        constexpr uint32_t gamma_face_w_bytes = gamma_element_bytes * tt::constants::FACE_WIDTH;
        const auto gamma = TensorAccessor(gamma_args, gamma_addr, size);

        cb_reserve_back(cb_gamma, num_cols_tile_gamma_beta);
        auto l1_write_addr_gamma = get_write_ptr(cb_gamma);

        // We want this data to appear as the first row of the tile.
        // This is 32B at the start of the first face, 32B at the start of the second face
        // However we must read at a 64 byte granularity for Blackhole NOC compatibility on DRAM reads
        // So instead of two 32B reads to the correct addresses, we read 64 bytes into the first face here
        // Then later, copy the second set of 32 bytes into the start of the second face
        // L1-L1 NOC transactions only need 16 byte alignment on BH, so this is legal after data is loaded
        // to L1

        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            uint64_t gamma_noc_addr = get_noc_addr(tile_id, gamma);

            // Read the first 64 bytes of the tile into the first face
#ifdef ARCH_BLACKHOLE
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma, NOC_DRAM_READ_ALIGNMENT_BYTES);
            gamma_noc_addr = get_noc_addr(l1_write_addr_gamma + gamma_face_w_bytes);
            noc_async_read_barrier();
#else
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma, gamma_face_w_bytes);
            gamma_noc_addr += gamma_face_w_bytes;
#endif

            // Copy the second set of 32 bytes into the second face
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma + gamma_face_bytes, gamma_face_w_bytes);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, num_cols_tile_gamma_beta);
    }

    if constexpr (fuse_beta) {
        // Just like gamma, we read at a 64 byte granularity for Blackhole NOC compatibility
        // Then copy the second set of 32 bytes into the second face

        constexpr uint32_t beta_tile_bytes = get_tile_size(cb_beta);
        constexpr uint32_t beta_element_bytes = beta_tile_bytes / tt::constants::TILE_HW;
        constexpr uint32_t beta_face_bytes = beta_element_bytes * tt::constants::FACE_HW;
        constexpr uint32_t beta_face_w_bytes = beta_element_bytes * tt::constants::FACE_WIDTH;
        const auto beta = TensorAccessor(beta_args, beta_addr, size);

        cb_reserve_back(cb_beta, num_cols_tile_gamma_beta);
        auto l1_write_addr_beta = get_write_ptr(cb_beta);

        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            uint64_t beta_noc_addr = get_noc_addr(tile_id, beta);

            // Read the first 64 bytes of the tile into the first face
#ifdef ARCH_BLACKHOLE
            noc_async_read(beta_noc_addr, l1_write_addr_beta, NOC_DRAM_READ_ALIGNMENT_BYTES);
            beta_noc_addr = get_noc_addr(l1_write_addr_beta + beta_face_w_bytes);
            noc_async_read_barrier();
#else
            noc_async_read(beta_noc_addr, l1_write_addr_beta, beta_face_w_bytes);
            beta_noc_addr += beta_face_w_bytes;
#endif

            // Copy the second set of 32 bytes into the second face
            noc_async_read(beta_noc_addr, l1_write_addr_beta + beta_face_bytes, beta_face_w_bytes);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, num_cols_tile_gamma_beta);
    }
}
