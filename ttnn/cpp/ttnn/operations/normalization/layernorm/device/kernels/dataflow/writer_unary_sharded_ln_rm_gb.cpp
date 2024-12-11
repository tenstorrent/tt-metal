// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        // Note: TileSlice has different parameters on reader/writer kernels and these are not quite accurate. That may
        // cause issues in terms of where data appears. But trying proper parameters caused errors in the dprint server.
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}
void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta = get_compile_time_arg_val(2) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t block_w = get_compile_time_arg_val(5);
    constexpr bool FLOAT32_DTYPE_GAMMA = get_compile_time_arg_val(8) == 1;
    constexpr bool FLOAT32_DTYPE_BETA = get_compile_time_arg_val(9) == 1;

    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(5);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;

    // constexpr uint32_t block_w = 4;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_gamma);

    {
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        const uint32_t scalar_w = get_arg_val<uint32_t>(1);
        generate_reduce_scaler(cb_in_2, scalar_w);
    }
    if constexpr (is_all_to_all_worker) {
        constexpr uint32_t cb_in_4 = tt::CBIndex::c_4;
        const uint32_t scalar_c = get_arg_val<uint32_t>(0);
        generate_reduce_scaler(cb_in_4, scalar_c);
    }
    constexpr uint32_t eps_cb_id = 3;
    const uint32_t eps = get_arg_val<uint32_t>(2);
    generate_bcast_col_scalar(eps_cb_id, eps);

#define stick_size_is_pow2 get_compile_time_arg_val(6) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(7);
#else
    constexpr uint32_t page_size = get_compile_time_arg_val(7);
#endif

    if constexpr (fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
#if (stick_size_is_pow2)
        const InterleavedPow2AddrGen<gamma_is_dram> gamma = {
            .bank_base_address = gamma_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
        const InterleavedAddrGen<gamma_is_dram> gamma = {.bank_base_address = gamma_addr, .page_size = page_size};
#endif

        uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE_GAMMA ? 64 : 32;
        uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE_GAMMA ? 1024 : 512;

        uint32_t l1_write_addr_gamma = get_write_ptr(cb_gamma);
        cb_reserve_back(cb_gamma, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            uint64_t gamma_noc_addr = get_noc_addr(tile_id, gamma);
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma, mask_read_tile_face_bytes);
            gamma_noc_addr += mask_read_tile_face_bytes;
            noc_async_read(
                gamma_noc_addr, l1_write_addr_gamma + mask_read_tile_offset_bytes, mask_read_tile_face_bytes);
            l1_write_addr_gamma += gamma_tile_bytes;
            // DPRINT << "cb_gamma writer:" << ENDL();
            // DPRINT_DATA1(( print_full_tile(cb_gamma, 0, true) ));
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, block_w);
    }

    if constexpr (fuse_beta) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
#if (stick_size_is_pow2)
        const InterleavedPow2AddrGen<beta_is_dram> beta = {
            .bank_base_address = beta_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
        const InterleavedAddrGen<beta_is_dram> beta = {.bank_base_address = beta_addr, .page_size = page_size};
#endif

        uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE_BETA ? 64 : 32;
        uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE_BETA ? 1024 : 512;

        uint32_t l1_write_addr_beta = get_write_ptr(cb_beta);
        cb_reserve_back(cb_beta, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            uint64_t beta_noc_addr = get_noc_addr(tile_id, beta);
            noc_async_read(beta_noc_addr, l1_write_addr_beta, mask_read_tile_face_bytes);
            beta_noc_addr += mask_read_tile_face_bytes;
            noc_async_read(beta_noc_addr, l1_write_addr_beta + mask_read_tile_offset_bytes, mask_read_tile_face_bytes);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, block_w);
    }
}
