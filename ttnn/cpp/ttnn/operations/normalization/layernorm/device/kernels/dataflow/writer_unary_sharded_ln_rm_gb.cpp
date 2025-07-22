// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "reshard_writer.hpp"

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta = get_compile_time_arg_val(2) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t block_w = get_compile_time_arg_val(5);
    constexpr bool FLOAT32_DTYPE_GAMMA = get_compile_time_arg_val(8) == 1;
    constexpr bool FLOAT32_DTYPE_BETA = get_compile_time_arg_val(9) == 1;

    // Reshard writer
    constexpr uint32_t worker_core_stride_w_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t storage_core_stride_w_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t block_ht = get_compile_time_arg_val(12);

    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(5);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(6);

    // Reshard writer
    const uint32_t num_segments_to_write_back = get_arg_val<uint32_t>(7);
    const uint32_t storage_core_start_offset = get_arg_val<uint32_t>(8);
    tt_l1_ptr uint32_t* segment_args = (tt_l1_ptr uint32_t*)(get_arg_addr(9));

    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_resharded = tt::CBIndex::c_17;

    const uint32_t out_single_tile_size_bytes = get_tile_size(cb_out);

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

    constexpr bool stick_size_is_pow2 = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t size = get_compile_time_arg_val(7);

    if constexpr (fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);

        const auto gamma = get_interleaved_addr_gen<gamma_is_dram, stick_size_is_pow2>(gamma_addr, size);

        constexpr uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE_GAMMA ? 64 : 32;
        constexpr uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE_GAMMA ? 1024 : 512;

        uint32_t l1_write_addr_gamma = get_write_ptr(cb_gamma);
        cb_reserve_back(cb_gamma, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            uint64_t gamma_noc_addr = get_noc_addr(tile_id, gamma);
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma, mask_read_tile_face_bytes * 2);
            gamma_noc_addr = get_noc_addr(l1_write_addr_gamma + mask_read_tile_face_bytes);
            noc_async_read_barrier();
            noc_async_read(
                gamma_noc_addr, l1_write_addr_gamma + mask_read_tile_offset_bytes, mask_read_tile_face_bytes);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, block_w);
    }

    if constexpr (fuse_beta) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);

        const auto beta = get_interleaved_addr_gen<beta_is_dram, stick_size_is_pow2>(beta_addr, size);

        uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE_BETA ? 64 : 32;
        uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE_BETA ? 1024 : 512;

        uint32_t l1_write_addr_beta = get_write_ptr(cb_beta);
        cb_reserve_back(cb_beta, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            uint64_t beta_noc_addr = get_noc_addr(tile_id, beta);
            noc_async_read(beta_noc_addr, l1_write_addr_beta, mask_read_tile_face_bytes * 2);
            beta_noc_addr = get_noc_addr(l1_write_addr_beta + mask_read_tile_face_bytes);
            noc_async_read_barrier();
            noc_async_read(beta_noc_addr, l1_write_addr_beta + mask_read_tile_offset_bytes, mask_read_tile_face_bytes);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, block_w);
    }

#ifndef SKIP_WRITE_BACK
    write_resharded_data(
        cb_out,
        cb_out_resharded,
        num_segments_to_write_back,
        storage_core_start_offset,
        segment_args,
        worker_core_stride_w_bytes,
        storage_core_stride_w_bytes,
        block_ht);
#endif
}
