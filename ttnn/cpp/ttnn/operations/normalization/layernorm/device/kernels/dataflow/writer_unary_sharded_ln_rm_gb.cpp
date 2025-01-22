// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "In writer!!!!!!!!!!!!!!!!!!!!!!    " << ENDL();

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

    // return;

    // constexpr uint32_t block_w = 4;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_gamma);
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

    uint32_t args_idx = 0;
    uint32_t worker_core_read_offset = 0;

    // cb_wait_front(cb_out, block_ht * block_w);
    uint32_t cb_out_read_base_addr = get_read_ptr(cb_out);
    // cb_reserve_back(cb_out_resharded, block_ht*block_w);
    uint32_t cb_out_reshard_write_base_addr = get_write_ptr(cb_out_resharded);

    uint32_t num_tiles_in_write_queue = 0;

    for (uint32_t i = 0; i < num_segments_to_write_back; ++i) {
        uint32_t write_size = segment_args[args_idx++];
        uint32_t storage_core_x = segment_args[args_idx++];
        uint32_t storage_core_y = segment_args[args_idx++];

        uint32_t num_tiles_to_write_in_current_segment = write_size / out_single_tile_size_bytes * block_ht;

        uint32_t worker_core_read_addr = cb_out_read_base_addr + worker_core_read_offset;
        uint32_t local_storage_core_write_addr = cb_out_reshard_write_base_addr;
        if (i == 0) {  // For the first segment we need to add the start offset; the following segments will start at 0
                       // offset
            local_storage_core_write_addr += storage_core_start_offset;
        }

        uint64_t remote_storage_core_write_addr =
            get_noc_addr(storage_core_x, storage_core_y, local_storage_core_write_addr);

        for (uint32_t h = 0; h < block_ht; ++h) {
            for (uint32_t w = 0; w < num_tiles_to_write_in_current_segment; ++w) {
                num_tiles_in_write_queue += 1;
                cb_wait_front(cb_out, num_tiles_in_write_queue);
                noc_async_write(worker_core_read_addr, remote_storage_core_write_addr, out_single_tile_size_bytes);
                worker_core_read_addr += out_single_tile_size_bytes;
                remote_storage_core_write_addr += out_single_tile_size_bytes;
            }
            // noc_async_write(worker_core_read_addr, remote_storage_core_write_addr, write_size);
            worker_core_read_addr += worker_core_stride_w_bytes;
            remote_storage_core_write_addr += storage_core_stride_w_bytes;
        }
        worker_core_read_offset += write_size;

        // noc_async_write_barrier();
        // cb_pop_front(cb_out, num_tiles_to_write);
    }
    noc_async_write_barrier();
    // cb_pop_front(cb_out, block_ht * block_w);
    // cb_push_back(cb_out_resharded, block_ht*block_w);
#endif
}
