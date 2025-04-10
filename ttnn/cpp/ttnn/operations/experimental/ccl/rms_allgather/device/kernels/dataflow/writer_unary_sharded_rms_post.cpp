// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "reshard_writer.hpp"

void kernel_main() {
    // Run Parameters CTs
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t block_w = get_compile_time_arg_val(3);

    // Circular Buffer CTs
    constexpr uint32_t cb_out_resharded = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t eps_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_in_4 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(8);

    // Data type CTs
    constexpr uint32_t stick_size = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE_GAMMA = get_compile_time_arg_val(10) == 1;

    // Reshard writer
    constexpr uint32_t worker_core_stride_w_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t storage_core_stride_w_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t block_ht = 1;

    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(4);

    // Reshard writer
#ifndef SKIP_WRITE_BACK
    const uint32_t num_segments_to_write_back = get_arg_val<uint32_t>(5);
    const uint32_t storage_core_start_offset = get_arg_val<uint32_t>(6);
    tt_l1_ptr uint32_t* segment_args = (tt_l1_ptr uint32_t*)(get_arg_addr(7));
#endif

    if constexpr (is_all_to_all_worker) {
        const uint32_t scalar_c = get_arg_val<uint32_t>(0);
        wh_generate_reduce_scaler<true>(cb_in_4, scalar_c);
    }

    const uint32_t out_single_tile_size_bytes = get_tile_size(cb_out);
    const uint32_t eps = get_arg_val<uint32_t>(2);
    generate_bcast_col_scalar(eps_cb_id, eps);

    if constexpr (fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const auto gamma = get_interleaved_addr_gen<gamma_is_dram, stick_size>(gamma_addr);

        constexpr uint32_t bytes_in_faceline = FLOAT32_DTYPE_GAMMA ? 64 : 32;
        constexpr uint32_t bytes_in_two_facelines = bytes_in_faceline * 2;
        constexpr uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE_GAMMA ? 1024 : 512;

        uint32_t l1_write_addr_gamma = get_write_ptr(cb_gamma);
        cb_reserve_back(cb_gamma, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            uint64_t gamma_noc_addr = get_noc_addr(tile_id, gamma);
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma, bytes_in_two_facelines);
            gamma_noc_addr = get_noc_addr(l1_write_addr_gamma + bytes_in_faceline);
            noc_async_read_barrier();  // might be faster to do two separate read instead of barrier.
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma + mask_read_tile_offset_bytes, bytes_in_faceline);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, block_w);
    }

#ifndef SKIP_WRITE_BACK
    write_minimal_resharded_data<cb_out, cb_out_resharded, worker_core_stride_w_bytes, storage_core_stride_w_bytes>(
        num_segments_to_write_back, storage_core_start_offset, segment_args);
#endif
}
