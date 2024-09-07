// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

template<uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main() {
    constexpr uint32_t NKH = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t _unused1 = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8); // num chunks in valid_seq_len

    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(9);
    constexpr uint32_t scale_val = get_compile_time_arg_val(10);
    constexpr uint32_t num_cores = get_compile_time_arg_val(11);

    const uint32_t out_addr  = get_arg_val<uint32_t>(0);
    const uint32_t core_id    = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_out = tt::CB::c_out0;

    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    uint32_t out_tile_id = 0;

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                // Wait for compute to deliver output chunk
                cb_wait_front(cb_out, out_chunk_tiles);
                cb_pop_front(cb_out, out_chunk_tiles);
            }
        }
    }
}
