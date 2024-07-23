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
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8);
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

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CB::c_out0;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    uint32_t out_tile_id = 0;

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        const uint32_t q_batch_offset = nb * NQH * St * DHt;
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk;
                #if defined BALANCED_Q_PARALLEL
                uint32_t q_chunk_div_2 = q_chunks_per_core / 2;
                if (q_iter < q_chunk_div_2) { // bottom half
                    q_chunk = local_q_start + q_iter;
                } else {
                    uint32_t back_q_iter = q_iter - q_chunk_div_2; // Back half should start at 0
                    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
                }
                #else
                q_chunk = local_q_start + q_iter;
                #endif

                uint32_t q_head_offset = nq * St * DHt;
                uint32_t q_chunk_offset = q_chunk * Sq_chunk_t * DHt;
                out_tile_id = q_batch_offset + q_head_offset + q_chunk_offset;

                // Wait for compute to deliver output chunk
                cb_wait_front(cb_out, out_chunk_tiles);
                barrier_count = 0;
                uint32_t l1_read_addr = get_read_ptr(cb_out);
                for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
                    noc_async_write_tile(out_tile_id, out_writer, l1_read_addr);
                    ++out_tile_id;
                    l1_read_addr += tile_bytes;

                    if (++barrier_count == barrier_threshold) {
                        noc_async_writes_flushed();
                        barrier_count = 0;
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, out_chunk_tiles);
            }
        }
    }
}
