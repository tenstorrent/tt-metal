// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(4);
    constexpr uint32_t unpadded_Sk = get_compile_time_arg_val(5);
    constexpr uint32_t DHt = get_compile_time_arg_val(6);
    constexpr uint32_t vDHt = get_compile_time_arg_val(7);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(12);
    constexpr uint32_t scale_val = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t is_causal = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(19);

    constexpr auto out_args = TensorAccessorArgs<20>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);
    const uint32_t num_phases = get_arg_val<uint32_t>(8);
    const uint32_t chunk_start_t_in_q_chunks_phase_1 = get_arg_val<uint32_t>(9);
    const uint32_t write_offset_phase_1 = get_arg_val<uint32_t>(10);
    uint32_t chunk_start_t_in_q_chunks_phase_2 = 0;
    uint32_t write_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunk_start_t_in_q_chunks_phase_2 = get_arg_val<uint32_t>(11);
        write_offset_phase_2 = get_arg_val<uint32_t>(12);
    }

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);

    const auto out_tile_shape = TensorTileShape(B, NQH, valid_Sqt, vDHt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    uint32_t chunk_start_t_in_q_chunks = 0;
    uint32_t write_offset = 0;
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_1;
            write_offset = write_offset_phase_1;
        } else {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_2;
            write_offset = write_offset_phase_2;
        }
        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
            const uint32_t q_batch_offset = nb * NQH * Sqt * DHt;
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                    uint32_t q_chunk;
#if defined BALANCED_Q_PARALLEL
                    uint32_t q_chunk_div_2 = q_chunks_per_core / 2;
                    if (q_iter < q_chunk_div_2) {  // bottom half
                        q_chunk = local_q_start + q_iter;
                    } else {
                        uint32_t back_q_iter = q_iter - q_chunk_div_2;  // Back half should start at 0
                        q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
                    }
#else
                    q_chunk = local_q_start + q_iter;
#endif

                    generate_mask<use_padded_mask, is_causal, false, is_chunked, sliding_window_size, cb_mask_in>(
                        Sq_chunk_t, Sk_chunk_t, q_chunk, chunk_start_t_in_q_chunks, false, false, unpadded_Sk, 0, 0);

                    /*
                      Determine how many rows of OUT will be written. Both start and end rows are
                      capped by valid_Sqt, since Sq padding is independent of Sk padding.
                    */
                    const uint32_t out_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
                    const uint32_t out_row_end_tile = std::min(out_row_start_tile + Sq_chunk_t, valid_Sqt);
                    const uint32_t out_row_tile_count = out_row_end_tile - out_row_start_tile;
                    uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile, 0);
                    write_block(
                        out_writer,
                        cb_out,
                        out_chunk_tiles,
                        out_row_tile_count,
                        vDHt,
                        out_tile_id,
                        tile_bytes,
                        barrier_threshold);
                }
            }
        }
    }
}
