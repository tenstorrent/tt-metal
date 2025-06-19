// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(8);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(9);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(10);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(11);
    constexpr uint32_t scale_val = get_compile_time_arg_val(12);
    constexpr uint32_t num_cores = get_compile_time_arg_val(13);
    constexpr uint32_t is_causal = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(17) == 1;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);
    const uint32_t chunk_start_t_in_q_chunks = get_arg_val<uint32_t>(8);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = data_format};

    const auto out_tile_shape = TensorTileShape(B, NQH, valid_Sqt, DHt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

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

                if constexpr (is_causal) {
                    uint32_t offset_q_chunk = q_chunk;
                    if constexpr (is_chunked) {
                        // Bump it up to the chunk start
                        offset_q_chunk += chunk_start_t_in_q_chunks;
                    }
                    uint32_t q_low_idx =
                        offset_q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
                    uint32_t q_high_idx = q_low_idx + Sq_chunk_t;

                    for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                        const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                        const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
                        // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                        // Q-range = [q_low, q_high)
                        // K-range = [k_low, k_high)
                        // does_overlap = not (q_low >= k_high or k_low >= q_high)
                        // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                        // Read mask chunk
                        if (!(q_low_idx >= k_high_idx)) {
                            generate_causal_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, offset_q_chunk, k_chunk);
                        }
                    }
                } else if constexpr (use_padded_mask) {
                    // Generate non-causal padded mask only once per q chunk since it is only non-zero on the last K
                    // chunk if it exists at all.
                    generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, unpadded_Sk);
                }

                // Wait for compute to deliver output chunk
                /*
                Determine how many rows of OUT will be written. Both start and end rows are
                capped by valid_Sqt, since Sq padding is independent of Sk padding.
                */
                const uint32_t out_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
                const uint32_t out_row_end_tile = std::min(out_row_start_tile + Sq_chunk_t, valid_Sqt);
                const uint32_t out_row_tile_count = out_row_end_tile - out_row_start_tile;
                uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, out_row_start_tile, 0);

                cb_wait_front(cb_out, out_chunk_tiles);
                barrier_count = 0;
                uint32_t l1_read_addr = get_read_ptr(cb_out);
                for (uint32_t row = 0; row < out_row_tile_count; ++row) {
                    for (uint32_t col = 0; col < DHt; ++col) {
                        noc_async_write_tile(out_tile_id, out_writer, l1_read_addr);
                        ++out_tile_id;
                        l1_read_addr += tile_bytes;

                        if (++barrier_count == barrier_threshold) {
                            noc_async_writes_flushed();
                            barrier_count = 0;
                        }
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, out_chunk_tiles);
            }
        }
    }
}
