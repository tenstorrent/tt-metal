// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t Skt = get_compile_time_arg_val(4);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(5);
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(6);
    constexpr uint32_t DHt = get_compile_time_arg_val(7);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores = get_compile_time_arg_val(12);
    constexpr uint32_t is_causal = get_compile_time_arg_val(13) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t page_table_is_dram = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t block_size_t = get_compile_time_arg_val(18);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(19);

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunked_q_chunk_offset = get_arg_val<uint32_t>(argidx++);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    // When chunked, update the bounds of valid K sequence length based on Q chunk offset
    uint32_t valid_Skt_bound = valid_Skt;
    if constexpr (is_chunked) {
        valid_Skt_bound += chunked_q_chunk_offset * Sq_chunk_t;
    }

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;

    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_6;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr DataFormat mask_data_format = get_dataformat(cb_mask_in);

    constexpr uint32_t q_heads_per_kv = NQH / NKH;

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const InterleavedAddrGenFast<is_dram> q_reader = {
        .bank_base_address = q_addr, .page_size = q_tile_bytes, .data_format = q_data_format};

    const InterleavedAddrGenFast<is_dram> k_reader = {
        .bank_base_address = k_addr, .page_size = k_tile_bytes, .data_format = k_data_format};

    const InterleavedAddrGenFast<is_dram> v_reader = {
        .bank_base_address = v_addr, .page_size = v_tile_bytes, .data_format = v_data_format};

    const InterleavedAddrGenFast<is_dram> mask_reader = {
        .bank_base_address = mask_addr, .page_size = mask_tile_bytes, .data_format = mask_data_format};

    const auto q_tile_shape = TensorTileShape(B, NQH, valid_Sqt, DHt);
    const auto k_tile_shape = TensorTileShape(B, NKH, valid_Skt, DHt);
    const auto v_tile_shape = TensorTileShape(B, NKH, valid_Skt, DHt);
    const auto mask_tile_shape = TensorTileShape(B, 1, valid_Sqt, valid_Skt);

    volatile tt_l1_ptr uint32_t* page_table_ptr;

    uint32_t v_tile_id = 0;
    uint32_t mask_tile_id = 0;
    uint32_t barrier_count = 0;

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        if constexpr (is_chunked) {
            // Chunked means that we have paged attention
            const InterleavedAddrGen<page_table_is_dram> page_table_gen = {
                .bank_base_address = page_table_addr, .page_size = page_table_stick_size};
            cb_reserve_back(cb_id_page_table, 1);
            uint32_t page_table_cb_wr_ptr = get_write_ptr(cb_id_page_table);
            uint64_t page_table_noc_addr = get_noc_addr(nb, page_table_gen);
            noc_async_read(page_table_noc_addr, page_table_cb_wr_ptr, page_table_stick_size);
            noc_async_read_barrier();
            cb_push_back(cb_id_page_table, 1);
            page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);
        }

        const uint32_t mask_batch_offset = nb * Sqt * Skt;
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                /*
                Read a chunk of Q. BALANCED_Q_PARALLEL evenly distributes Q chunks
                across cores when causal and other conditions are met.
                When chunked, we must treat Q as offset by some factor.
                When causal, we set up the bounds such that we only read the lower triangle of K and V.
                When non-causal, read all of K and V.
                */
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
                /*
                Determine how many rows of Q will be read. Both start and end rows are
                capped by valid_Sqt, since Sq padding is independent of Sk padding.
                */
                const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
                const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
                const uint32_t q_row_tile_count = q_row_end_tile - q_row_start_tile;
                const uint32_t q_tile_id = q_tile_shape.id_of(nb, nq, q_row_start_tile, 0);

                read_chunk_with_padding<is_dram, q_tile_bytes>(
                    q_reader, cb_q_in, q_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);

                if constexpr (is_chunked) {
                    q_chunk = chunked_q_chunk_offset + q_chunk;
                }
                uint32_t q_low_idx =
                    q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
                uint32_t q_high_idx;
                if constexpr (is_causal) {
                    q_high_idx = q_low_idx + Sq_chunk_t;
                } else {
                    q_high_idx = Skt;
                }

                const uint32_t kv_head = nq / q_heads_per_kv;

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;

                    const uint32_t k_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt_bound);
                    const uint32_t k_row_end_tile = std::min(k_row_start_tile + Sk_chunk_t, valid_Skt_bound);
                    const uint32_t k_row_tile_count = k_row_end_tile - k_row_start_tile;
                    const uint32_t k_start_tile_id = k_tile_shape.id_of(nb, kv_head, k_row_start_tile, 0);

                    if constexpr (is_chunked) {
                        // Use page table to read K chunk
                        const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
                        read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                            k_reader,
                            cb_k_in,
                            kv_head,
                            k_chunk_start_row_num,
                            k_row_tile_count,
                            DHt,
                            Sk_chunk_t,
                            DHt,
                            k_tile_bytes,
                            barrier_threshold,
                            page_table_ptr,
                            true  // transpose=true for K reads
                        );
                    } else {
                        read_chunk_with_padding<is_dram, k_tile_bytes>(
                            k_reader,
                            cb_k_in,
                            k_start_tile_id,
                            k_row_tile_count,
                            DHt,
                            Sk_chunk_t,
                            DHt,
                            barrier_threshold,
                            true  // transpose=true for K reads
                        );
                    }

                    if constexpr (use_provided_mask) {
                        // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                        // Q-range = [q_low, q_high)
                        // K-range = [k_low, k_high)
                        // does_overlap = not (q_low >= k_high or k_low >= q_high)
                        // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                        // Read mask chunk
                        // When a mask is provided, there will be no padding on q or kv.
                        cb_reserve_back(cb_mask_in, mask_chunk_tiles);
                        uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
                        barrier_count = 0;
                        mask_tile_id = mask_batch_offset + q_chunk * Sq_chunk_t * Skt /*row_offset*/ + k_chunk * Sk_chunk_t /*col_offset*/;
                        for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                            for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                                noc_async_read_tile(mask_tile_id, mask_reader, mask_write_ptr);
                                mask_tile_id += 1;
                                mask_write_ptr += mask_tile_bytes;
                                if (++barrier_count == barrier_threshold) {
                                    noc_async_read_barrier();
                                    barrier_count = 0;
                                }
                            }
                            // Strid along columns to get to next row
                            mask_tile_id -= Sk_chunk_t;
                            mask_tile_id += Skt;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_mask_in, mask_chunk_tiles);
                    }

                    if constexpr (is_chunked) {
                        // Use page table to read V chunk
                        const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
                        read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                            v_reader,
                            cb_v_in,
                            kv_head,
                            k_chunk_start_row_num,
                            k_row_tile_count,
                            DHt,
                            Sk_chunk_t,
                            DHt,
                            v_tile_bytes,
                            barrier_threshold,
                            page_table_ptr,
                            false);
                    } else {
                        read_chunk_with_padding<is_dram, v_tile_bytes>(
                            v_reader,
                            cb_v_in,
                            k_start_tile_id,
                            k_row_tile_count,
                            DHt,
                            Sk_chunk_t,
                            DHt,
                            barrier_threshold,
                            false);
                    }
                }
            }
        }

        if constexpr (is_chunked) {
            cb_pop_front(cb_id_page_table, 1);
        }
    }
}
