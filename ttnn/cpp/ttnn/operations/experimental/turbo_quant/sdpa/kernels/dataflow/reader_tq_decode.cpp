// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TurboQuant SDPA decode reader kernel.
//
// Reads 5 input tensors for paged SDPA that dequantizes BFP4-quantized KV cache
// indices on-the-fly:
//
//   Q          (BF16, interleaved)            [B, NQH, 1, DH]
//   K indices  (BFP4, paged via page table)   [B, NKH, Sk, DH]   -- read TRANSPOSED
//   K norms    (BF16, interleaved contiguous) [B, NKH, Sk, 1]    -- NOT transposed
//   V indices  (BFP4, paged via page table)   [B, NKH, Sk, vDH]  -- NOT transposed
//   V norms    (BF16, interleaved contiguous) [B, NKH, Sk, 1]    -- NOT transposed
//   Page table (Int32)                        [B, max_pages]
//
// Simplified from the full SDPA reader: no chain forwarding, no attention sink,
// no MLA, no flexible chunked mode, no Q subblock push.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    // ── Compile-time args ──
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);         // Q seq tiles (= 1 for decode)
    constexpr uint32_t Skt = get_compile_time_arg_val(4);         // K seq tiles (total)
    constexpr uint32_t DHt = get_compile_time_arg_val(5);         // Head dim tiles (128/32 = 4)
    constexpr uint32_t vDHt = get_compile_time_arg_val(6);        // V head dim tiles (usually = DHt)
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(7);  // Q chunk tiles (= 1 for decode)
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);  // K chunk size in tiles
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(11);  // Page block size in tiles
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(12);

    // TensorAccessorArgs for each tensor, chained by next_compile_time_args_offset()
    constexpr auto q_args = TensorAccessorArgs<13>();
    constexpr auto k_idx_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_norms_args = TensorAccessorArgs<k_idx_args.next_compile_time_args_offset()>();
    constexpr auto v_idx_args = TensorAccessorArgs<k_norms_args.next_compile_time_args_offset()>();
    constexpr auto v_norms_args = TensorAccessorArgs<v_idx_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<v_norms_args.next_compile_time_args_offset()>();

    // ── Runtime args ──
    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_norms_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_norms_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);

    // ── CB indices ──
    // Must match compute kernel's expectations.
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_idx = tt::CBIndex::c_10;
    constexpr uint32_t cb_k_norms = tt::CBIndex::c_11;
    constexpr uint32_t cb_v_idx = tt::CBIndex::c_12;
    constexpr uint32_t cb_v_norms = tt::CBIndex::c_13;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_6;

    // ── Tile sizes ──
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_idx_tile_bytes = get_tile_size(cb_k_idx);
    constexpr uint32_t k_norms_tile_bytes = get_tile_size(cb_k_norms);
    constexpr uint32_t v_idx_tile_bytes = get_tile_size(cb_v_idx);
    constexpr uint32_t v_norms_tile_bytes = get_tile_size(cb_v_norms);

    // ── Derived constants ──
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t q_heads_per_kv = NQH / NKH;

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    // ── Tensor accessors ──
    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_idx_reader = TensorAccessor(k_idx_args, k_idx_addr, k_idx_tile_bytes);
    const auto k_norms_reader = TensorAccessor(k_norms_args, k_norms_addr, k_norms_tile_bytes);
    const auto v_idx_reader = TensorAccessor(v_idx_args, v_idx_addr, v_idx_tile_bytes);
    const auto v_norms_reader = TensorAccessor(v_norms_args, v_norms_addr, v_norms_tile_bytes);

    // Tile shape helpers for computing flat tile IDs.
    // Q: [B, NQH, Sqt, DHt]
    const auto q_tile_shape = TensorTileShape(B, NQH, Sqt, DHt);
    // Norms: [B, NKH, Skt, 1] -- contiguous, tile-layout, 1 tile wide
    const auto k_norms_tile_shape = TensorTileShape(B, NKH, Skt, 1);
    const auto v_norms_tile_shape = TensorTileShape(B, NKH, Skt, 1);

    volatile tt_l1_ptr uint32_t* page_table_ptr;

    // ── Main loop ──
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        // Read page table for this batch into L1
        cb_reserve_back(cb_id_page_table, 1);
        page_table_ptr =
            read_page_table_for_batch(cb_id_page_table, nb, page_table_args, page_table_addr, page_table_stick_size);
        cb_push_back(cb_id_page_table, 1);

        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            // ── Read Q (single token, 1 x DHt tiles) ──
            {
                const uint32_t q_start_tile_id = q_tile_shape.id_of(nb, nq, 0, 0);
                read_chunk_with_padding<q_tile_bytes>(
                    q_reader,
                    cb_q_in,
                    q_start_tile_id,
                    Sq_chunk_t,  // src_rows (1 for decode)
                    DHt,         // src_cols
                    Sq_chunk_t,  // dst_rows
                    DHt,         // dst_cols
                    barrier_threshold);
            }

            // Determine KV head for GQA mapping
            const uint32_t k_head = nq / q_heads_per_kv;

            // ── Loop over K/V chunks ──
            for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                const uint32_t chunk_start_row = k_chunk * Sk_chunk_t;
                // For the last chunk, kv_row_count may be less than Sk_chunk_t.
                // Compute kernel softmax handles padding via cur_pos masking, so
                // for simplicity we read Sk_chunk_t rows for all chunks except
                // when the total sequence length does not fill the last chunk.
                const uint32_t chunk_end_row =
                    (chunk_start_row + Sk_chunk_t < Skt) ? chunk_start_row + Sk_chunk_t : Skt;
                const uint32_t kv_row_count = chunk_end_row - chunk_start_row;

                // ── K indices: paged read, TRANSPOSED (K^T for QK matmul) ──
                read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                    k_idx_reader,
                    cb_k_idx,
                    k_head,
                    chunk_start_row,
                    kv_row_count,  // src_rows (valid rows)
                    DHt,           // src_cols
                    Sk_chunk_t,    // dst_rows
                    DHt,           // dst_cols
                    k_idx_tile_bytes,
                    barrier_threshold,
                    page_table_ptr,
                    false  // NOT transposed — matmul_blocks handles transpose
                );

                // ── K norms: contiguous read, NOT transposed ──
                // Norms shape: [B, NKH, Skt, 1] -- 1 tile per sequence position.
                // Read kv_row_count tiles, zero-pad the rest to fill Sk_chunk_t.
                {
                    const uint32_t norm_tiles_total = Sk_chunk_t;
                    cb_reserve_back(cb_k_norms, norm_tiles_total);
                    uint32_t norm_write_ptr = get_write_ptr(cb_k_norms);

                    for (uint32_t row = 0; row < kv_row_count; ++row) {
                        const uint32_t seq_tile = chunk_start_row + row;
                        const uint32_t norm_tile_id = k_norms_tile_shape.id_of(nb, k_head, seq_tile, 0);
                        noc_async_read_tile(norm_tile_id, k_norms_reader, norm_write_ptr);
                        norm_write_ptr += k_norms_tile_bytes;
                    }
                    noc_async_read_barrier();

                    // Zero-pad remaining tiles if kv_row_count < Sk_chunk_t
                    for (uint32_t row = kv_row_count; row < Sk_chunk_t; ++row) {
                        fill_tile_zeros<k_norms_tile_bytes, false>(cb_k_norms, row);
                    }
                    noc_async_read_barrier();

                    cb_push_back(cb_k_norms, norm_tiles_total);
                }

                // ── V indices: paged read, NOT transposed ──
                read_paged_chunk_with_padding<NKH, block_size_t, vDHt>(
                    v_idx_reader,
                    cb_v_idx,
                    k_head,
                    chunk_start_row,
                    kv_row_count,  // src_rows
                    vDHt,          // src_cols
                    Sk_chunk_t,    // dst_rows
                    vDHt,          // dst_cols
                    v_idx_tile_bytes,
                    barrier_threshold,
                    page_table_ptr,
                    false  // transpose=false for V reads
                );

                // ── V norms: contiguous read, NOT transposed ──
                {
                    const uint32_t norm_tiles_total = Sk_chunk_t;
                    cb_reserve_back(cb_v_norms, norm_tiles_total);
                    uint32_t norm_write_ptr = get_write_ptr(cb_v_norms);

                    for (uint32_t row = 0; row < kv_row_count; ++row) {
                        const uint32_t seq_tile = chunk_start_row + row;
                        const uint32_t norm_tile_id = v_norms_tile_shape.id_of(nb, k_head, seq_tile, 0);
                        noc_async_read_tile(norm_tile_id, v_norms_reader, norm_write_ptr);
                        norm_write_ptr += v_norms_tile_bytes;
                    }
                    noc_async_read_barrier();

                    // Zero-pad remaining tiles
                    for (uint32_t row = kv_row_count; row < Sk_chunk_t; ++row) {
                        fill_tile_zeros<v_norms_tile_bytes, false>(cb_v_norms, row);
                    }
                    noc_async_read_barrier();

                    cb_push_back(cb_v_norms, norm_tiles_total);
                }
            }
        }

        // Release page table CB for this batch
        cb_pop_front(cb_id_page_table, 1);
    }
}
