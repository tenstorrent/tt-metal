// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TurboQuant SDPA decode reader kernel.
//
// Reads Q (BF16) to c_0, K/V indices (BFP4) to c_10/c_12, K/V norms to c_11/c_13.
// K indices are read NOT transposed (compute handles transpose after dequant).
// Supports both contiguous (non-paged) and paged KV cache layouts.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t Skt = get_compile_time_arg_val(4);
    constexpr uint32_t DHt = get_compile_time_arg_val(5);
    constexpr uint32_t vDHt = get_compile_time_arg_val(6);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);
    constexpr bool pre_rescaled = get_compile_time_arg_val(11) == 1;
    constexpr bool is_paged_attention = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t block_size_t = get_compile_time_arg_val(13);  // block_size / TILE_HEIGHT
    // Hybrid plumbing (Phase 2): when recent_window > 0, the reader has
    // ring_K / ring_V / ring_page_table accessors available for the per-chunk
    // source branch. Phase 2 just parses them — Phase 3 wires the branch.
    [[maybe_unused]] constexpr uint32_t recent_window = get_compile_time_arg_val(14);
    [[maybe_unused]] constexpr uint32_t ring_W_padded = get_compile_time_arg_val(15);
    [[maybe_unused]] constexpr uint32_t ring_block_size_t = get_compile_time_arg_val(16);

    constexpr auto q_args = TensorAccessorArgs<17>();
    constexpr auto k_idx_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_norms_args = TensorAccessorArgs<k_idx_args.next_compile_time_args_offset()>();
    constexpr auto v_idx_args = TensorAccessorArgs<k_norms_args.next_compile_time_args_offset()>();
    constexpr auto v_norms_args = TensorAccessorArgs<v_idx_args.next_compile_time_args_offset()>();
    constexpr auto cur_pos_args = TensorAccessorArgs<v_norms_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<cur_pos_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto k_ring_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto v_ring_args = TensorAccessorArgs<k_ring_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto ring_page_table_args =
        TensorAccessorArgs<v_ring_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_norms_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_norms_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(argidx++);
    const uint32_t cur_pos_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t cur_pos_stick_size = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    // Tier 2A Phase 2.3 — chunk-slice routing args (must match compute kernel's
    // [7] and [8] slot semantics). Currently the program factory sends (0, 1)
    // so this reader reads the full [0, valid_k_chunks) range as before.
    const uint32_t core_idx_in_group_arg = get_arg_val<uint32_t>(argidx++);
    const uint32_t cores_per_head_arg = get_arg_val<uint32_t>(argidx++);
    // Hybrid runtime addresses. In legacy mode these alias the TQ tensors and
    // are unread; in hybrid mode they point at the ring buffers.
    [[maybe_unused]] const uint32_t k_ring_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const uint32_t v_ring_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const uint32_t ring_page_table_addr = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    // When pre_rescaled: push KV directly to sdpa's native CBs (c_1/c_2).
    // sdpa_standard handles the data format natively (BFP8 or BFP4).
    // Reader pushes 1 chunk, sdpa consumes it, reader pushes next — pipelined.
    constexpr uint32_t cb_k_idx = pre_rescaled ? tt::CBIndex::c_1 : tt::CBIndex::c_10;
    constexpr uint32_t cb_k_norms = tt::CBIndex::c_11;
    constexpr uint32_t cb_v_idx = pre_rescaled ? tt::CBIndex::c_2 : tt::CBIndex::c_12;
    constexpr uint32_t cb_v_norms = tt::CBIndex::c_13;
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_page_table = tt::CBIndex::c_9;
    // Hybrid mode (recent_window > 0): ring K / V land here. Allocated as
    // ring-tile-format CBs by the program factory, sized for one chunk.
    // Forced num_cores_per_head == 1 in hybrid mode means c_18/c_19 are
    // free of their Tier-2A use.
    constexpr uint32_t cb_k_ring = tt::CBIndex::c_18;
    constexpr uint32_t cb_v_ring = tt::CBIndex::c_19;
    constexpr bool hybrid_mode = (recent_window > 0);

    // ── Load cur_pos tensor into cb_cur_pos (one-shot at kernel start) ──
    // Both reader (this kernel) and compute use cur_pos to derive a per-batch
    // valid_k_chunks bound. Without this, the kernel iterates the whole padded
    // KV cache (e.g. 256 chunks for max_num_blocks=1024) instead of just the
    // chunks containing real data — wasting up to 100x compute at small seqs.
    cb_reserve_back(cb_cur_pos, 1);
    const uint32_t cur_pos_cb_wr_ptr = get_write_ptr(cb_cur_pos);
    const auto cur_pos_reader = TensorAccessor(cur_pos_args, cur_pos_addr, cur_pos_stick_size);
    noc_async_read(cur_pos_reader.get_noc_addr(0), cur_pos_cb_wr_ptr, cur_pos_stick_size);
    noc_async_read_barrier();
    cb_push_back(cb_cur_pos, 1);
    volatile tt_l1_ptr uint32_t* cur_pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cur_pos_cb_wr_ptr);

    // Sk_chunk in TOKENS = Sk_chunk_t tiles × 32 rows/tile.
    constexpr uint32_t k_chunk_size_tokens = Sk_chunk_t * 32;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_idx_tile_bytes = get_tile_size(cb_k_idx);
    constexpr uint32_t k_norms_tile_bytes = get_tile_size(cb_k_norms);
    constexpr uint32_t v_idx_tile_bytes = get_tile_size(cb_v_idx);
    constexpr uint32_t v_norms_tile_bytes = get_tile_size(cb_v_norms);
    [[maybe_unused]] const uint32_t k_ring_tile_bytes = hybrid_mode ? get_tile_size(cb_k_ring) : 0;
    [[maybe_unused]] const uint32_t v_ring_tile_bytes = hybrid_mode ? get_tile_size(cb_v_ring) : 0;

    constexpr uint32_t q_heads_per_kv = NQH / NKH;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_idx_reader = TensorAccessor(k_idx_args, k_idx_addr, k_idx_tile_bytes);
    const auto k_norms_reader = TensorAccessor(k_norms_args, k_norms_addr, k_norms_tile_bytes);
    const auto v_idx_reader = TensorAccessor(v_idx_args, v_idx_addr, v_idx_tile_bytes);
    const auto v_norms_reader = TensorAccessor(v_norms_args, v_norms_addr, v_norms_tile_bytes);
    [[maybe_unused]] const auto k_ring_reader = TensorAccessor(k_ring_args, k_ring_addr, k_ring_tile_bytes);
    [[maybe_unused]] const auto v_ring_reader = TensorAccessor(v_ring_args, v_ring_addr, v_ring_tile_bytes);

    const auto q_tile_shape = TensorTileShape(B, NQH, Sqt, DHt);
    const auto k_idx_tile_shape = TensorTileShape(B, NKH, Skt, DHt);
    const auto v_idx_tile_shape = TensorTileShape(B, NKH, Skt, vDHt);
    const auto k_norms_tile_shape = TensorTileShape(B, NKH, Skt, 1);
    const auto v_norms_tile_shape = TensorTileShape(B, NKH, Skt, 1);

    // ── Identity ring page table on stack (RISC-V scratchpad lives in L1, so
    // a stack array can serve as an L1 page-table buffer for
    // read_paged_chunk_with_padding). Avoids allocating a CB just to hold
    // [0, 1, 2, ...]. Sized for a small fixed maximum; hybrid mode validates
    // ring_W_padded against this implicitly via Sk_chunk_t * (#ring chunks).
    constexpr uint32_t MAX_RING_BLOCKS_T = 16;  // 16 * block_size = up to ~4096 ring tokens
    uint32_t identity_ring_pt[MAX_RING_BLOCKS_T];
    if constexpr (hybrid_mode) {
        for (uint32_t i = 0; i < MAX_RING_BLOCKS_T; ++i) {
            identity_ring_pt[i] = i;
        }
    }
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* ring_page_table_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(identity_ring_pt);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        // Load page table for this batch (paged mode only).
        volatile tt_l1_ptr uint32_t* page_table_ptr = nullptr;
        if constexpr (is_paged_attention) {
            page_table_ptr =
                read_page_table_for_batch(cb_page_table, nb, page_table_args, page_table_addr, page_table_page_size);
        }

        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            // Q (always contiguous)
            {
                const uint32_t q_start = q_tile_shape.id_of(nb, nq, 0, 0);
                read_chunk_with_padding<q_tile_bytes>(
                    q_reader, cb_q_in, q_start, Sq_chunk_t, DHt, Sq_chunk_t, DHt, barrier_threshold);
            }

            const uint32_t k_head = nq / q_heads_per_kv;

            // Limit k_chunks to those that actually contain valid (filled) data.
            // valid_k_chunks = ceil((cur_pos[nb] + 1) / k_chunk_size_tokens), capped by k_num_chunks.
            const uint32_t cur_pos_nb = cur_pos_ptr[nb];
            const uint32_t valid_k_chunks_raw = (cur_pos_nb + k_chunk_size_tokens) / k_chunk_size_tokens;
            const uint32_t valid_k_chunks = valid_k_chunks_raw < k_num_chunks ? valid_k_chunks_raw : k_num_chunks;

            // Tier 2A Phase 2.3: each core in a (B, NQH) group reads only its
            // chunk slice. Bounds derive from the same args the compute kernel
            // uses (slots [7] / [8]). With cores_per_head_arg = 1 the slice is
            // [0, valid_k_chunks) — same as before.
            //
            // Hybrid (Option A) — TQ-full + ring-overlay semantics. The chunk
            // loop runs `valid_k_chunks + actual_W_chunks` iterations:
            //   - chunks [0, valid_k_chunks)              → TQ source (full
            //     range [0, cur_pos], same as legacy single TQ SDPA call).
            //   - chunks [valid_k_chunks, +actual_W_chunks) → ring source
            //     (most recent W positions, mapped to ring chunk index
            //     `k_chunk - valid_k_chunks`).
            // Online softmax accumulates across both; recent positions are
            // counted twice (once via TQ-quantized, once via ring-precise),
            // matching the LSE-emphasis-blend math the legacy dual-call
            // hybrid_sdpa_decode used to produce. This is the math Llama's
            // downstream weights expect — disjoint coverage (Phase 3b's
            // earlier "Option B") drifted enough across 32 layers to wipe
            // out token accuracy in eval_token_accuracy.py.
            constexpr uint32_t actual_W_chunks = hybrid_mode ? (ring_W_padded / k_chunk_size_tokens) : 0;
            const uint32_t total_chunks_to_read = valid_k_chunks + actual_W_chunks;
            const uint32_t chunks_per_worker_r = (total_chunks_to_read + cores_per_head_arg - 1) / cores_per_head_arg;
            const uint32_t k_chunk_start_r = core_idx_in_group_arg * chunks_per_worker_r;
            const uint32_t k_chunk_end_r = (k_chunk_start_r + chunks_per_worker_r < total_chunks_to_read)
                                               ? k_chunk_start_r + chunks_per_worker_r
                                               : total_chunks_to_read;

            for (uint32_t k_chunk = k_chunk_start_r; k_chunk < k_chunk_end_r; ++k_chunk) {
                const uint32_t chunk_start_row = k_chunk * Sk_chunk_t;
                const uint32_t chunk_end_row =
                    (chunk_start_row + Sk_chunk_t < Skt) ? chunk_start_row + Sk_chunk_t : Skt;
                const uint32_t kv_row_count = chunk_end_row - chunk_start_row;

                // ── Ring chunk: read from BFP8 ring into c_18 / c_19 and
                //    skip the TQ idx+norms reads. K transposed (matches the
                //    pre_rescaled layout the compute matmul expects); V not
                //    transposed.
                if (hybrid_mode && k_chunk >= valid_k_chunks) {
                    const uint32_t ring_chunk_idx = k_chunk - valid_k_chunks;
                    const uint32_t ring_start_row = ring_chunk_idx * Sk_chunk_t;
                    // ring chunks always span a full Sk_chunk_t (no boundary
                    // since ring_W_padded is chunk-aligned).
                    read_paged_chunk_with_padding<NKH, ring_block_size_t, DHt>(
                        k_ring_reader,
                        cb_k_ring,
                        k_head,
                        ring_start_row,
                        Sk_chunk_t,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        k_ring_tile_bytes,
                        barrier_threshold,
                        ring_page_table_ptr,
                        true /*transpose K*/);
                    read_paged_chunk_with_padding<NKH, ring_block_size_t, vDHt>(
                        v_ring_reader,
                        cb_v_ring,
                        k_head,
                        ring_start_row,
                        Sk_chunk_t,
                        vDHt,
                        Sk_chunk_t,
                        vDHt,
                        v_ring_tile_bytes,
                        barrier_threshold,
                        ring_page_table_ptr,
                        false /*not transposed*/);
                    continue;
                }

                // K indices: transposed when pre_rescaled (compute skips dequant),
                // NOT transposed otherwise (compute transposes after dequant).
                if constexpr (is_paged_attention) {
                    read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                        k_idx_reader,
                        cb_k_idx,
                        k_head,
                        chunk_start_row,
                        kv_row_count,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        k_idx_tile_bytes,
                        barrier_threshold,
                        page_table_ptr,
                        pre_rescaled /*transpose*/);
                } else {
                    const uint32_t k_start = k_idx_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                    read_chunk_with_padding<k_idx_tile_bytes>(
                        k_idx_reader,
                        cb_k_idx,
                        k_start,
                        kv_row_count,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        barrier_threshold,
                        pre_rescaled);
                }

                // K norms (skip when pre_rescaled — values already include norms)
                if constexpr (!pre_rescaled) {
                    if constexpr (is_paged_attention) {
                        read_paged_chunk_with_padding<NKH, block_size_t, 1>(
                            k_norms_reader,
                            cb_k_norms,
                            k_head,
                            chunk_start_row,
                            kv_row_count,
                            1,
                            Sk_chunk_t,
                            1,
                            k_norms_tile_bytes,
                            barrier_threshold,
                            page_table_ptr,
                            false /*transpose*/);
                    } else {
                        const uint32_t n_start = k_norms_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                        read_chunk_with_padding<k_norms_tile_bytes>(
                            k_norms_reader, cb_k_norms, n_start, kv_row_count, 1, Sk_chunk_t, 1, barrier_threshold);
                    }
                }

                // V indices: NOT transposed
                if constexpr (is_paged_attention) {
                    read_paged_chunk_with_padding<NKH, block_size_t, vDHt>(
                        v_idx_reader,
                        cb_v_idx,
                        k_head,
                        chunk_start_row,
                        kv_row_count,
                        vDHt,
                        Sk_chunk_t,
                        vDHt,
                        v_idx_tile_bytes,
                        barrier_threshold,
                        page_table_ptr,
                        false /*transpose*/);
                } else {
                    const uint32_t v_start = v_idx_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                    read_chunk_with_padding<v_idx_tile_bytes>(
                        v_idx_reader, cb_v_idx, v_start, kv_row_count, vDHt, Sk_chunk_t, vDHt, barrier_threshold);
                }

                // V norms (skip when pre_rescaled)
                if constexpr (!pre_rescaled) {
                    if constexpr (is_paged_attention) {
                        read_paged_chunk_with_padding<NKH, block_size_t, 1>(
                            v_norms_reader,
                            cb_v_norms,
                            k_head,
                            chunk_start_row,
                            kv_row_count,
                            1,
                            Sk_chunk_t,
                            1,
                            v_norms_tile_bytes,
                            barrier_threshold,
                            page_table_ptr,
                            false /*transpose*/);
                    } else {
                        const uint32_t n_start = v_norms_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                        read_chunk_with_padding<v_norms_tile_bytes>(
                            v_norms_reader, cb_v_norms, n_start, kv_row_count, 1, Sk_chunk_t, 1, barrier_threshold);
                    }
                }
            }
        }
    }
}
