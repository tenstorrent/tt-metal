// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "dataflow_common.hpp"

// Fetch a KV chunk into L1 for forwarding. No CB lifecycle — caller manages
// cb_reserve_back / cb_push_back. Single read barrier at end for lower latency.
template <uint32_t tile_bytes, bool transpose, typename ReaderType>
FORCE_INLINE void read_chunk_for_forwarding(
    const ReaderType& reader,
    const uint32_t dst_addr,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t skip_src_cols = 0) {
    const uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    const uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t tile_id = start_tile_id;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = dst_addr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < src_cols; ++col) {
            noc_async_read_tile(tile_id++, reader, write_ptr);
            write_ptr += inner_ptr_stride;
        }
        tile_id += skip_src_cols;
    }
    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_idx = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_zeros_async(dst_addr + tile_idx * tile_bytes, tile_bytes);
        }
    }
    noc_async_read_barrier();
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t NVH = get_compile_time_arg_val(3);
    constexpr uint32_t Sqt = get_compile_time_arg_val(4);
    constexpr uint32_t Skt = get_compile_time_arg_val(5);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(6);
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(7);
    constexpr uint32_t DHt = get_compile_time_arg_val(8);
    constexpr uint32_t vDHt = get_compile_time_arg_val(9);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(12);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t is_causal = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t broadcast_provided_mask_batch = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t broadcast_provided_mask_heads = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t block_size_t = get_compile_time_arg_val(21);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(22);
    constexpr uint32_t use_attention_sink = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t use_mla = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t mla_kv_overlap = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(26);

    constexpr uint32_t sender_semaphore_id = get_compile_time_arg_val(27);
    constexpr uint32_t receiver_semaphore_id = get_compile_time_arg_val(28);
    constexpr uint32_t valid_semaphore_id = get_compile_time_arg_val(29);
    constexpr bool mcast_enabled = get_compile_time_arg_val(30) == 1;

    constexpr auto q_args = TensorAccessorArgs<31>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto chunk_start_idx_args = TensorAccessorArgs<attention_sink_args.next_compile_time_args_offset()>();
    // Ring proxy case (tail CT arg) drives both flat work distribution and the K/Q halving.
    constexpr RingProxyMode proxy_mode =
        static_cast<RingProxyMode>(get_compile_time_arg_val(chunk_start_idx_args.next_compile_time_args_offset()));
    constexpr bool flatten_work = proxy_uses_flat_work(proxy_mode);
    constexpr bool flat_use_zigzag = flatten_work && is_causal && (q_num_chunks % 2 == 0);

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t attention_sink_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunk_start_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_phases = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunked_q_chunk_offset_phase_1 = get_arg_val<uint32_t>(argidx++);
    const uint32_t read_offset_phase_1 = get_arg_val<uint32_t>(argidx++);
    uint32_t chunked_q_chunk_offset_phase_2 = 0;
    uint32_t read_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunked_q_chunk_offset_phase_2 = get_arg_val<uint32_t>(argidx++);
        read_offset_phase_2 = get_arg_val<uint32_t>(argidx++);
    }
    uint32_t chunked_q_chunk_offset_phase_1_local = chunked_q_chunk_offset_phase_1;
    uint32_t chunked_q_chunk_offset_phase_2_local = chunked_q_chunk_offset_phase_2;

    // Per-core [start, count) slice of the B*NQH*q_num_effective flat range.
    uint32_t global_q_start = 0;
    uint32_t global_q_count = 0;
    if constexpr (flatten_work) {
        global_q_start = get_arg_val<uint32_t>(argidx++);
        global_q_count = get_arg_val<uint32_t>(argidx++);
    }

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    // KV chain forwarding (runtime args + semaphores populated when SDPA_KV_CHAIN_ENABLED).
    uint32_t is_chain_participant = 0;
    uint32_t is_injector = 0;
    uint32_t is_sink = 0;
    uint32_t chain_batch = 0;
    uint32_t chain_head = 0;
    uint32_t prev_physical_x = 0;
    uint32_t prev_physical_y = 0;
    uint32_t next_physical_x = 0;
    uint32_t next_physical_y = 0;
    uint32_t next_core_q_chunks = 0;
    uint32_t mcast_num_dests = 0;
    uint32_t mcast_sender_wait = 0;
    uint64_t mcast_base_noc_addr = 0;

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* valid_semaphore_addr_ptr = nullptr;
    uint64_t sender_semaphore_noc_addr = 0;
    uint64_t receiver_semaphore_noc_addr = 0;
    uint32_t valid_semaphore_addr = 0;
    uint32_t receiver_semaphore_l1_addr = 0;
    uint64_t mcast_sem_noc_addr = 0;
    uint32_t sender_wait_count = 1;

#if defined(SDPA_KV_CHAIN_ENABLED)
    {
        is_chain_participant = get_arg_val<uint32_t>(argidx++);
        is_injector = get_arg_val<uint32_t>(argidx++);
        is_sink = get_arg_val<uint32_t>(argidx++);
        chain_batch = get_arg_val<uint32_t>(argidx++);
        chain_head = get_arg_val<uint32_t>(argidx++);
        argidx += 2;  // skip chain_q_chunk_start, chain_q_chunk_count (host-only metadata)
        prev_physical_x = get_arg_val<uint32_t>(argidx++);
        prev_physical_y = get_arg_val<uint32_t>(argidx++);
        next_physical_x = get_arg_val<uint32_t>(argidx++);
        next_physical_y = get_arg_val<uint32_t>(argidx++);
        next_core_q_chunks = get_arg_val<uint32_t>(argidx++);
        mcast_num_dests = get_arg_val<uint32_t>(argidx++);
        mcast_sender_wait = get_arg_val<uint32_t>(argidx++);

        if (is_chain_participant) {
            const uint32_t sender_semaphore_addr = get_semaphore(sender_semaphore_id);
            const uint32_t receiver_semaphore_addr = get_semaphore(receiver_semaphore_id);
            valid_semaphore_addr = get_semaphore(valid_semaphore_id);
            receiver_semaphore_l1_addr = receiver_semaphore_addr;

            sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
            receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);
            valid_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_semaphore_addr);

            *valid_semaphore_addr_ptr = VALID;

            if constexpr (mcast_enabled) {
                // All chains use mcast (all-or-nothing compile-time decision)
                sender_semaphore_noc_addr = get_noc_addr(prev_physical_x, prev_physical_y, sender_semaphore_addr);
                if (is_injector) {
                    // prev_physical = mcast_start (first receiver), next_physical = mcast_end (last receiver)
                    mcast_base_noc_addr = get_noc_multicast_addr(
                        prev_physical_x,
                        prev_physical_y,
                        next_physical_x,
                        next_physical_y,
                        0);  // addr=0; will OR in actual L1 addr at use site
                    mcast_sem_noc_addr = mcast_base_noc_addr | receiver_semaphore_l1_addr;
                    sender_wait_count = mcast_sender_wait;
                }
            } else {
                sender_semaphore_noc_addr = get_noc_addr(prev_physical_x, prev_physical_y, sender_semaphore_addr);
                receiver_semaphore_noc_addr = get_noc_addr(next_physical_x, next_physical_y, receiver_semaphore_addr);
            }
        }
    }
#endif

    // When chunked: only process K/V up to (chunk_start_idx + Q_chunk_length) tokens.
    // valid_Skt_bound = min(offset_tiles + valid_Sqt, valid_Skt); cap at valid_Skt for callers that pass
    // different valid_Sqt (e.g. ring_distributed uses full Q length in tiles).

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_6;
    constexpr uint32_t cb_id_chunk_start_idx_compute = tt::CBIndex::c_8;
    constexpr uint32_t cb_id_chunk_start_idx_writer = tt::CBIndex::c_9;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr uint32_t attention_sink_tile_bytes = use_attention_sink ? get_tile_size(cb_attention_sink) : 0;

    constexpr uint32_t q_heads_per_k = NQH / NKH;
    constexpr uint32_t q_heads_per_v = NQH / NVH;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);
    const auto mask_reader = TensorAccessor(mask_args, mask_addr);
    const auto attention_sink_reader = TensorAccessor(attention_sink_args, attention_sink_addr);
    const auto chunk_start_idx_reader = TensorAccessor(chunk_start_idx_args, chunk_start_idx_addr);

    constexpr uint32_t skip_src_cols = (use_mla && mla_kv_overlap) ? DHt - vDHt : 0;

    const auto q_tile_shape = TensorTileShape(B, NQH, valid_Sqt, DHt);
    const auto k_tile_shape = TensorTileShape(B, NKH, valid_Skt, DHt);

    // If we have MLA:
    // - if k and v tensors are overlapped, we want to read from the k tensor, but just a portion of it, hence setting
    // the v tile shape dim to DHt (and skip accordingly based on skip_src_cols)
    // - if k and v tensors are not overlapped, we want to read from the v tensor, hence setting the v tile shape dim to
    // vDHt Otherwise head dim of k and v are same
    const auto v_tile_shape = TensorTileShape(B, NVH, valid_Skt, use_mla && !mla_kv_overlap ? vDHt : DHt);
    const auto attention_sink_tile_shape = TensorTileShape(B, NQH, 1, 1);

    volatile tt_l1_ptr uint32_t* page_table_ptr;

    if constexpr (is_chunked) {
        if (chunk_start_idx_addr != 0) {
            cb_reserve_back(cb_id_chunk_start_idx_compute, 1);
            uint32_t chunk_start_write_ptr = get_write_ptr(cb_id_chunk_start_idx_compute);
            noc_async_read(chunk_start_idx_reader.get_noc_addr(0), chunk_start_write_ptr, 4);
            noc_async_read_barrier();
            uint32_t chunk_start_idx = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chunk_start_write_ptr);
            cb_push_back(cb_id_chunk_start_idx_compute, 1);

            cb_reserve_back(cb_id_chunk_start_idx_writer, 1);
            uint32_t chunk_start_write_ptr_2 = get_write_ptr(cb_id_chunk_start_idx_writer);
            noc_async_read(chunk_start_idx_reader.get_noc_addr(0), chunk_start_write_ptr_2, 4);
            noc_async_read_barrier();
            cb_push_back(cb_id_chunk_start_idx_writer, 1);

            const uint32_t q_chunk_size = Sq_chunk_t * tt::constants::TILE_HEIGHT;
            chunked_q_chunk_offset_phase_1_local = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunked_q_chunk_offset_phase_2_local = chunked_q_chunk_offset_phase_1_local;
            }
        }
    }
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        const uint32_t chunked_q_chunk_offset =
            (phase == 0) ? chunked_q_chunk_offset_phase_1_local : chunked_q_chunk_offset_phase_2_local;
        const uint32_t read_offset = (phase == 0) ? read_offset_phase_1 : read_offset_phase_2;
        const uint32_t valid_Skt_bound =
            (chunk_start_idx_addr != 0)
                ? std::min(chunked_q_chunk_offset * Sq_chunk_t + valid_Sqt, valid_Skt)  // flexible/ring
                : (valid_Skt + chunked_q_chunk_offset * Sq_chunk_t);                    // legacy

        // Walk the (nb, nq, q_chunk) work items this core owns. Flat proxy mode decodes each slot
        // from a single linear index; hierarchical mode linearises its (nb, nq, q_iter) nested
        // iteration into the same loop so the 250-line K/V body below appears once. Host
        // guarantees !is_chunked and !use_attention_sink when flatten_work is true, so the paged
        // page_table and attention_sink blocks only engage on the hierarchical path.
        constexpr uint32_t kFirstIteration = static_cast<uint32_t>(-1);
        const uint32_t heads_local = local_nh_end - local_nh_start;
        const uint32_t total_q_iters =
            flatten_work ? global_q_count : (local_batch_end - local_batch_start) * heads_local * q_chunks_per_core;

        uint32_t prev_nb = kFirstIteration;
        uint32_t prev_nq = kFirstIteration;
#if defined(SDPA_KV_CHAIN_ENABLED)
        // Flat path tracks q_iter_local across same-head iterations — hierarchical gets it for
        // free from its inner q_iter counter.
        uint32_t flat_prev_head_id = kFirstIteration;
        uint32_t flat_q_iter_local = 0;
#endif

        for (uint32_t gq = 0; gq < total_q_iters; ++gq) {
            // 1. Resolve the work item (nb, nq, q_chunk, q_iter_local) for this gq.
            uint32_t nb, nq, q_chunk, q_iter_local;
            if constexpr (flatten_work) {
                const auto w =
                    decompose_flat_q_index(global_q_start + gq, q_num_chunks, NQH, flat_use_zigzag, proxy_mode);
                nb = w.nb;
                nq = w.nq;
                q_chunk = w.q_chunk;
#if defined(SDPA_KV_CHAIN_ENABLED)
                const uint32_t cur_head_id = nb * NQH + nq;
                flat_q_iter_local = (cur_head_id == flat_prev_head_id) ? (flat_q_iter_local + 1) : 0;
                flat_prev_head_id = cur_head_id;
                q_iter_local = flat_q_iter_local;
#else
                q_iter_local = 0;
#endif
            } else {
                const auto w = decompose_hierarchical_index(gq, heads_local, q_chunks_per_core);
                nb = local_batch_start + w.nb_idx;
                nq = local_nh_start + w.nq_idx;
#if defined BALANCED_Q_PARALLEL
                constexpr bool kBalancedQParallel = true;
#else
                constexpr bool kBalancedQParallel = false;
#endif
                q_chunk =
                    balanced_q_chunk(w.q_iter, local_q_start, q_chunks_per_core, q_num_chunks, kBalancedQParallel);
                q_iter_local = w.q_iter;
            }

            // 2. Per-batch paged-page-table handoff on nb transitions (hierarchical + chunked).
            //    Pop the previous batch's page_table, push the new one. First iteration skips pop.
            if constexpr (!flatten_work && is_chunked) {
                if (nb != prev_nb) {
                    if (prev_nb != kFirstIteration) {
                        cb_pop_front(cb_id_page_table, 1);
                    }
                    cb_reserve_back(cb_id_page_table, 1);
                    page_table_ptr = read_page_table_for_batch(
                        cb_id_page_table, nb, page_table_args, page_table_addr, page_table_stick_size);
                    cb_push_back(cb_id_page_table, 1);
                }
            }

            // 3. Mask batch offset: depends on nb only.
            uint32_t mask_batch_offset = 0;
            if constexpr (!broadcast_provided_mask_batch) {
                constexpr uint32_t heads_factor = broadcast_provided_mask_heads ? 1u : NQH;
                mask_batch_offset = nb * valid_Sqt * valid_Skt * heads_factor;
            }

            // 4. Per-(nb, nq) attention sink read, on (nb, nq) transitions (hierarchical + sink).
            if constexpr (!flatten_work && use_attention_sink) {
                if (nb != prev_nb || nq != prev_nq) {
                    cb_reserve_back(cb_attention_sink, Sq_chunk_t);
                    uint32_t attention_sink_write_ptr = get_write_ptr(cb_attention_sink);
                    const uint32_t sink_tile_id = attention_sink_tile_shape.id_of(0, nq, 0, 0);
                    noc_async_read_tile(sink_tile_id, attention_sink_reader, attention_sink_write_ptr);
                    noc_async_read_barrier();
                    fill_attention_sink_tiles<attention_sink_tile_bytes>(
                        cb_attention_sink, Sq_chunk_t, attention_sink_write_ptr);
                    cb_push_back(cb_attention_sink, Sq_chunk_t);
                }
            }

            prev_nb = nb;
            prev_nq = nq;

            // 5. Q read + K/V loop.

            // Determine how many rows of Q will be read. Both start and end rows are capped by valid_Sqt,
            // since Sq padding is independent of Sk padding.
            const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
            const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
            const uint32_t q_row_tile_count = q_row_end_tile - q_row_start_tile;
            // Non-const: read_q_subblock advances the tile id by reference.
            uint32_t q_read_tile_id = q_tile_shape.id_of(nb, nq, read_offset + q_row_start_tile, 0);

            // Q read is deferred into the K loop (k_chunk==0) for subblock interleaving.
            // When use_q_subblock_push is false, Q is read in full before the K loop (original behavior).
            if constexpr (!use_q_subblock_push) {
                read_chunk_with_padding<q_tile_bytes>(
                    q_reader, cb_q_in, q_read_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);
            }

            const uint32_t q_chunk_offs = chunked_q_chunk_offset + q_chunk;
            const uint32_t q_low_idx = q_chunk_offs * Sq_chunk_t;  // sequence index of first tile in this chunk
            const uint32_t q_high_idx = is_causal ? (q_low_idx + Sq_chunk_t) : Skt;

            const uint32_t k_head = nq / q_heads_per_k;
            const uint32_t v_head = nq / q_heads_per_v;

            // q_iter_local is the per-(nb, nq) slot index on this core, so a straddling flat-mode
            // core gates forwards against the next core's head slot count, not its whole-range gq.
            const bool should_forward = is_chain_participant && !is_sink && (nb == chain_batch) && (nq == chain_head) &&
                                        (q_iter_local < next_core_q_chunks);
            const bool should_receive =
                is_chain_participant && !is_injector && (nb == chain_batch) && (nq == chain_head);

            // UP proxy halves the K loop. Chain cores walk the full K range so injector + receivers
            // stay in lockstep; the lightweight causal mask zeroes softmax columns past q_high.
            uint32_t k_chunk_end;
            if constexpr (proxy_mode == RingProxyMode::Up) {
                k_chunk_end = k_num_chunks / 2;
            } else if (is_chain_participant) {
                k_chunk_end = k_num_chunks;
            } else {
                k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;
            }

            for (uint32_t k_chunk = 0; k_chunk < k_chunk_end; ++k_chunk) {
                const uint32_t kv_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt_bound);
                const uint32_t kv_row_end_tile = std::min(kv_row_start_tile + Sk_chunk_t, valid_Skt_bound);
                const uint32_t kv_row_tile_count = kv_row_end_tile - kv_row_start_tile;
                const uint32_t k_start_tile_id = k_tile_shape.id_of(nb, k_head, kv_row_start_tile, 0);
                const uint32_t v_start_tile_id = v_tile_shape.id_of(nb, v_head, kv_row_start_tile, 0);

                // K: either read locally (injector or not participant) or receive from previous core
                uint32_t cb_k_start_address = 0;

                if (should_receive) {
                    // Receive forwarded K chunk from previous core
                    cb_reserve_back(cb_k_in, k_chunk_tiles);
                    cb_k_start_address = get_write_ptr(cb_k_in);
                    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_k_in, k_chunk_tiles);
                } else {
                    // Read K chunk from DRAM
                    if constexpr (is_chunked) {
                        // Use page table to read K chunk (forwarding not supported for paged mode)
                        const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
                        read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                            k_reader,
                            cb_k_in,
                            k_head,
                            k_chunk_start_row_num,
                            kv_row_tile_count,
                            DHt,
                            Sk_chunk_t,
                            DHt,
                            k_tile_bytes,
                            barrier_threshold,
                            page_table_ptr,
                            true  // transpose=true for K reads
                        );
                    } else {
                        if (should_forward) {
                            cb_reserve_back(cb_k_in, k_chunk_tiles);
                            cb_k_start_address = get_write_ptr(cb_k_in);
                            read_chunk_for_forwarding<k_tile_bytes, true>(
                                k_reader, cb_k_start_address, k_start_tile_id, kv_row_tile_count, DHt, Sk_chunk_t, DHt);
                        } else {
                            read_chunk_with_padding<k_tile_bytes, decltype(k_reader), true, false>(
                                k_reader,
                                cb_k_in,
                                k_start_tile_id,
                                kv_row_tile_count,
                                DHt,
                                Sk_chunk_t,
                                DHt,
                                barrier_threshold,
                                true  // transpose=true for K reads
                            );
                        }
                    }
                }

                // Forward K chunk to next core(s): initiate async write (NOC write channel)
                // For mcast: send linked data + companion semaphore back-to-back.
                // The companion must be issued immediately after the linked write —
                // any noc_async_read_barrier() between them deadlocks (the read barrier
                // blocks while a linked write awaits its companion).
                if (should_forward) {
                    noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count);
                    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
                    if constexpr (mcast_enabled) {
                        uint64_t k_mcast_addr = mcast_base_noc_addr | cb_k_start_address;
                        noc_async_write_multicast(
                            cb_k_start_address,
                            k_mcast_addr,
                            k_chunk_tiles * k_tile_bytes,
                            mcast_num_dests,
                            true /* linked: semaphore mcast follows */);
                        noc_semaphore_set_multicast(valid_semaphore_addr, mcast_sem_noc_addr, mcast_num_dests);
                        noc_async_writes_flushed();
                        if (!should_receive) {
                            cb_push_back(cb_k_in, k_chunk_tiles);
                        }
                    } else {
                        uint64_t k_unicast_data_addr =
                            get_noc_addr(next_physical_x, next_physical_y, cb_k_start_address);
                        noc_async_write(cb_k_start_address, k_unicast_data_addr, k_chunk_tiles * k_tile_bytes);
                    }
                }

                // Mask read — safe after linked write pair is complete
                if constexpr (use_provided_mask) {
                    cb_reserve_back(cb_mask_in, mask_chunk_tiles);
                    uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
                    uint32_t barrier_count = 0;

                    uint32_t mask_row_start = mask_batch_offset + q_chunk * Sq_chunk_t * valid_Skt;
                    if constexpr (!broadcast_provided_mask_heads) {
                        mask_row_start += nq * valid_Sqt * valid_Skt;
                    }

                    uint32_t tile_idx = 0;
                    for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                        const uint32_t global_q_tile = q_chunk * Sq_chunk_t + row;
                        const bool q_valid = !use_padded_mask || (global_q_tile < valid_Sqt);
                        for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                            const uint32_t global_k_tile = k_chunk * Sk_chunk_t + col;
                            const bool k_valid = !use_padded_mask || (global_k_tile < valid_Skt);
                            if (q_valid && k_valid) {
                                noc_async_read_tile(mask_row_start + global_k_tile, mask_reader, mask_write_ptr);
                            } else {
                                fill_neginf_tile<mask_tile_bytes>(cb_mask_in, tile_idx);
                            }
                            mask_write_ptr += mask_tile_bytes;
                            tile_idx++;
                            if (++barrier_count == barrier_threshold) {
                                noc_async_read_barrier();
                                barrier_count = 0;
                            }
                        }
                        if (q_valid) {
                            mask_row_start += valid_Skt;
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_mask_in, mask_chunk_tiles);
                }

                // Complete K forward: flush write and signal receiver(s)
                // (mcast path already completed above — companion sent with linked write)
                if (should_forward) {
                    if constexpr (!mcast_enabled) {
                        noc_async_writes_flushed();
                        if (!should_receive) {
                            cb_push_back(cb_k_in, k_chunk_tiles);
                        }
                        noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
                    }
                }

                // Q subblock push: K is fully forwarded, now push Q one subblock at
                // a time. Compute waits for K first (cb_wait_front(cb_k_in, K*N)),
                // then waits for Q subblocks incrementally (accumulating cb_wait_front).
                // Each push unblocks the next QK subblock computation.
                // Placed after K forward complete so no outstanding NOC writes remain
                // (noc_async_read_barrier inside read_q_subblock deadlocks on BH
                // when NOC writes are in-flight).
                if constexpr (use_q_subblock_push) {
                    if (k_chunk == 0) {
                        for (uint32_t q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {
                            read_q_subblock<q_tile_bytes>(
                                q_reader,
                                cb_q_in,
                                q_read_tile_id,
                                q_sub * qk_subblock_h,
                                qk_subblock_h,
                                q_row_tile_count,
                                DHt,
                                DHt,
                                barrier_threshold);
                        }
                    }
                }

                // V: either read locally (injector or not participant) or receive from previous core
                uint32_t cb_v_start_address = 0;

                if (should_receive) {
                    // Receive forwarded V chunk from previous core
                    cb_reserve_back(cb_v_in, v_chunk_tiles);
                    cb_v_start_address = get_write_ptr(cb_v_in);
                    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_v_in, v_chunk_tiles);
                } else {
                    // Read V chunk from DRAM
                    if constexpr (is_chunked) {
                        // Use page table to read V chunk (forwarding not supported for paged mode)
                        const uint32_t kv_chunk_start_row_num = k_chunk * Sk_chunk_t;
                        constexpr uint32_t head_dim = (use_mla && !mla_kv_overlap) ? vDHt : DHt;
                        read_paged_chunk_with_padding<NVH, block_size_t, head_dim>(
                            v_reader,
                            cb_v_in,
                            v_head,
                            kv_chunk_start_row_num,
                            kv_row_tile_count,
                            vDHt,
                            Sk_chunk_t,
                            vDHt,
                            v_tile_bytes,
                            barrier_threshold,
                            page_table_ptr,
                            false,
                            skip_src_cols);
                    } else {
                        if (should_forward) {
                            cb_reserve_back(cb_v_in, v_chunk_tiles);
                            cb_v_start_address = get_write_ptr(cb_v_in);
                            read_chunk_for_forwarding<v_tile_bytes, false>(
                                v_reader,
                                cb_v_start_address,
                                v_start_tile_id,
                                kv_row_tile_count,
                                vDHt,
                                Sk_chunk_t,
                                vDHt,
                                skip_src_cols);
                        } else {
                            read_chunk_with_padding<v_tile_bytes, decltype(v_reader), true, false>(
                                v_reader,
                                cb_v_in,
                                v_start_tile_id,
                                kv_row_tile_count,
                                vDHt,
                                Sk_chunk_t,
                                vDHt,
                                barrier_threshold,
                                false,
                                skip_src_cols);
                        }
                    }
                }

                // Forward V chunk to next core(s) before push_back — prevents compute from
                // popping the buffer while the mcast is still reading from it.
                if (should_forward) {
                    noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count);
                    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
                    if constexpr (mcast_enabled) {
                        uint64_t v_mcast_addr = mcast_base_noc_addr | cb_v_start_address;
                        noc_async_write_multicast(
                            cb_v_start_address,
                            v_mcast_addr,
                            v_chunk_tiles * v_tile_bytes,
                            mcast_num_dests,
                            true /* linked: semaphore mcast follows */);
                        noc_semaphore_set_multicast(valid_semaphore_addr, mcast_sem_noc_addr, mcast_num_dests);
                    } else {
                        uint64_t v_unicast_data_addr =
                            get_noc_addr(next_physical_x, next_physical_y, cb_v_start_address);
                        noc_async_write(cb_v_start_address, v_unicast_data_addr, v_chunk_tiles * v_tile_bytes);
                    }
                    noc_async_writes_flushed();
                    if constexpr (!mcast_enabled) {
                        noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
                    }
                    if (!should_receive) {
                        cb_push_back(cb_v_in, v_chunk_tiles);
                    }
                }
            }  // close k_chunk
        }  // close gq

        // Close out the last batch's page_table push — phase loop invariant is "no entries left".
        if constexpr (!flatten_work && is_chunked) {
            if (prev_nb != kFirstIteration) {
                cb_pop_front(cb_id_page_table, 1);
            }
        }
    }  // close phase
}
