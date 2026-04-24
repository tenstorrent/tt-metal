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

    // Semaphore IDs for KV chain forwarding (non-causal only, but always present in compile args)
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
    // Flat-work gate + zigzag sub-mode (tail args; zigzag is only meaningful when flatten_work is true).
    constexpr uint32_t flat_work_cta_base = chunk_start_idx_args.next_compile_time_args_offset();
    constexpr bool flatten_work = get_compile_time_arg_val(flat_work_cta_base) == 1;
    constexpr bool flat_use_zigzag = get_compile_time_arg_val(flat_work_cta_base + 1) == 1;

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

    // Flat work distribution: causal only, non-chunked, no attention sink. num_phases is always 1
    // for this factory, so these args sit right after read_offset_phase_1 with no intervening args.
    // Zigzag sub-mode is compile-time arg flat_use_zigzag (declared above).
    uint32_t global_q_start = 0;
    uint32_t global_q_count = 0;
    if constexpr (flatten_work) {
        global_q_start = get_arg_val<uint32_t>(argidx++);
        global_q_count = get_arg_val<uint32_t>(argidx++);
    }

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    // Parse chain metadata for KV forwarding (non-causal only)
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

    // Initialize semaphore addresses and NOC addresses for chain forwarding
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

    uint32_t chunked_q_chunk_offset = 0;
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
    uint32_t read_offset = 0;

    // Shared per-(nb, nq, q_chunk, q_iter_local, mask_batch_offset) body. Flat and hierarchical
    // branches below only differ in how they iterate (nb, nq, q_chunk) and how they derive
    // q_iter_local / mask_batch_offset; everything from the Q read through the K-loop is identical.
    uint32_t valid_Skt_bound = 0;
    auto process_q_chunk =
        [&](uint32_t nb, uint32_t nq, uint32_t q_chunk, uint32_t q_iter_local, uint32_t mask_batch_offset) {
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

            // Chain forwarding conditions are loop-invariant — compute once. q_iter_local counts slots
            // within the current (batch, head) so that straddling cores in flat mode (whose range spans
            // multiple heads) gate forwards on the per-head slot count rather than the whole-range gq.
            bool should_forward = false;
            bool should_receive = false;
#if defined(SDPA_KV_CHAIN_ENABLED)
            should_forward = is_chain_participant && !is_sink && (nb == chain_batch && nq == chain_head) &&
                             (q_iter_local < next_core_q_chunks);
            should_receive = is_chain_participant && !is_injector && (nb == chain_batch && nq == chain_head);
#endif

        // Loop while k_low < q_high. Ring proxy UP caps K at k_num_chunks/2 to mirror the ring_joint
        // non-diag iter that sees only half of K per Q. Under SDPA_KV_CHAIN_ENABLED, chain cores
        // loop the full k_num_chunks regardless of Q position so injector + receivers walk matching
        // K ranges — lightweight_causal mask zeroes out the extra columns past q_high_idx.
#if defined(SDPA_RING_PROXY_UP)
            const uint32_t k_chunk_end = k_num_chunks / 2;
#elif defined(SDPA_KV_CHAIN_ENABLED)
            const uint32_t k_chunk_end =
                is_chain_participant ? k_num_chunks : ((q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t);
#else
            const uint32_t k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;
#endif

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
        };  // close process_q_chunk lambda

    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_1_local;
            read_offset = read_offset_phase_1;
        } else {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_2_local;
            read_offset = read_offset_phase_2;
        }
        if (chunk_start_idx_addr != 0) {
            // Flexible or ring: cap at valid_Skt so we never read past K/V extent.
            valid_Skt_bound = std::min(chunked_q_chunk_offset * Sq_chunk_t + valid_Sqt, valid_Skt);
        } else {
            // Legacy: extend by offset so one program can serve all chunks (valid_Skt is chunk 0's).
            valid_Skt_bound = valid_Skt + chunked_q_chunk_offset * Sq_chunk_t;
        }

        if constexpr (flatten_work) {
            // Flat iteration over a linear range of B*NQH*q_num_chunks chunks. Host enforces
            // !is_chunked and !use_attention_sink, so the hierarchical-only page-table / sink reads
            // don't apply. q_iter_local resets at every (batch, head) transition so chain forwarding
            // guards (`q_iter_local < next_core_q_chunks`) count slots within the current head only;
            // for straddling cores whose range spans multiple heads, this differs from gq.
            uint32_t prev_nb_flat = static_cast<uint32_t>(-1);
            uint32_t mask_batch_offset = 0;
            uint32_t q_iter_local_counter = 0;
            uint32_t prev_head_id_flat = static_cast<uint32_t>(-1);
            for (uint32_t gq = 0; gq < global_q_count; ++gq) {
                const auto decoded = decompose_flat_q_index_with_proxy(
                    global_q_start + gq, q_num_chunks, NQH, flat_use_zigzag, sdpa_proxy_mode);
                const uint32_t cur_head_id = decoded.nb * NQH + decoded.nq;
                if (cur_head_id != prev_head_id_flat) {
                    q_iter_local_counter = 0;
                    prev_head_id_flat = cur_head_id;
                } else {
                    q_iter_local_counter++;
                }
                if (decoded.nb != prev_nb_flat) {
                    prev_nb_flat = decoded.nb;
                    if constexpr (!broadcast_provided_mask_batch) {
                        if constexpr (broadcast_provided_mask_heads) {
                            mask_batch_offset = decoded.nb * valid_Sqt * valid_Skt;
                        } else {
                            mask_batch_offset = decoded.nb * valid_Sqt * valid_Skt * NQH;
                        }
                    }
                }
                process_q_chunk(decoded.nb, decoded.nq, decoded.q_chunk, q_iter_local_counter, mask_batch_offset);
            }
        } else {
            for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
                if constexpr (is_chunked) {
                    // Chunked means that we have paged attention
                    cb_reserve_back(cb_id_page_table, 1);
                    page_table_ptr = read_page_table_for_batch(
                        cb_id_page_table, nb, page_table_args, page_table_addr, page_table_stick_size);
                    cb_push_back(cb_id_page_table, 1);
                }

                // Calculate mask batch offset based on broadcasting (using unpadded mask dimensions):
                // - If batch is broadcasted [1 x ...]: always use batch=0, so offset = 0
                // - If batch is not broadcasted [b x ...]: use actual batch nb
                uint32_t mask_batch_offset = 0;
                if constexpr (!broadcast_provided_mask_batch) {
                    if constexpr (broadcast_provided_mask_heads) {
                        // [b x 1 x s x s]: batch offset without head factor
                        mask_batch_offset = nb * valid_Sqt * valid_Skt;
                    } else {
                        // [b x h x s x s]: batch offset with all heads
                        mask_batch_offset = nb * valid_Sqt * valid_Skt * NQH;
                    }
                }
                for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                    // Read attention sink for this Q chunk if enabled
                    if constexpr (use_attention_sink) {
                        cb_reserve_back(cb_attention_sink, Sq_chunk_t);
                        uint32_t attention_sink_write_ptr = get_write_ptr(cb_attention_sink);

                        // Attention sink has shape [1, NH, 1, 1] - single value per head
                        const uint32_t sink_tile_id = attention_sink_tile_shape.id_of(0, nq, 0, 0);
                        noc_async_read_tile(sink_tile_id, attention_sink_reader, attention_sink_write_ptr);
                        noc_async_read_barrier();

                        fill_attention_sink_tiles<attention_sink_tile_bytes>(
                            cb_attention_sink, Sq_chunk_t, attention_sink_write_ptr);

                        cb_push_back(cb_attention_sink, Sq_chunk_t);
                    }
                    for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                        // BALANCED_Q_PARALLEL evenly distributes light/heavy Q chunks across cores
                        // when causal+even; otherwise consecutive.
                        uint32_t q_chunk;
#if defined BALANCED_Q_PARALLEL
                        const uint32_t q_chunk_div_2 = q_chunks_per_core / 2;
                        if (q_iter < q_chunk_div_2) {  // bottom half
                            q_chunk = local_q_start + q_iter;
                        } else {
                            const uint32_t back_q_iter = q_iter - q_chunk_div_2;  // back half starts at 0
                            q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
                        }
#else
                        q_chunk = local_q_start + q_iter;
#endif
                        // Non-flat mode: one (nb, nq) pair per q_iter, so q_iter is the per-head slot.
                        process_q_chunk(nb, nq, q_chunk, q_iter, mask_batch_offset);
                    }
                }
                if constexpr (is_chunked) {
                    cb_pop_front(cb_id_page_table, 1);
                }
            }
        }
    }  // close phase
}
