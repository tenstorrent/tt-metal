// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    constexpr uint32_t vDHt = get_compile_time_arg_val(8);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(9);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(10);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(11);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_cores = get_compile_time_arg_val(13);
    constexpr uint32_t is_causal = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t block_size_t = get_compile_time_arg_val(18);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(19);
    constexpr uint32_t use_attention_sink = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t q_per_core = get_compile_time_arg_val(21);
    uint32_t sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(22));
    uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(23));
    uint32_t valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(24));

    constexpr auto q_args = TensorAccessorArgs<25>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t attention_sink_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_count = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_phases = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunked_q_chunk_offset_phase_1 = get_arg_val<uint32_t>(argidx++);
    const uint32_t read_offset_phase_1 = get_arg_val<uint32_t>(argidx++);
    uint32_t chunked_q_chunk_offset_phase_2 = 0;
    uint32_t read_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunked_q_chunk_offset_phase_2 = get_arg_val<uint32_t>(argidx++);
        read_offset_phase_2 = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t is_chain_participant = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_injector = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_batch = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_head = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_q_chunk_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t chain_q_chunk_count = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_physical_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t receiver_bbox_start_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t receiver_bbox_start_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t receiver_bbox_end_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t receiver_bbox_end_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t receiver_count = get_arg_val<uint32_t>(argidx++);

    // VALID sem used to write L1-L1 valid semaphore
    volatile tt_l1_ptr uint32_t* valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_semaphore_addr);
    *(valid_semaphore_addr_ptr) = VALID;

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);

    const bool participates_in_chain = is_chain_participant == 1;
    const bool is_chain_sender = participates_in_chain && (is_injector == 1);
    const bool is_chain_receiver = participates_in_chain && (is_sink == 1);
    const bool has_receivers = receiver_count > 0;

    uint64_t sender_semaphore_noc_addr = 0;
    uint64_t receiver_semaphore_mcast_addr = 0;
    if (participates_in_chain && is_chain_receiver) {
        sender_semaphore_noc_addr = get_noc_addr(sender_physical_x, sender_physical_y, sender_semaphore_addr);
    }
    if (is_chain_sender && has_receivers) {
        receiver_semaphore_mcast_addr = get_noc_multicast_addr(
            receiver_bbox_start_x,
            receiver_bbox_start_y,
            receiver_bbox_end_x,
            receiver_bbox_end_y,
            receiver_semaphore_addr);
    }

    // When chunked, update the bounds of valid K sequence length based on Q chunk offset

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

    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr uint32_t attention_sink_tile_bytes = use_attention_sink ? get_tile_size(cb_attention_sink) : 0;

    constexpr uint32_t q_heads_per_kv = NQH / NKH;

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto v_reader = TensorAccessor(v_args, v_addr, v_tile_bytes);
    const auto mask_reader = TensorAccessor(mask_args, mask_addr, mask_tile_bytes);
    const auto attention_sink_reader =
        TensorAccessor(attention_sink_args, attention_sink_addr, attention_sink_tile_bytes);

    const auto q_tile_shape = TensorTileShape(B, NQH, valid_Sqt, DHt);
    const auto k_tile_shape = TensorTileShape(B, NKH, valid_Skt, DHt);
    const auto v_tile_shape = TensorTileShape(B, NKH, valid_Skt, DHt);
    const auto mask_tile_shape = TensorTileShape(B, 1, valid_Sqt, valid_Skt);
    const auto attention_sink_tile_shape = TensorTileShape(B, NQH, 1, 1);

    volatile tt_l1_ptr uint32_t* page_table_ptr;

    uint32_t v_tile_id = 0;
    uint32_t mask_tile_id = 0;
    uint32_t barrier_count = 0;
    uint32_t chunked_q_chunk_offset = 0;
    uint32_t read_offset = 0;
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_1;
            read_offset = read_offset_phase_1;
        } else {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_2;
            read_offset = read_offset_phase_2;
        }
        uint32_t valid_Skt_bound = valid_Skt + chunked_q_chunk_offset * Sq_chunk_t;

        for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_start + global_q_count;
             ++global_q_chunk) {
            const uint32_t nb = global_q_chunk / (NQH * q_num_chunks);
            const uint32_t nq = (global_q_chunk % (NQH * q_num_chunks)) / q_num_chunks;
            const uint32_t q_chunk = global_q_chunk % q_num_chunks;
            const uint32_t q_iter = q_chunk - global_q_start % q_num_chunks;

            const bool matching_chain_iteration = participates_in_chain && (nb == chain_batch) && (nq == chain_head);
            const bool receiver_needs_data = matching_chain_iteration && is_chain_receiver;
            const bool sender_should_multicast = matching_chain_iteration && is_chain_sender && has_receivers;

            const uint32_t mask_batch_offset = nb * Sqt * Skt;
            /*
            Read a chunk of Q. BALANCED_Q_PARALLEL evenly distributes Q chunks
            across cores when causal and other conditions are met.
            When chunked, we must treat Q as offset by some factor.
            When causal, we set up the bounds such that we only read the lower triangle of K and V.
            When non-causal, read all of K and V.
            */

            /*
            Determine how many rows of Q will be read. Both start and end rows are
            capped by valid_Sqt, since Sq padding is independent of Sk padding.
            */
            const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
            const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
            const uint32_t q_row_tile_count = q_row_end_tile - q_row_start_tile;
            const uint32_t q_tile_id = q_tile_shape.id_of(nb, nq, read_offset + q_row_start_tile, 0);
            read_chunk_with_padding<q_tile_bytes>(
                q_reader, cb_q_in, q_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);

            uint32_t q_low_idx = q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
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

                cb_reserve_back(cb_k_in, k_chunk_tiles);
                uint32_t cb_k_start_address = get_write_ptr(cb_k_in);
                if (!receiver_needs_data) {
                    DPRINT << "reading K chunk " << k_chunk << " for Q chunk " << q_chunk << ENDL();
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
                        read_chunk_with_padding<k_tile_bytes>(
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
                } else {
                    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                    DPRINT << "waiting for k chunk from " << sender_physical_x << ", " << sender_physical_y << ENDL();
                    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_k_in, k_chunk_tiles);
                }

                if (sender_should_multicast) {
                    DPRINT << "waiting for all receivers to be ready" << ENDL();
                    noc_semaphore_wait(sender_semaphore_addr_ptr, receiver_count);
                    noc_semaphore_set(sender_semaphore_addr_ptr, 0);

                    const uint64_t k_multicast_data_addr = get_noc_multicast_addr(
                        receiver_bbox_start_x,
                        receiver_bbox_start_y,
                        receiver_bbox_end_x,
                        receiver_bbox_end_y,
                        cb_k_start_address);
                    noc_async_write_multicast(
                        cb_k_start_address, k_multicast_data_addr, k_chunk_tiles * k_tile_bytes, receiver_count);

#if defined(ARCH_BLACKHOLE)
                    noc_async_writes_flushed();
#endif

                    DPRINT << "setting multicast valid semaphore" << ENDL();
                    noc_semaphore_set_multicast(valid_semaphore_addr, receiver_semaphore_mcast_addr, receiver_count);
                }

                cb_reserve_back(cb_v_in, v_chunk_tiles);
                uint32_t cb_v_start_address = get_write_ptr(cb_v_in);
                if (!receiver_needs_data) {
                    if constexpr (is_chunked) {
                        // Use page table to read V chunk
                        const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
                        read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                            v_reader,
                            cb_v_in,
                            kv_head,
                            k_chunk_start_row_num,
                            k_row_tile_count,
                            vDHt,
                            Sk_chunk_t,
                            vDHt,
                            v_tile_bytes,
                            barrier_threshold,
                            page_table_ptr,
                            false,
                            DHt - vDHt /* src_skip_cols */);
                    } else {
                        read_chunk_with_padding<v_tile_bytes>(
                            v_reader,
                            cb_v_in,
                            k_start_tile_id,
                            k_row_tile_count,
                            vDHt,
                            Sk_chunk_t,
                            vDHt,
                            barrier_threshold,
                            false,
                            DHt - vDHt /* src_skip_cols */);
                    }
                } else {
                    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                    DPRINT << "waiting for k chunk from " << sender_physical_x << ", " << sender_physical_y << ENDL();
                    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_v_in, v_chunk_tiles);
                }

                if (sender_should_multicast) {
                    DPRINT << "waiting for all receivers to be ready" << ENDL();
                    noc_semaphore_wait(sender_semaphore_addr_ptr, receiver_count);
                    noc_semaphore_set(sender_semaphore_addr_ptr, 0);

                    const uint64_t v_multicast_data_addr = get_noc_multicast_addr(
                        receiver_bbox_start_x,
                        receiver_bbox_start_y,
                        receiver_bbox_end_x,
                        receiver_bbox_end_y,
                        cb_v_start_address);
                    noc_async_write_multicast(
                        cb_v_start_address, v_multicast_data_addr, v_chunk_tiles * v_tile_bytes, receiver_count);
#if defined(ARCH_BLACKHOLE)
                    noc_async_writes_flushed();
#endif

                    DPRINT << "setting multicast valid semaphore" << ENDL();
                    noc_semaphore_set_multicast(valid_semaphore_addr, receiver_semaphore_mcast_addr, receiver_count);
                }
            }

        if constexpr (is_chunked) {
            cb_pop_front(cb_id_page_table, 1);
        }
        }
    }
}
