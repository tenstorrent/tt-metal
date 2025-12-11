// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
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
    constexpr uint32_t broadcast_provided_mask_heads = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t block_size_t = get_compile_time_arg_val(19);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(20);
    constexpr uint32_t use_attention_sink = get_compile_time_arg_val(21) == 1;

    constexpr auto q_args = TensorAccessorArgs<22>();
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
    const uint32_t prev_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t prev_physical_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_physical_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_physical_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t next_core_q_chunks = get_arg_val<uint32_t>(argidx++);

    // VALID sem used to write L1-L1 valid semaphore
    volatile tt_l1_ptr uint32_t* valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_semaphore_addr);
    *(valid_semaphore_addr_ptr) = VALID;

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    const uint64_t sender_semaphore_noc_addr = get_noc_addr(prev_physical_x, prev_physical_y, sender_semaphore_addr);

    const uint64_t receiver_semaphore_noc_addr =
        get_noc_addr(next_physical_x, next_physical_y, receiver_semaphore_addr);

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

            uint32_t mask_batch_offset = nb * Sqt * Skt;
            if constexpr (!broadcast_provided_mask_heads) {
                mask_batch_offset *= NQH;
            }
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                // Read attention sink for this Q chunk if enabled
                if constexpr (use_attention_sink) {
                    cb_reserve_back(cb_attention_sink, Sq_chunk_t);
                    uint32_t attention_sink_write_ptr = get_write_ptr(cb_attention_sink);

                    // Attention sink has shape [1, NH, 1, 1] - single value per head
                    // Read the single tile for this head into the first tile of the CB
                    const uint32_t sink_tile_id =
                        attention_sink_tile_shape.id_of(0, nq, 0, 0);  // batch=0 since shape is [1,NH,1,1]
                    noc_async_read_tile(sink_tile_id, attention_sink_reader, attention_sink_write_ptr);
                    noc_async_read_barrier();

                    // Fill all Sq_chunk_t tiles in the CB by copying the first element of the source tile
                    // to the first element of every row in each destination tile
                    fill_attention_sink_tiles<attention_sink_tile_bytes>(
                        cb_attention_sink, Sq_chunk_t, attention_sink_write_ptr);

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
                if (is_injector or !is_chain_participant or (nb != chain_batch or nq != chain_head)) {
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
                } else if (is_chain_participant) {
                    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                    DPRINT << "waiting for k chunk from " << prev_physical_x << ", " << prev_physical_y << ENDL();
                    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_k_in, k_chunk_tiles);
                }

                if (is_chain_participant && !is_sink && q_iter < next_core_q_chunks &&
                    (nb == chain_batch and nq == chain_head)) {
                    // TODO: and the receiver can process this Q iter!
                    DPRINT << "waiting for sender semaphore to be valid" << ENDL();
                    noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(sender_semaphore_addr_ptr, 0);

                    uint64_t unicast_data_addr = get_noc_addr(next_physical_x, next_physical_y, cb_k_start_address);
                    noc_async_write(cb_k_start_address, unicast_data_addr, k_chunk_tiles * k_tile_bytes);
                    noc_async_writes_flushed();

                    DPRINT << "setting remote valid semaphore to " << receiver_semaphore_noc_addr << ENDL();
                    noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
                }

                        if constexpr (use_provided_mask) {
                            // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                            // Q-range = [q_low, q_high)
                            // K-range = [k_low, k_high)
                            // does_overlap = not (q_low >= k_high or k_low >= q_high)
                            // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional
                            // check Read mask chunk When a mask is provided, there will be no padding on q or kv.
                            cb_reserve_back(cb_mask_in, mask_chunk_tiles);
                            uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
                            barrier_count = 0;
                            mask_tile_id = mask_batch_offset + q_chunk * Sq_chunk_t * Skt /*row_offset*/ +
                                           k_chunk * Sk_chunk_t /*col_offset*/;
                            if constexpr (!broadcast_provided_mask_heads) {
                                mask_tile_id += nq * Sqt * Skt;
                            }
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
                        }
                        // Strid along columns to get to next row
                        mask_tile_id -= Sk_chunk_t;
                        mask_tile_id += Skt;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_mask_in, mask_chunk_tiles);
                }

                cb_reserve_back(cb_v_in, v_chunk_tiles);
                uint32_t cb_v_start_address = get_write_ptr(cb_v_in);
                if (is_injector or !is_chain_participant or (nb != chain_batch or nq != chain_head)) {
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
                } else if (is_chain_participant) {
                    // Receive forwarded V chunk from previous core
                    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_v_in, v_chunk_tiles);
                }

                if (is_chain_participant && !is_sink && q_iter < next_core_q_chunks &&
                    (nb == chain_batch and nq == chain_head)) {
                    // Forward V chunk to next core
                    noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(sender_semaphore_addr_ptr, 0);

                    uint64_t v_unicast_data_addr = get_noc_addr(next_physical_x, next_physical_y, cb_v_start_address);
                    noc_async_write(cb_v_start_address, v_unicast_data_addr, v_chunk_tiles * v_tile_bytes);
                    noc_async_writes_flushed();

                    noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
                }
            }

        if constexpr (is_chunked) {
            cb_pop_front(cb_id_page_table, 1);
        }
        }
    }
}
