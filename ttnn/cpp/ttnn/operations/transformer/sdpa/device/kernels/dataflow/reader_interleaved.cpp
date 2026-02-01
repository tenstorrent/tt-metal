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
    constexpr uint32_t broadcast_provided_mask_batch = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t broadcast_provided_mask_heads = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t block_size_t = get_compile_time_arg_val(20);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(21);
    constexpr uint32_t use_attention_sink = get_compile_time_arg_val(22) == 1;

    // Semaphore IDs for KV chain forwarding (non-causal only, but always present in compile args)
    constexpr uint32_t sender_semaphore_id = get_compile_time_arg_val(23);
    constexpr uint32_t receiver_semaphore_id = get_compile_time_arg_val(24);
    constexpr uint32_t valid_semaphore_id = get_compile_time_arg_val(25);

    constexpr auto q_args = TensorAccessorArgs<26>();
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

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    // Parse chain metadata for KV forwarding (non-causal only)
    uint32_t is_chain_participant = 0;
    uint32_t is_injector = 0;
    uint32_t is_sink = 0;
    uint32_t chain_batch = 0;
    uint32_t chain_head = 0;
    uint32_t chain_q_chunk_start = 0;
    uint32_t chain_q_chunk_count = 0;
    uint32_t prev_physical_x = 0;
    uint32_t prev_physical_y = 0;
    uint32_t next_physical_x = 0;
    uint32_t next_physical_y = 0;
    uint32_t next_core_q_chunks = 0;

    // Initialize semaphore addresses and NOC addresses for chain forwarding
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* valid_semaphore_addr_ptr = nullptr;
    uint64_t sender_semaphore_noc_addr = 0;
    uint64_t receiver_semaphore_noc_addr = 0;

    if constexpr (!is_causal) {
        is_chain_participant = get_arg_val<uint32_t>(argidx++);
        is_injector = get_arg_val<uint32_t>(argidx++);
        is_sink = get_arg_val<uint32_t>(argidx++);
        chain_batch = get_arg_val<uint32_t>(argidx++);
        chain_head = get_arg_val<uint32_t>(argidx++);
        chain_q_chunk_start = get_arg_val<uint32_t>(argidx++);
        chain_q_chunk_count = get_arg_val<uint32_t>(argidx++);
        prev_physical_x = get_arg_val<uint32_t>(argidx++);
        prev_physical_y = get_arg_val<uint32_t>(argidx++);
        next_physical_x = get_arg_val<uint32_t>(argidx++);
        next_physical_y = get_arg_val<uint32_t>(argidx++);
        next_core_q_chunks = get_arg_val<uint32_t>(argidx++);

        if (is_chain_participant) {
            const uint32_t sender_semaphore_addr = get_semaphore(sender_semaphore_id);
            const uint32_t receiver_semaphore_addr = get_semaphore(receiver_semaphore_id);
            const uint32_t valid_semaphore_addr = get_semaphore(valid_semaphore_id);

            sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
            receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);
            valid_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_semaphore_addr);

            *valid_semaphore_addr_ptr = VALID;

            sender_semaphore_noc_addr = get_noc_addr(prev_physical_x, prev_physical_y, sender_semaphore_addr);
            receiver_semaphore_noc_addr = get_noc_addr(next_physical_x, next_physical_y, receiver_semaphore_addr);
        }
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
                    // Read the single tile for this head into the first tile of the CB
                    const uint32_t sink_tile_id =
                        attention_sink_tile_shape.id_of(0, nq, 0, 0);  // batch=0 since shape is [1,NH,1,1]
                    noc_async_read_tile(sink_tile_id, attention_sink_reader, attention_sink_write_ptr);
                    noc_async_read_barrier();

                    // Fill all Sq_chunk_t tiles in the CB by copying the first element of the source tile
                    // to the first element of every row in each destination tile
                    fill_attention_sink_tiles<attention_sink_tile_bytes>(
                        cb_attention_sink, Sq_chunk_t, attention_sink_write_ptr);

                    cb_push_back(cb_attention_sink, Sq_chunk_t);
                }
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
                    const uint32_t q_tile_id = q_tile_shape.id_of(nb, nq, read_offset + q_row_start_tile, 0);
                    read_chunk_with_padding<q_tile_bytes>(
                        q_reader, cb_q_in, q_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);
                    q_chunk = chunked_q_chunk_offset + q_chunk;
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

                        // K: either read locally (injector or not participant) or receive from previous core
                        uint32_t cb_k_start_address = 0;
                        bool should_forward_k = false;
                        bool should_receive_k = false;

                        if constexpr (!is_causal) {
                            should_forward_k = is_chain_participant && !is_sink &&
                                               (nb == chain_batch && nq == chain_head) && (q_iter < next_core_q_chunks);
                            should_receive_k =
                                is_chain_participant && !is_injector && (nb == chain_batch && nq == chain_head);
                        }

                        if (should_receive_k) {
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
                                if (should_forward_k) {
                                    // Read without auto-push so we can capture address for forwarding
                                    cb_reserve_back(cb_k_in, k_chunk_tiles);
                                    cb_k_start_address = get_write_ptr(cb_k_in);
                                    // Manually read tiles with transpose (K reads are transposed)
                                    constexpr bool transpose = true;
                                    const uint32_t outer_ptr_stride = transpose ? k_tile_bytes : DHt * k_tile_bytes;
                                    const uint32_t inner_ptr_stride =
                                        transpose ? k_tile_bytes * Sk_chunk_t : k_tile_bytes;
                                    uint32_t tile_id = k_start_tile_id;
                                    for (uint32_t row = 0; row < k_row_tile_count; ++row) {
                                        uint32_t write_ptr = cb_k_start_address + row * outer_ptr_stride;
                                        for (uint32_t col = 0; col < DHt; ++col) {
                                            noc_async_read_tile(tile_id++, k_reader, write_ptr);
                                            write_ptr += inner_ptr_stride;
                                        }
                                    }
                                    // Zero padding if needed (row and col padding)
                                    for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                                        for (uint32_t col = 0; col < DHt; ++col) {
                                            if (row < k_row_tile_count && col < DHt) {
                                                continue;  // Already read
                                            }
                                            uint32_t tile_idx = transpose ? col * Sk_chunk_t + row : row * DHt + col;
                                            fill_tile_zeros<k_tile_bytes, false>(cb_k_in, tile_idx);
                                        }
                                    }
                                    noc_async_read_barrier();
                                    cb_push_back(cb_k_in, k_chunk_tiles);
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
                            }
                        }

                        // Forward K chunk to next core if applicable
                        if (should_forward_k) {
                            noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
                            noc_semaphore_set(sender_semaphore_addr_ptr, 0);
                            uint64_t k_unicast_data_addr =
                                get_noc_addr(next_physical_x, next_physical_y, cb_k_start_address);
                            noc_async_write(cb_k_start_address, k_unicast_data_addr, k_chunk_tiles * k_tile_bytes);
                            noc_async_writes_flushed();
                            const uint32_t valid_semaphore_addr = get_semaphore(valid_semaphore_id);
                            noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
                        }

                        if constexpr (use_provided_mask) {
                            cb_reserve_back(cb_mask_in, mask_chunk_tiles);
                            uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
                            barrier_count = 0;

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
                                        noc_async_read_tile(
                                            mask_row_start + global_k_tile, mask_reader, mask_write_ptr);
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

                        // V: either read locally (injector or not participant) or receive from previous core
                        uint32_t cb_v_start_address = 0;
                        bool should_forward_v = false;
                        bool should_receive_v = false;

                        if constexpr (!is_causal) {
                            should_forward_v = is_chain_participant && !is_sink &&
                                               (nb == chain_batch && nq == chain_head) && (q_iter < next_core_q_chunks);
                            should_receive_v =
                                is_chain_participant && !is_injector && (nb == chain_batch && nq == chain_head);
                        }

                        if (should_receive_v) {
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
                                if (should_forward_v) {
                                    // Read without auto-push so we can capture address for forwarding
                                    cb_reserve_back(cb_v_in, v_chunk_tiles);
                                    cb_v_start_address = get_write_ptr(cb_v_in);
                                    // Manually read tiles without transpose (V reads are not transposed)
                                    constexpr bool transpose = false;
                                    const uint32_t skip_cols = DHt - vDHt;
                                    uint32_t tile_id = k_start_tile_id;  // V uses same addressing as K
                                    for (uint32_t row = 0; row < k_row_tile_count; ++row) {
                                        uint32_t write_ptr = cb_v_start_address + row * vDHt * v_tile_bytes;
                                        for (uint32_t col = 0; col < vDHt; ++col) {
                                            noc_async_read_tile(tile_id++, v_reader, write_ptr);
                                            write_ptr += v_tile_bytes;
                                        }
                                        tile_id += skip_cols;  // Skip columns
                                    }
                                    // Zero padding if needed
                                    for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                                        for (uint32_t col = 0; col < vDHt; ++col) {
                                            if (row < k_row_tile_count && col < vDHt) {
                                                continue;  // Already read
                                            }
                                            uint32_t tile_idx = row * vDHt + col;
                                            fill_tile_zeros<v_tile_bytes, false>(cb_v_in, tile_idx);
                                        }
                                    }
                                    noc_async_read_barrier();
                                    cb_push_back(cb_v_in, v_chunk_tiles);
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
                            }
                        }

                        // Forward V chunk to next core if applicable
                        if (should_forward_v) {
                            noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
                            noc_semaphore_set(sender_semaphore_addr_ptr, 0);
                            uint64_t v_unicast_data_addr =
                                get_noc_addr(next_physical_x, next_physical_y, cb_v_start_address);
                            noc_async_write(cb_v_start_address, v_unicast_data_addr, v_chunk_tiles * v_tile_bytes);
                            noc_async_writes_flushed();
                            const uint32_t valid_semaphore_addr = get_semaphore(valid_semaphore_id);
                            noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
                        }
                    }
                }
            }

            if constexpr (is_chunked) {
                cb_pop_front(cb_id_page_table, 1);
            }
        }
    }
}
