// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "dataflow_common.hpp"

// Fetch a KV chunk into L1 for forwarding. No CB lifecycle — caller manages
// cb_reserve_back / cb_push_back. Single read barrier at end for lower latency.
template <uint32_t tile_bytes, bool transpose, typename ReaderType>
FORCE_INLINE void read_chunk_for_forwarding(
    const ReaderType& reader,
    const uint32_t cb_id,
    const uint32_t dst_addr,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t skip_src_cols = 0) {
    Noc noc;
    const uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    const uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t tile_id = start_tile_id;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = dst_addr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < src_cols; ++col) {
            noc.async_read(reader, CoreLocalMem<uint32_t>(write_ptr), tile_bytes, {.page_id = tile_id++}, {});
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
            fill_zeros_async(noc, cb_id, tile_bytes, tile_idx * tile_bytes);
        }
    }
    // On WH/BH, async_write_zeros is implemented via noc_async_read from MEM_ZEROS_BASE,
    // so async_read_barrier() covers both real reads and zero-fills on the same path.
    // On Quasar, iDMA zero uses a separate completion path (iDMA ack) that needs its own barrier.
    noc.async_read_barrier();
#ifdef ARCH_QUASAR
    noc.write_zeros_l1_barrier();
#endif
}

void kernel_main() {
    Noc noc;

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
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(27);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(28) == 1;

    // Semaphore IDs for KV chain forwarding (non-causal only, but always present in compile args)
    constexpr uint32_t sender_semaphore_id = get_compile_time_arg_val(29);
    constexpr uint32_t receiver_semaphore_id = get_compile_time_arg_val(30);
    constexpr uint32_t valid_semaphore_id = get_compile_time_arg_val(31);
    constexpr bool mcast_enabled = get_compile_time_arg_val(32) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(33) == 1;

    constexpr auto q_args = TensorAccessorArgs<34>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto chunk_start_idx_args = TensorAccessorArgs<attention_sink_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t attention_sink_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunk_start_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
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

    // Global Q scheduling runtime args (parsed after chain metadata so the host-side ordering
    // aligns for both causal and non-causal).
    uint32_t global_q_start = 0;
    uint32_t global_q_count = 0;

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

    // Initialize NOC/semaphore state for chain forwarding
    uint32_t sender_wait_count = 1;

    if constexpr (!is_causal) {
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
            Semaphore<>(valid_semaphore_id).set(VALID);

            if constexpr (mcast_enabled) {
                if (is_injector) {
                    sender_wait_count = mcast_sender_wait;
                }
            }
        }
    }

    // Global Q scheduling runtime args sit right after chain metadata.
    global_q_start = get_arg_val<uint32_t>(argidx++);
    global_q_count = get_arg_val<uint32_t>(argidx++);

    // When chunked: only process K/V up to (chunk_start_idx + Q_chunk_length) tokens.
    // valid_Skt_bound = min(offset_tiles + valid_Sqt, valid_Skt); cap at valid_Skt for callers that pass
    // different valid_Sqt (e.g. ring_distributed uses full Q length in tiles).

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;

    constexpr uint32_t cb_arg_offset = chunk_start_idx_args.next_compile_time_args_offset();
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_id_page_table = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_id_chunk_start_idx_compute = get_compile_time_arg_val(cb_arg_offset + 6);
    constexpr uint32_t cb_id_chunk_start_idx_writer = get_compile_time_arg_val(cb_arg_offset + 7);

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t mask_tile_bytes = use_provided_mask ? get_tile_size(cb_mask_in) : 0;
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

    CircularBuffer cb_k(cb_k_in);
    CircularBuffer cb_v(cb_v_in);
    CircularBuffer cb_mask(cb_mask_in);
    CircularBuffer cb_attn_sink(cb_attention_sink);
    CircularBuffer cb_page_table(cb_id_page_table);

    uint32_t chunked_q_chunk_offset = 0;
    if constexpr (is_chunked) {
        if (chunk_start_idx_addr != 0) {
            CircularBuffer cb_chunk_compute(cb_id_chunk_start_idx_compute);
            cb_chunk_compute.reserve_back(1);
            uint32_t chunk_start_write_ptr = cb_chunk_compute.get_write_ptr();
            noc.async_read(
                chunk_start_idx_reader, CoreLocalMem<uint32_t>(chunk_start_write_ptr), 4, {.page_id = 0}, {});
            noc.async_read_barrier();
            uint32_t chunk_start_idx = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chunk_start_write_ptr);
            cb_chunk_compute.push_back(1);

            CircularBuffer cb_chunk_writer(cb_id_chunk_start_idx_writer);
            cb_chunk_writer.reserve_back(1);
            uint32_t chunk_start_write_ptr_2 = cb_chunk_writer.get_write_ptr();
            noc.async_read(
                chunk_start_idx_reader, CoreLocalMem<uint32_t>(chunk_start_write_ptr_2), 4, {.page_id = 0}, {});
            noc.async_read_barrier();
            cb_chunk_writer.push_back(1);

            const uint32_t q_chunk_size = Sq_chunk_t * tt::constants::TILE_HEIGHT;
            chunked_q_chunk_offset_phase_1_local = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunked_q_chunk_offset_phase_2_local = chunked_q_chunk_offset_phase_1_local;
            }
        }
    }
    uint32_t read_offset = 0;
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_1_local;
            read_offset = read_offset_phase_1;
        } else {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_2_local;
            read_offset = read_offset_phase_2;
        }
        uint32_t valid_Skt_bound;
        if (chunk_start_idx_addr != 0) {
            // Flexible or ring: cap at valid_Skt so we never read past K/V extent.
            valid_Skt_bound = std::min(chunked_q_chunk_offset * Sq_chunk_t + valid_Sqt, valid_Skt);
        } else {
            // Legacy: extend by offset so one program can serve all chunks (valid_Skt is chunk 0's).
            valid_Skt_bound = valid_Skt + chunked_q_chunk_offset * Sq_chunk_t;
        }

        // Global Q scheduling: iterate over a linear range of B*NQH*q_num_chunks chunks.
        // - per_head_q_iter resets on (nb, nq) transition: chain forwarding's
        //   `q_iter < next_core_q_chunks` gate expects this (chains are non-causal only).
        // - is_chunked: page-table read on nb transition, single-entry CB rotated forward.
        // - use_attention_sink: pushed every iter, since compute pops Sq_chunk_t per
        //   sdpa_inner_loop call and under global_q each iter is exactly one call.
        uint32_t prev_nb = static_cast<uint32_t>(-1);
        uint32_t prev_nq = static_cast<uint32_t>(-1);
        uint32_t per_head_q_iter = 0;
        uint32_t mask_batch_offset = 0;
        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {
            const auto decoded =
                decompose_global_q_index(global_q_start + global_q_iter, q_num_chunks, NQH, use_zigzag_balancing);
            if (decoded.nb != prev_nb) {
                if constexpr (!broadcast_provided_mask_batch) {
                    if constexpr (broadcast_provided_mask_heads) {
                        mask_batch_offset = decoded.nb * valid_Sqt * valid_Skt;
                    } else {
                        mask_batch_offset = decoded.nb * valid_Sqt * valid_Skt * NQH;
                    }
                }
                if constexpr (is_chunked) {
                    if (prev_nb != static_cast<uint32_t>(-1)) {
                        cb_page_table.pop_front(1);
                    }
                    cb_page_table.reserve_back(1);
                    page_table_ptr = read_page_table_for_batch(
                        noc, cb_id_page_table, decoded.nb, page_table_args, page_table_addr, page_table_stick_size);
                    cb_page_table.push_back(1);
                }
            }
            if (decoded.nb != prev_nb || decoded.nq != prev_nq) {
                per_head_q_iter = 0;
                prev_nb = decoded.nb;
                prev_nq = decoded.nq;
            }
            if constexpr (use_attention_sink) {
                constexpr uint32_t sink_tiles = use_streaming_compute ? 1 : Sq_chunk_t;
                cb_attn_sink.reserve_back(sink_tiles);
                uint32_t attention_sink_write_ptr = cb_attn_sink.get_write_ptr();
                const uint32_t sink_tile_id = attention_sink_tile_shape.id_of(0, decoded.nq, 0, 0);
                noc.async_read(
                    attention_sink_reader,
                    CoreLocalMem<uint32_t>(attention_sink_write_ptr),
                    attention_sink_tile_bytes,
                    {.page_id = sink_tile_id},
                    {});
                noc.async_read_barrier();
                if constexpr (!use_streaming_compute) {
                    fill_attention_sink_tiles<attention_sink_tile_bytes>(
                        cb_attention_sink, sink_tiles, attention_sink_write_ptr);
                }
                cb_attn_sink.push_back(sink_tiles);
            }

            const uint32_t nb = decoded.nb;
            const uint32_t nq = decoded.nq;
            uint32_t q_chunk = decoded.q_chunk;
            const uint32_t q_iter = per_head_q_iter;
            ++per_head_q_iter;

            /*
            Determine how many rows of Q will be read. Both start and end rows are
            capped by valid_Sqt, since Sq padding is independent of Sk padding.
            */
            const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
            const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
            const uint32_t q_row_tile_count = q_row_end_tile - q_row_start_tile;
            uint32_t q_read_tile_id = q_tile_shape.id_of(nb, nq, read_offset + q_row_start_tile, 0);

            // Q read is deferred into the K loop (k_chunk==0) for subblock interleaving.
            // When use_q_subblock_push is false, Q is read in full before the K loop (original behavior).
            if constexpr (!use_q_subblock_push) {
                read_chunk_with_padding<q_tile_bytes>(
                    q_reader, cb_q_in, q_read_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);
            }

            q_chunk = chunked_q_chunk_offset + q_chunk;
            uint32_t q_low_idx = q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
            uint32_t q_high_idx;
            if constexpr (is_causal) {
                // Clamp to total K-tile extent (Skt = k_num_chunks * Sk_chunk_t). Without
                // this, when Q-chunk extends past K (e.g., Sq_chunk_t > k_num_chunks*Sk_chunk_t),
                // the K-loop pushes more chunks than compute consumes → CB deadlock.
                const uint32_t q_high_unclamped = q_low_idx + Sq_chunk_t;
                q_high_idx = q_high_unclamped < Skt ? q_high_unclamped : Skt;
            } else {
                q_high_idx = Skt;
            }
            uint32_t k_loop_start = 0;
            if constexpr (use_streaming_compute && sliding_window_size > 0) {
                // Must match the compute kernel's K-loop bounds (see sliding_window_geometry.hpp).
                using window_geom =
                    SlidingWindowLoopGeometry<sliding_window_size, is_causal, tt::constants::TILE_HEIGHT>;
                constexpr uint32_t left_window_tiles = window_geom::left_window_tiles;
                constexpr uint32_t right_window_tiles = window_geom::right_window_tiles;
                if (q_low_idx > left_window_tiles) {
                    k_loop_start = (q_low_idx - left_window_tiles) / Sk_chunk_t;
                }
                if constexpr (!is_causal) {
                    const uint32_t window_high_unclamped = q_low_idx + Sq_chunk_t + right_window_tiles;
                    q_high_idx = window_high_unclamped < Skt ? window_high_unclamped : Skt;
                }
            }

            const uint32_t k_head = nq / q_heads_per_k;
            const uint32_t v_head = nq / q_heads_per_v;

            // Chain forwarding conditions are loop-invariant — compute once
            bool should_forward = false;
            bool should_receive = false;
            if constexpr (!is_causal) {
                should_forward = is_chain_participant && !is_sink && (nb == chain_batch && nq == chain_head) &&
                                 (q_iter < next_core_q_chunks);
                should_receive = is_chain_participant && !is_injector && (nb == chain_batch && nq == chain_head);
            }

            // loop while k_low < q_high
            for (uint32_t k_chunk = k_loop_start; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                const uint32_t kv_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt_bound);
                const uint32_t kv_row_end_tile = std::min(kv_row_start_tile + Sk_chunk_t, valid_Skt_bound);
                const uint32_t kv_row_tile_count = kv_row_end_tile - kv_row_start_tile;
                const uint32_t k_start_tile_id = k_tile_shape.id_of(nb, k_head, kv_row_start_tile, 0);
                const uint32_t v_start_tile_id = v_tile_shape.id_of(nb, v_head, kv_row_start_tile, 0);

                // K: either read locally (injector or not participant) or receive from previous core
                uint32_t cb_k_start_address = 0;

                if (should_receive) {
                    // Receive forwarded K chunk from previous core
                    cb_k.reserve_back(k_chunk_tiles);
                    cb_k_start_address = cb_k.get_write_ptr();
                    Semaphore<> receiver_sem(receiver_semaphore_id);
                    receiver_sem.set(INVALID);
                    Semaphore<>(sender_semaphore_id).up(noc, prev_physical_x, prev_physical_y, 1);
                    receiver_sem.wait(VALID);
                    cb_k.push_back(k_chunk_tiles);
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
                            cb_k.reserve_back(k_chunk_tiles);
                            cb_k_start_address = cb_k.get_write_ptr();
                            read_chunk_for_forwarding<k_tile_bytes, true>(
                                k_reader,
                                cb_k_in,
                                cb_k_start_address,
                                k_start_tile_id,
                                kv_row_tile_count,
                                DHt,
                                Sk_chunk_t,
                                DHt);
                        } else {
                            read_chunk_with_padding<k_tile_bytes>(
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
                // any NOC read barrier between them deadlocks (the read barrier
                // blocks while a linked write awaits its companion).
                if (should_forward) {
                    Semaphore<> sender_sem(sender_semaphore_id);
                    sender_sem.wait(sender_wait_count);
                    sender_sem.set(0);
                    if constexpr (mcast_enabled) {
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(cb_k_start_address),
                            MulticastEndpoint{},
                            k_chunk_tiles * k_tile_bytes,
                            mcast_num_dests,
                            {},
                            {.noc_x_start = prev_physical_x,
                             .noc_y_start = prev_physical_y,
                             .noc_x_end = next_physical_x,
                             .noc_y_end = next_physical_y,
                             .addr = cb_k_start_address},
                            true /* linked: semaphore mcast follows */);
                        // Companion semaphore mcast: write the local valid_semaphore value into the
                        // remote receiver_semaphore slot (different L1 offset). Must be issued
                        // back-to-back after the linked data write — any noc.async_*_barrier()
                        // between them deadlocks the linked transaction.
                        Semaphore<>(valid_semaphore_id)
                            .relay_multicast(
                                noc,
                                Semaphore<>(receiver_semaphore_id),
                                prev_physical_x,
                                prev_physical_y,
                                next_physical_x,
                                next_physical_y,
                                mcast_num_dests,
                                /*linked=*/false);
                        noc.async_writes_flushed();
                        if (!should_receive) {
                            cb_k.push_back(k_chunk_tiles);
                        }
                    } else {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(cb_k_start_address),
                            UnicastEndpoint{},
                            k_chunk_tiles * k_tile_bytes,
                            {},
                            {.noc_x = next_physical_x, .noc_y = next_physical_y, .addr = cb_k_start_address});
                    }
                }

                // Mask read — safe after linked write pair is complete.
                // Stream the chunk one Q-tile-row (Sk_chunk_t tiles) at a time and push each row as
                // soon as it lands, so the compute kernel can start applying a q_subblock's rows
                // without waiting for the whole Sq_chunk_t x Sk_chunk_t block (matches the per-subblock
                // wait/pop in compute_streaming.hpp).
                if constexpr (use_provided_mask) {
                    uint32_t mask_row_start = mask_batch_offset + q_chunk * Sq_chunk_t * valid_Skt;
                    if constexpr (!broadcast_provided_mask_heads) {
                        mask_row_start += nq * valid_Sqt * valid_Skt;
                    }

                    for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                        const uint32_t global_q_tile = q_chunk * Sq_chunk_t + row;
                        const bool q_valid = !use_padded_mask || (global_q_tile < valid_Sqt);
                        cb_mask.reserve_back(Sk_chunk_t);
                        uint32_t mask_write_ptr = cb_mask.get_write_ptr();
                        uint32_t barrier_count = 0;
                        for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                            const uint32_t global_k_tile = k_chunk * Sk_chunk_t + col;
                            const bool k_valid = !use_padded_mask || (global_k_tile < valid_Skt);
                            if (q_valid && k_valid) {
                                noc.async_read(
                                    mask_reader,
                                    CoreLocalMem<uint32_t>(mask_write_ptr),
                                    mask_tile_bytes,
                                    {.page_id = mask_row_start + global_k_tile},
                                    {});
                            } else {
                                // reserve_back reset the write ptr to this row, so index by col.
                                fill_neginf_tile<mask_tile_bytes>(cb_mask_in, col);
                            }
                            mask_write_ptr += mask_tile_bytes;
                            if (++barrier_count == barrier_threshold) {
                                noc.async_read_barrier();
                                barrier_count = 0;
                            }
                        }
                        noc.async_read_barrier();
                        cb_mask.push_back(Sk_chunk_t);
                        if (q_valid) {
                            mask_row_start += valid_Skt;
                        }
                    }
                }

                // Complete K forward: flush write and signal receiver(s)
                // (mcast path already completed above — companion sent with linked write)
                if (should_forward) {
                    if constexpr (!mcast_enabled) {
                        noc.async_writes_flushed();
                        if (!should_receive) {
                            cb_k.push_back(k_chunk_tiles);
                        }
                        Semaphore<>(valid_semaphore_id)
                            .relay_unicast(noc, Semaphore<>(receiver_semaphore_id), next_physical_x, next_physical_y);
                    }
                }

                // Q subblock push: K is fully forwarded, now push Q one subblock at
                // a time. Compute waits for K first (waiting for K*N tiles in cb_k_in),
                // then waits for Q subblocks incrementally (accumulating waits on cb_q_in).
                // Each push unblocks the next QK subblock computation.
                // Placed after K forward complete so no outstanding NOC writes remain
                // (noc_async_read_barrier inside read_q_subblock deadlocks on BH
                // when NOC writes are in-flight).
                if constexpr (use_q_subblock_push) {
                    if (k_chunk == k_loop_start) {
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
                    cb_v.reserve_back(v_chunk_tiles);
                    cb_v_start_address = cb_v.get_write_ptr();
                    Semaphore<> receiver_sem(receiver_semaphore_id);
                    receiver_sem.set(INVALID);
                    Semaphore<>(sender_semaphore_id).up(noc, prev_physical_x, prev_physical_y, 1);
                    receiver_sem.wait(VALID);
                    cb_v.push_back(v_chunk_tiles);
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
                            cb_v.reserve_back(v_chunk_tiles);
                            cb_v_start_address = cb_v.get_write_ptr();
                            read_chunk_for_forwarding<v_tile_bytes, false>(
                                v_reader,
                                cb_v_in,
                                cb_v_start_address,
                                v_start_tile_id,
                                kv_row_tile_count,
                                vDHt,
                                Sk_chunk_t,
                                vDHt,
                                skip_src_cols);
                        } else {
                            read_chunk_with_padding<v_tile_bytes>(
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
                    Semaphore<> sender_sem(sender_semaphore_id);
                    sender_sem.wait(sender_wait_count);
                    sender_sem.set(0);
                    if constexpr (mcast_enabled) {
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(cb_v_start_address),
                            MulticastEndpoint{},
                            v_chunk_tiles * v_tile_bytes,
                            mcast_num_dests,
                            {},
                            {.noc_x_start = prev_physical_x,
                             .noc_y_start = prev_physical_y,
                             .noc_x_end = next_physical_x,
                             .noc_y_end = next_physical_y,
                             .addr = cb_v_start_address},
                            true /* linked: semaphore mcast follows */);
                        // Companion semaphore mcast — see K path above for rationale.
                        Semaphore<>(valid_semaphore_id)
                            .relay_multicast(
                                noc,
                                Semaphore<>(receiver_semaphore_id),
                                prev_physical_x,
                                prev_physical_y,
                                next_physical_x,
                                next_physical_y,
                                mcast_num_dests,
                                /*linked=*/false);
                    } else {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(cb_v_start_address),
                            UnicastEndpoint{},
                            v_chunk_tiles * v_tile_bytes,
                            {},
                            {.noc_x = next_physical_x, .noc_y = next_physical_y, .addr = cb_v_start_address});
                    }
                    noc.async_writes_flushed();
                    if constexpr (!mcast_enabled) {
                        Semaphore<>(valid_semaphore_id)
                            .relay_unicast(noc, Semaphore<>(receiver_semaphore_id), next_physical_x, next_physical_y);
                    }
                    if (!should_receive) {
                        cb_v.push_back(v_chunk_tiles);
                    }
                }
            }  // close k_chunk
        }  // close global_q_iter
        if constexpr (is_chunked) {
            if (prev_nb != static_cast<uint32_t>(-1)) {
                cb_page_table.pop_front(1);
            }
        }
    }  // close phase
}
