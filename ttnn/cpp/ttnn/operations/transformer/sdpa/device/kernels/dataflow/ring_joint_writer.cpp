// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"
#include "fused_op_receiver.hpp"

// Eager-path reader: reads the previous ring iteration's normalized output and LSE from DRAM.
// Used by the non-streaming (old sdpa_ring) path for sigmoid-based inter-iteration merging.
// Pushes output tiles into cb_prev_out (c_7) and LSE tiles into cb_lse_in (c_6).
//
// @param cat_out_generator   Address generator for the output DRAM tensor (local or joint)
// @param stats_writer        TensorAccessor for the stats DRAM tensor
// @param stats_tile_logical  Tile shape of the stats tensor (for address computation)
// @param nb                  Batch index
// @param nq                  Head index
// @param Sq_chunk_t          Q chunk size in tiles
// @param out_slice           Row/col tile range in the output tensor for this Q chunk
// @param end_seq_tile        Last valid sequence tile (for padding-aware reads)
// @param stats_seq_start_tile  First tile row in the stats tensor for this Q chunk
// @param stats_seq_end_tile    One-past-last tile row (clamped to avoid reading past padding)
// @param cb_prev_out         CB to push previous output tiles into (c_7, read by compute)
// @param cb_lse_in           CB to push previous LSE tiles into (c_6, read by compute)
// @param tile_bytes          Output tile size in bytes
// @param stats_tile_bytes    Stats tile size in bytes
template <typename ReaderType, typename TensorAccessorType>
void read_prev_output_and_lse(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t cb_prev_out,
    const uint32_t cb_lse_in,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    // Read previous output for this Q chunk
    read_block(cat_out_generator, out_slice, end_seq_tile, cb_prev_out, tile_bytes, false);

    // Read previous LSE for this Q chunk
    cb_reserve_back(cb_lse_in, Sq_chunk_t);
    uint32_t lse_addr = get_write_ptr(cb_lse_in);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_read_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, lse_addr);
        lse_addr += stats_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_lse_in, Sq_chunk_t);
}

// Issue restore reads for accumulators — non-blocking.
// Reserves CB space and issues NOC async reads for output, max, and sum.
// Call complete_restore() later to barrier and push.
//
// Split from the original read_prev_accumulators to enable prefetch: issue reads for Q[q+1]
// while Q[q]'s save is draining, hiding DRAM read latency behind compute.
template <typename ReaderType, typename TensorAccessorType>
void issue_restore_reads(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_prev_out,
    const uint32_t cb_max_in,
    const uint32_t cb_sum_in,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    // Issue output reads (reserve + async reads, no barrier)
    const uint32_t out_rows = out_slice.get_d2_size();
    const uint32_t out_cols = out_slice.get_d3_size();
    const uint32_t out_num_tiles = out_rows * out_cols;
    cb_reserve_back(cb_prev_out, out_num_tiles);
    const uint32_t out_base_ptr = get_write_ptr(cb_prev_out);
    for (uint32_t row = 0; row < out_rows; ++row) {
        uint32_t write_ptr = out_base_ptr + row * out_cols * tile_bytes;
        for (uint32_t col = 0; col < out_cols; ++col) {
            cat_out_generator.maybe_read_tile(
                out_slice.d0,
                out_slice.d1,
                out_slice.d2_start + row,
                out_slice.d3_start + col,
                end_seq_tile,
                write_ptr);
            write_ptr += tile_bytes;
        }
    }

    // Issue max reads
    cb_reserve_back(cb_max_in, Sq_chunk_t);
    uint32_t max_addr = get_write_ptr(cb_max_in);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_read_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, max_addr);
        max_addr += stats_tile_bytes;
    }

    // Issue sum reads
    cb_reserve_back(cb_sum_in, Sq_chunk_t);
    uint32_t sum_addr = get_write_ptr(cb_sum_in);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_read_tile(stats_tile_logical.id_of(nb, nq, sum_offset + i, 0), stats_writer, sum_addr);
        sum_addr += stats_tile_bytes;
    }
    // NO barrier, NO push — caller must call complete_restore()
}

// Complete a previously issued restore — single barrier for all 3 CBs, then push.
void complete_restore(
    const uint32_t cb_prev_out,
    const uint32_t out_num_tiles,
    const uint32_t cb_max_in,
    const uint32_t cb_sum_in,
    const uint32_t Sq_chunk_t) {
    noc_async_read_barrier();
    cb_push_back(cb_prev_out, out_num_tiles);
    cb_push_back(cb_max_in, Sq_chunk_t);
    cb_push_back(cb_sum_in, Sq_chunk_t);
}

// Wait for compute to push all save CBs (output, max, sum).
// Returns after all three cb_wait_fronts succeed — at this point compute has finished the K-loop
// for this Q chunk, and the restore CBs (cb_prev_out, cb_max_in, cb_sum_in) are free.
// This is the ideal time to issue prefetch reads for the next Q chunk.
void wait_for_save_cbs(
    const uint32_t Sq_chunk_t,
    const Slice& out_slice,
    const uint32_t cb_out,
    const uint32_t cb_max_out,
    const uint32_t cb_sum_out) {
    const uint32_t out_num_tiles = out_slice.get_d2_size() * out_slice.get_d3_size();
    cb_wait_front(cb_out, out_num_tiles);
    cb_wait_front(cb_max_out, Sq_chunk_t);
    cb_wait_front(cb_sum_out, Sq_chunk_t);
}

// Drain save CBs to DRAM after wait_for_save_cbs + any prefetch reads have been issued.
// Uses writes_flushed (not write_barrier) — sufficient because restore of Q[q+1] reads a
// different DRAM address than save of Q[q], so no RAW hazard within a ring iteration.
// The write_barrier at the end of each ring iteration ensures all saves land before the next
// iteration reads them back.
template <typename ReaderType, typename TensorAccessorType>
void drain_save_cbs(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice& out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_out,
    const uint32_t cb_max_out,
    const uint32_t cb_sum_out,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    // Write output to DRAM
    const uint32_t out_rows = out_slice.get_d2_size();
    const uint32_t out_cols = out_slice.get_d3_size();
    const uint32_t out_num_tiles = out_rows * out_cols;
    const uint32_t out_base_ptr = get_read_ptr(cb_out);
    for (uint32_t row = 0; row < out_rows; ++row) {
        uint32_t read_ptr = out_base_ptr + row * out_cols * tile_bytes;
        for (uint32_t col = 0; col < out_cols; ++col) {
            cat_out_generator.maybe_write_tile(
                out_slice.d0, out_slice.d1, out_slice.d2_start + row, out_slice.d3_start + col, end_seq_tile, read_ptr);
            read_ptr += tile_bytes;
        }
    }

    // Write max to stats DRAM
    uint32_t max_write_addr = get_read_ptr(cb_max_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, max_write_addr);
        max_write_addr += stats_tile_bytes;
    }

    // Write sum to stats DRAM (second half, offset by sum_offset)
    uint32_t sum_write_addr = get_read_ptr(cb_sum_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, sum_offset + i, 0), stats_writer, sum_write_addr);
        sum_write_addr += stats_tile_bytes;
    }

    // Flush: DMA has finished reading L1 source data — safe to free all save CBs.
    noc_async_writes_flushed();
    cb_pop_front(cb_out, out_num_tiles);
    cb_pop_front(cb_max_out, Sq_chunk_t);
    cb_pop_front(cb_sum_out, Sq_chunk_t);
}

// Eager-path writer: writes normalized output and LSE to DRAM every ring iteration.
// Used by the non-streaming (old sdpa_ring) path.
// Reads from: cb_out (c_16), cb_lse_out (c_17).
//
// @param cat_out_generator   Address generator for the output DRAM tensor (local or joint)
// @param stats_writer        TensorAccessor for the stats DRAM tensor
// @param stats_tile_logical  Tile shape of the stats tensor
// @param nb                  Batch index
// @param nq                  Head index
// @param Sq_chunk_t          Q chunk size in tiles
// @param out_slice           Row/col tile range in the output tensor for this Q chunk
// @param end_seq_tile        Last valid sequence tile
// @param stats_seq_start_tile  First tile row in stats tensor for this Q chunk's LSE
// @param stats_seq_end_tile    One-past-last tile row (clamped to sequence bounds)
// @param cb_out              CB to drain output tiles from (c_16)
// @param cb_lse_out          CB to drain LSE tiles from (c_17)
// @param tile_bytes          Output tile size in bytes
// @param stats_tile_bytes    Stats tile size in bytes
template <typename ReaderType, typename TensorAccessorType>
void write_output_and_lse(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t cb_out,
    const uint32_t cb_lse_out,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    write_block(cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes);

    cb_wait_front(cb_lse_out, Sq_chunk_t);
    uint32_t lse_addr = get_read_ptr(cb_lse_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, lse_addr);
        lse_addr += stats_tile_bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_lse_out, Sq_chunk_t);
}

struct QChunkInfo {
    bool is_joint_q;
    Slice out_slice;
    uint32_t end_seq_tile;
    uint32_t stats_seq_start_tile;
    uint32_t stats_seq_end_tile;
};

// Compute output slice and stats tile range for one Q chunk.
// is_joint_q distinguishes local-sequence Q chunks from joint-context Q chunks,
// which write to different output tensors and have different causal extents.
//
// @param q_chunk             Q chunk index within [0, num_q_chunks) (local then joint)
// @param nb                  Batch index
// @param nq                  Head index
// @param ring_id             Device ID that owns the current ring iteration's KV shard
// @param num_local_q_chunks  Number of Q chunks from the local sequence (joint starts after)
// @param Sq_chunk_t          Q chunk size in tiles
// @param DHt                 Head dimension in tiles
// @param Lt                  Joint (cross-attention context) sequence length in tiles
// @param local_padded_Nt     Per-device padded local sequence length in tiles
inline QChunkInfo get_q_chunk_info(
    const uint32_t q_chunk,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t ring_id,
    const uint32_t num_local_q_chunks,
    const uint32_t Sq_chunk_t,
    const uint32_t DHt,
    const uint32_t Lt,
    const uint32_t local_padded_Nt) {
    QChunkInfo info;
    info.is_joint_q = q_chunk >= num_local_q_chunks;
    if (info.is_joint_q) {
        const uint32_t joint_out_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
        info.out_slice = Slice(nb, nq, joint_out_row_start_tile, joint_out_row_start_tile + Sq_chunk_t, 0, DHt);
        info.end_seq_tile = Lt;
        info.stats_seq_start_tile = local_padded_Nt + (q_chunk - num_local_q_chunks) * Sq_chunk_t;
        info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
        info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, local_padded_Nt + Lt);
        info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, local_padded_Nt + Lt);
    } else {
        const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
        info.out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
        info.end_seq_tile = local_padded_Nt * (ring_id + 1);
        info.stats_seq_start_tile = q_chunk * Sq_chunk_t;
        info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
        info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, local_padded_Nt);
        info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, local_padded_Nt);
    }
    return info;
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(5);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t logical_n = get_compile_time_arg_val(8);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(9);
    constexpr uint32_t Lt = get_compile_time_arg_val(10);
    constexpr uint32_t L = get_compile_time_arg_val(11);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(17);
    constexpr uint32_t scale_val = get_compile_time_arg_val(18);
    constexpr uint32_t ring_size = get_compile_time_arg_val(19);
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(20);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(21);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(22) == 1;

    constexpr auto out_args = TensorAccessorArgs<23>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

    // c_6/c_17 carry softmax statistics between compute and writer for DRAM round-trips.
    // Aliased by role: cb_max_* for deferred norm, cb_lse_* for eager norm.
    constexpr uint32_t cb_max_in = tt::CBIndex::c_6;  // deferred norm: DRAM → compute (running max)
    constexpr uint32_t cb_lse_in = tt::CBIndex::c_6;  // eager norm: DRAM → compute (LSE)
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_max_out = tt::CBIndex::c_17;  // deferred norm: compute → DRAM (running max)
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;  // eager norm: compute → DRAM (LSE)
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_sum_out = tt::CBIndex::c_10;
    constexpr uint32_t cb_sum_in = tt::CBIndex::c_11;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_max_in);

    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr, tile_bytes);
    const auto stats_writer = TensorAccessor(stats_args, stats_addr, stats_tile_bytes);

    const auto output_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, DHt);
    // stats tensor is 2× the sequence length: first half stores max (used by both eager and
    // deferred-norm paths), second half stores sum (deferred-norm only).
    const auto stats_tile_logical = TensorTileShape(B, NH, (local_padded_Nt + Lt) * 2, 1);

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);
    const auto stats_generator = PaddedAddrGenerator(stats_writer, stats_tile_logical);

    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // Lightweight mask: generate all mask tiles once into single CB before the ring loop.
    // Only needed when any K/joint dimension has padding that doesn't fill a chunk.
    constexpr bool local_n_has_padding = local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool joint_has_padding = L > 0 && L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool needs_lightweight_mask = local_n_has_padding || global_n_has_padding || joint_has_padding;
    if constexpr (needs_lightweight_mask) {
        generate_lightweight_mask_tiles<global_n_partial_col, joint_l_partial_col, cb_mask_in>();
    }

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const uint32_t ring_iter_kv_end_tile = ring_iter_kv_start_tile + num_local_k_chunks * Sk_chunk_t;
        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);
        if (!ring_iter_does_work) {
            continue;
        }

        /**
        We have 3 possible masks
        - global N mask
        - local N mask
        - joint L mask

        Global N mask:
            - If the logical_n falls within this ring iter's KV range
            - And logical_n length (within local_padded_N) does not divide by K chunk size

        Local N mask
            - If local_padded_N does not divide by K chunk size, the last chunk needs a mask

        Joint L mask
            - If joint length L does not divide by K chunk size, the last chunk needs a mask
        */

        // GLOBAL N MASK
        // Find out if logical_n falls within this ring iter's KV range
        const int32_t global_n_within_ring_iter = logical_n - ring_id * local_padded_N;
        // Note the > and <=. This means there is real length of logical_n within this ring iter.
        const bool global_n_is_within_ring_iter =
            global_n_within_ring_iter > 0 && global_n_within_ring_iter <= (int32_t)local_padded_N;
        const bool global_n_needs_masking = global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_global_n_mask = global_n_is_within_ring_iter && global_n_needs_masking;

        // LOCAL N MASK
        const bool local_n_needs_masking = local_padded_Nt % Sk_chunk_t != 0;
        // If global N is in the ring iter, it supersedes the local N mask.
        const bool ring_iter_needs_local_n_mask = local_n_needs_masking && !global_n_is_within_ring_iter;

        // JOINT L MASK
        const bool joint_n_needs_masking = L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;

        // Deferred normalization is always paired with streaming compute.
        constexpr bool use_deferred_norm = use_streaming_compute;
        if constexpr (use_deferred_norm) {
            // Deferred norm: accumulates across ring iterations with exponential rescaling.
            // Single Q-chunk: accumulators persist in L1, write final output on last ring_iter.
            // Multi Q-chunk: raw accumulators round-trip through DRAM between ring iterations.
            //
            // Prefetch optimization: issue restore reads for Q[q+1] right after completing
            // Q[q]'s restore. cb_reserve_back blocks until compute pops Q[q]'s data, then
            // reads are issued and fly during Q[q]'s ENTIRE K-loop. This also works across
            // ring iteration boundaries: after Q[N-1]'s save drain, we prefetch Q[0] for the
            // next ring iteration. The reads fly during the write_barrier + ring sync + mask
            // setup, so Q[0]'s restore on the next iteration is instant.
            const bool is_last_ring_iter = (ring_iter == ring_size - 1);
            const bool single_q_chunk = (global_q_end - global_q_start == 1);
            constexpr uint32_t sum_offset = local_padded_Nt + Lt;
            constexpr uint32_t out_num_tiles = Sq_chunk_t * DHt;
            // For accumulator restore, all tiles are valid (we wrote them), so end_seq_tile
            // doesn't matter. Use a large value to bypass maybe_read_tile's padding logic.
            // This decouples restore prefetch from ring_id, enabling cross-ring-iteration prefetch.
            constexpr uint32_t restore_end_seq_tile = 0xFFFFFFFF;

            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi = get_q_chunk_info(
                    q_chunk, nb, nq, ring_id, num_local_q_chunks, Sq_chunk_t, DHt, Lt, local_padded_Nt);

                if (!single_q_chunk && ring_iter > 0) {
                    // Complete previously issued restore (barrier + push).
                    // Reads were prefetched during Q[q-1]'s K-loop (or during the prior
                    // ring iteration's tail for Q[0]) — should be instant.
                    complete_restore(cb_prev_out, out_num_tiles, cb_max_in, cb_sum_in, Sq_chunk_t);

                    // Prefetch: issue restore reads for Q[q+1].
                    // cb_reserve_back blocks until compute pops Q[q]'s restore data
                    // (via copy_block at the start of the K-loop). Once unblocked,
                    // reads are issued and fly during Q[q]'s entire K-loop.
                    const uint32_t next_q = global_q_chunk + 1;
                    if (next_q < global_q_end) {
                        const uint32_t nb_next = next_q / (NH * num_q_chunks);
                        const uint32_t nq_next = (next_q % (NH * num_q_chunks)) / num_q_chunks;
                        const uint32_t qc_next = next_q % num_q_chunks;
                        const auto qi_next = get_q_chunk_info(
                            qc_next,
                            nb_next,
                            nq_next,
                            ring_id,
                            num_local_q_chunks,
                            Sq_chunk_t,
                            DHt,
                            Lt,
                            local_padded_Nt);
                        issue_restore_reads(
                            qi_next.is_joint_q ? joint_out_generator : out_generator,
                            stats_writer,
                            stats_tile_logical,
                            nb_next,
                            nq_next,
                            Sq_chunk_t,
                            qi_next.out_slice,
                            restore_end_seq_tile,
                            qi_next.stats_seq_start_tile,
                            qi_next.stats_seq_end_tile,
                            sum_offset,
                            cb_prev_out,
                            cb_max_in,
                            cb_sum_in,
                            tile_bytes,
                            stats_tile_bytes);
                    }
                }

                // === Compute runs K-loop for this Q chunk ===
                // Prefetch reads for Q[q+1] are flying in parallel.
                // Writer blocks on cb_wait_front until compute pushes save/output data.

                if (is_last_ring_iter) {
                    write_block(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        qi.out_slice,
                        qi.end_seq_tile,
                        cb_out,
                        tile_bytes);
                } else if (!single_q_chunk) {
                    wait_for_save_cbs(Sq_chunk_t, qi.out_slice, cb_out, cb_max_out, cb_sum_out);
                    drain_save_cbs(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        stats_writer,
                        stats_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        qi.out_slice,
                        qi.end_seq_tile,
                        qi.stats_seq_start_tile,
                        qi.stats_seq_end_tile,
                        sum_offset,
                        cb_out,
                        cb_max_out,
                        cb_sum_out,
                        tile_bytes,
                        stats_tile_bytes);
                }
            }

            // Cross-ring-iteration prefetch: after Q[N-1]'s save, issue restore reads
            // for Q[0] of the NEXT ring iteration. Reads fly during the write_barrier,
            // ring sync, and mask setup — so Q[0]'s complete_restore is instant.
            // On ring_iter 0 the restore CBs were never used, so cb_reserve_back succeeds
            // immediately. On ring_iter > 0, compute has popped Q[N-1]'s restore data.
            // Q[0]'s save data is guaranteed in DRAM: it was written early in this ring
            // iteration and writes_flushed long ago (N-1 full Q cycles have elapsed).
            if (!single_q_chunk && !is_last_ring_iter) {
                noc_async_write_barrier();
                const uint32_t gq0 = global_q_start;
                const uint32_t nb0 = gq0 / (NH * num_q_chunks);
                const uint32_t nq0 = (gq0 % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t qc0 = gq0 % num_q_chunks;
                const auto qi0 = get_q_chunk_info(
                    qc0, nb0, nq0, 0 /*unused*/, num_local_q_chunks, Sq_chunk_t, DHt, Lt, local_padded_Nt);
                issue_restore_reads(
                    qi0.is_joint_q ? joint_out_generator : out_generator,
                    stats_writer,
                    stats_tile_logical,
                    nb0,
                    nq0,
                    Sq_chunk_t,
                    qi0.out_slice,
                    restore_end_seq_tile,
                    qi0.stats_seq_start_tile,
                    qi0.stats_seq_end_tile,
                    sum_offset,
                    cb_prev_out,
                    cb_max_in,
                    cb_sum_in,
                    tile_bytes,
                    stats_tile_bytes);
            } else {
                noc_async_write_barrier();
            }
        } else {
            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi = get_q_chunk_info(
                    q_chunk, nb, nq, ring_id, num_local_q_chunks, Sq_chunk_t, DHt, Lt, local_padded_Nt);

                // If not on the first iteration, read LSE and previous output chunk.
                // No race condition because writer kernel writes previous output before reading it again
                if (ring_iter > 0) {
                    read_prev_output_and_lse(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        stats_writer,
                        stats_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        qi.out_slice,
                        qi.end_seq_tile,
                        qi.stats_seq_start_tile,
                        qi.stats_seq_end_tile,
                        cb_prev_out,
                        cb_lse_in,
                        tile_bytes,
                        stats_tile_bytes);
                }

                write_output_and_lse(
                    qi.is_joint_q ? joint_out_generator : out_generator,
                    stats_writer,
                    stats_tile_logical,
                    nb,
                    nq,
                    Sq_chunk_t,
                    qi.out_slice,
                    qi.end_seq_tile,
                    qi.stats_seq_start_tile,
                    qi.stats_seq_end_tile,
                    cb_out,
                    cb_lse_out,
                    tile_bytes,
                    stats_tile_bytes);
            }
            noc_async_write_barrier();  // Ensure writes of output and LSE complete before next iteration
        }
    }
}
