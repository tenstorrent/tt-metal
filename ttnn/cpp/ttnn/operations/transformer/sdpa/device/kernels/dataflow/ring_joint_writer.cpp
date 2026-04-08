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

// Non-blocking restore: reserves CB space and issues NOC reads for all 3 accumulators.
// Call complete_restore() later to barrier and push.
// Split from blocking read_prev_accumulators to enable prefetch: issue reads for Q[q+1]
// while Q[q]'s K-loop runs, hiding DRAM read latency behind compute.
template <typename ReaderType, typename TensorAccessorType>
void issue_restore_reads(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_prev_out,
    const uint32_t cb_max_in,
    const uint32_t cb_sum_in,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    // All accumulator tiles are valid (we wrote them), so bypass maybe_read_tile's
    // padding logic. This decouples restore from ring_id, enabling cross-ring prefetch.
    constexpr uint32_t end_seq_tile = 0xFFFFFFFF;

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

// Three transaction IDs for fine-grained write barrier tracking.
// Q[0] → TRID_FIRST, Q[1..N-2] → TRID_INNER, Q[N-1] → TRID_LAST.
// Only 3 barriers fire per ring iteration (one per TRID):
//   Q[0]: wB(TRID_INNER), Q[N-2]: wB(TRID_LAST), Q[N-1]: wB(TRID_FIRST).
// Start from 1 — TRID 0 is the default for all NOC writes and must not be used
// for per-TRID barriers, as unrelated writes (e.g. write_out_row_by_row on last
// ring iter) would inflate the outstanding count and stall the barrier.
constexpr uint32_t TRID_FIRST = 1;
constexpr uint32_t TRID_INNER = 2;
constexpr uint32_t TRID_LAST = 3;

// Row-by-row drain of output tiles from cb_out to DRAM.
// Waits for each row group (sbh tile-rows), writes to DRAM, pops.
// Overlaps DMA with compute: writes issue as soon as each row is ready.
template <typename ReaderType>
void write_out_row_by_row(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const Slice& out_slice,
    const uint32_t end_seq_tile,
    const uint32_t cb_out,
    const uint32_t tile_bytes,
    const uint32_t sbh) {
    const uint32_t out_rows = out_slice.get_d2_size();
    const uint32_t out_cols = out_slice.get_d3_size();
    const uint32_t row_tiles = sbh * out_cols;
    const uint32_t num_row_groups = out_rows / sbh;

    for (uint32_t rg = 0; rg < num_row_groups; ++rg) {
        cb_wait_front(cb_out, row_tiles);
        uint32_t read_ptr = get_read_ptr(cb_out);
        for (uint32_t r = 0; r < sbh; r++) {
            for (uint32_t col = 0; col < out_cols; ++col) {
                cat_out_generator.maybe_write_tile(
                    out_slice.d0,
                    out_slice.d1,
                    out_slice.d2_start + rg * sbh + r,
                    out_slice.d3_start + col,
                    end_seq_tile,
                    read_ptr);
                read_ptr += tile_bytes;
            }
        }
        cb_pop_front(cb_out, row_tiles);
    }
}

// Save all 3 accumulators (out, max, sum) to DRAM, tagged with a TRID for prefetch barriers.
// Output is drained row-by-row (overlapping with compute's SALAD pushes); max/sum are bulk-drained.
template <typename ReaderType, typename TensorAccessorType>
void save_accumulators_with_trid(
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
    const uint32_t stats_tile_bytes,
    const uint32_t sbh,
    const uint32_t save_trid) {
    noc_async_write_set_trid(save_trid);

    write_out_row_by_row(cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes, sbh);

    // Bulk drain of max/sum
    cb_wait_front(cb_max_out, Sq_chunk_t);
    cb_wait_front(cb_sum_out, Sq_chunk_t);

    uint32_t max_write_addr = get_read_ptr(cb_max_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, max_write_addr);
        max_write_addr += stats_tile_bytes;
    }

    uint32_t sum_write_addr = get_read_ptr(cb_sum_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, sum_offset + i, 0), stats_writer, sum_write_addr);
        sum_write_addr += stats_tile_bytes;
    }

    noc_async_write_flushed_with_trid(save_trid);
    // Reset TRID to 0 to avoid leaking it to unrelated writes (e.g. write_out_row_by_row on last ring iter).
    // Without this, subsequent noc_async_write calls would inflate save_trid's outstanding count,
    // causing noc_async_write_barrier_with_trid(save_trid) to wait for unrelated writes.
    noc_async_write_set_trid(0);
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
    uint32_t stats_seq_start_tile;
    uint32_t stats_seq_end_tile;
};

// Compute DRAM address info (output slice + stats range) for one Q chunk.
inline QChunkInfo get_q_chunk_info(
    const uint32_t q_chunk,
    const uint32_t nb,
    const uint32_t nq,
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
        info.stats_seq_start_tile = local_padded_Nt + (q_chunk - num_local_q_chunks) * Sq_chunk_t;
        info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
        info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, local_padded_Nt + Lt);
        info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, local_padded_Nt + Lt);
    } else {
        const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
        info.out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
        info.stats_seq_start_tile = q_chunk * Sq_chunk_t;
        info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
        info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, local_padded_Nt);
        info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, local_padded_Nt);
    }
    return info;
}

// Padding boundary for write paths — needs ring_id to know how far the valid sequence extends.
inline uint32_t get_end_seq_tile(const QChunkInfo& qi, uint32_t ring_id, uint32_t Lt, uint32_t local_padded_Nt) {
    return qi.is_joint_q ? Lt : local_padded_Nt * (ring_id + 1);
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NHK = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(7);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(8);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(9);
    constexpr uint32_t logical_n = get_compile_time_arg_val(10);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(11);
    constexpr uint32_t Lt = get_compile_time_arg_val(12);
    constexpr uint32_t L = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(14);

    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(17);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(19);
    constexpr uint32_t scale_val = get_compile_time_arg_val(20);
    constexpr uint32_t ring_size = get_compile_time_arg_val(21);
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(22);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(23);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t is_causal = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t is_balanced = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(27);

    constexpr auto out_args = TensorAccessorArgs<28>();
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
    constexpr uint32_t cb_signal = tt::CBIndex::c_12;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_max_in);

    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr, tile_bytes);
    const auto stats_writer = TensorAccessor(stats_args, stats_addr, stats_tile_bytes);

    const auto output_tile_logical = TensorTileShape(B, NH, local_padded_Nt, vDHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, vDHt);
    // stats tensor is 2× the sequence length: first half stores max (used by both eager and
    // deferred-norm paths), second half stores sum (deferred-norm only).
    const auto stats_tile_logical = TensorTileShape(B, NH, (local_padded_Nt + Lt) * 2, 1);

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);

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
    constexpr bool needs_lightweight_mask = (local_n_has_padding || global_n_has_padding || joint_has_padding) && !is_causal;
    if constexpr (needs_lightweight_mask) {
        generate_lightweight_mask_tiles<global_n_partial_col, joint_l_partial_col, cb_mask_in>();
    }

    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_receiver.seq, local_padded_Nt, logical_n / tt::constants::TILE_HEIGHT, L);

    uint32_t ring_index = fused_op_receiver.seq.ring_index;
    uint32_t half_sequence = num_q_chunks / 2;
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const uint32_t ring_iter_kv_end_tile = ring_iter_kv_start_tile + num_local_k_chunks * Sk_chunk_t;
        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_does_work = (ring_iter_processes_KV_chunks || (do_joint_kv && L != 0)) &&
                                         !(is_causal && ring_index < ring_id && !is_balanced);
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
            const bool is_last_ring_iter = (ring_iter == last_active_ring_iter);
            const bool single_q_chunk = (global_q_end - global_q_start == 1);
            constexpr uint32_t sum_offset = local_padded_Nt + Lt;
            constexpr uint32_t out_num_tiles = Sq_chunk_t * vDHt;

            const uint32_t q_per_core = global_q_end - global_q_start;
            const uint32_t last_q_index = q_per_core - 1;

            auto q_trid = [last_q_index](uint32_t q_idx) -> uint32_t {
                if (q_idx == 0) {
                    return TRID_FIRST;
                }
                if (q_idx == last_q_index) {
                    return TRID_LAST;
                }
                return TRID_INNER;
            };

            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                const uint32_t q_index = global_q_chunk - global_q_start;
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi =
                    get_q_chunk_info(q_chunk, nb, nq, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, local_padded_Nt);
                const uint32_t end_seq_tile = get_end_seq_tile(qi, ring_id, Lt, local_padded_Nt);

                // 1. Complete restore + intra-ring prefetch (ring_iter > 0 only)
                if (!single_q_chunk && ring_iter > 0) {
                    complete_restore(cb_prev_out, out_num_tiles, cb_max_in, cb_sum_in, Sq_chunk_t);

                    // Intra-ring prefetch: issue reads for Q[q+1] during Q[q]'s K-loop.
                    // Only barrier when next Q's TRID hasn't been cleared yet this ring iter:
                    //   Q[0]: wB(TRID_INNER) — clears all prev-ring inner saves
                    //   Q[N-2]: wB(TRID_LAST) — clears prev-ring Q[N-1] save
                    //   Q[1..N-3]: skip — TRID_INNER already cleared at Q[0]
                    // Without this, Q[j+1]'s wB(TRID_INNER) would stall on Q[j]'s
                    // current-ring save (near-zero flight time) for q_per_core >= 5.
                    const uint32_t next_q_index = q_index + 1;
                    if (next_q_index < q_per_core) {
                        const uint32_t next_trid = q_trid(next_q_index);
                        // First inner Q (next_q_index==1) needs the barrier; subsequent
                        // inner Qs (next_q_index>1 && next_trid==TRID_INNER) don't.
                        if (next_trid != TRID_INNER || next_q_index == 1) {
                            noc_async_write_barrier_with_trid(next_trid);
                        }
                        const uint32_t next_q_global = global_q_chunk + 1;
                        const uint32_t nb_next = next_q_global / (NH * num_q_chunks);
                        const uint32_t nq_next = (next_q_global % (NH * num_q_chunks)) / num_q_chunks;
                        const uint32_t qc_next = next_q_global % num_q_chunks;
                        const auto qi_next = get_q_chunk_info(
                            qc_next, nb_next, nq_next, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, local_padded_Nt);
                        issue_restore_reads(
                            qi_next.is_joint_q ? joint_out_generator : out_generator,
                            stats_writer,
                            stats_tile_logical,
                            nb_next,
                            nq_next,
                            Sq_chunk_t,
                            qi_next.out_slice,
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

                // 2. Cross-ring prefetch: Q[N-1] → Q[0] of next ring iter.
                // Reads fly during Q[N-1]'s K-loop + save drain.
                if (!single_q_chunk && !is_last_ring_iter && (global_q_chunk + 1 >= global_q_end)) {
                    noc_async_write_barrier_with_trid(TRID_FIRST);
                    const uint32_t gq0 = global_q_start;
                    const uint32_t nb0 = gq0 / (NH * num_q_chunks);
                    const uint32_t nq0 = (gq0 % (NH * num_q_chunks)) / num_q_chunks;
                    const uint32_t qc0 = gq0 % num_q_chunks;
                    const auto qi0 =
                        get_q_chunk_info(qc0, nb0, nq0, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, local_padded_Nt);
                    issue_restore_reads(
                        qi0.is_joint_q ? joint_out_generator : out_generator,
                        stats_writer,
                        stats_tile_logical,
                        nb0,
                        nq0,
                        Sq_chunk_t,
                        qi0.out_slice,
                        qi0.stats_seq_start_tile,
                        qi0.stats_seq_end_tile,
                        sum_offset,
                        cb_prev_out,
                        cb_max_in,
                        cb_sum_in,
                        tile_bytes,
                        stats_tile_bytes);
                }

                // === Compute runs K-loop for this Q chunk ===

                // Wait for compute to signal last K-chunk start (multi-Q only).
                if (!single_q_chunk) {
                    cb_wait_front(cb_signal, 1);
                    cb_pop_front(cb_signal, 1);
                }

                if (is_last_ring_iter) {
                    write_out_row_by_row(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        qi.out_slice,
                        end_seq_tile,
                        cb_out,
                        tile_bytes,
                        out_subblock_h);
                    noc_async_write_barrier();
                } else if (!single_q_chunk) {
                    // Accumulators are raw compute state — all tiles are valid (including padded rows),
                    // so bypass maybe_write_tile's padding skip (same convention as restore reads).
                    constexpr uint32_t all_tiles_valid = 0xFFFFFFFF;
                    save_accumulators_with_trid(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        stats_writer,
                        stats_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        qi.out_slice,
                        all_tiles_valid,
                        qi.stats_seq_start_tile,
                        qi.stats_seq_end_tile,
                        sum_offset,
                        cb_out,
                        cb_max_out,
                        cb_sum_out,
                        tile_bytes,
                        stats_tile_bytes,
                        out_subblock_h,
                        q_trid(q_index));
                }
            }
            // No global write_barrier — per-trid barriers in the Q loop
            // ensure each save has landed before its data is read back.
        } else {
            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi =
                    get_q_chunk_info(q_chunk, nb, nq, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, local_padded_Nt);
                const uint32_t end_seq_tile = get_end_seq_tile(qi, ring_id, Lt, local_padded_Nt);

                // Only truly causal case appear in the iteration with local KV
                // Other iterations will just skip the computation with subsequent KV chunks
                bool causality = (ring_iter == 0 ? is_causal : false);

                if (q_chunk < half_sequence && is_balanced && ring_index < ring_id) {
                    continue;
                }

                if (is_causal) {
                    generate_mask<false, 0, true, cb_mask_in>(
                        Sq_chunk_t,
                        Sk_chunk_t,
                        q_chunk,
                        0,
                        ring_iter_needs_global_n_mask || ring_iter_needs_local_n_mask,
                        ring_iter_needs_joint_n_mask,
                        ring_iter_needs_global_n_mask ? global_n_within_ring_iter : local_padded_N,
                        L,
                        causality);
                }

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
                        end_seq_tile,
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
                    end_seq_tile,
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
