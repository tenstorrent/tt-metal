// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"
#include "ring_joint_kv_pad_derivation.hpp"
#include "fused_op_receiver.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Eager-path reader: reads the previous ring iteration's normalized output and LSE from DRAM.
// Used by the non-streaming (old sdpa_ring) path for sigmoid-based inter-iteration merging.
// Pushes output tiles into cb_prev_out and LSE tiles into cb_lse_in.
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
// @param cb_prev_out         CB to push previous output tiles into (read by compute)
// @param cb_lse_in           CB to push previous LSE tiles into (read by compute)
// @param tile_bytes          Output tile size in bytes
// @param stats_tile_bytes    Stats tile size in bytes
template <typename CatAddrGeneratorType, typename TensorAccessorType, typename StatsShapeType>
void read_prev_output_and_lse(
    Noc noc,
    const CatAddrGeneratorType& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const StatsShapeType& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice& out_slice,
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
    CircularBuffer cb_lse(cb_lse_in);
    cb_lse.reserve_back(Sq_chunk_t);
    uint32_t lse_addr = cb_lse.get_write_ptr();
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc.async_read(
            stats_writer,
            CoreLocalMem<uint32_t>(lse_addr),
            stats_tile_bytes,
            {.page_id = stats_tile_logical.id_of(nb, nq, i, 0)},
            {});
        lse_addr += stats_tile_bytes;
    }
    noc.async_read_barrier();
    cb_lse.push_back(Sq_chunk_t);
}

template <typename TensorAccessorType, typename StatsShapeType>
static __attribute__((noinline, noclone)) void issue_stats_column_reads(
    Noc noc,
    const TensorAccessorType& stats_writer,
    const StatsShapeType& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t row_start,
    const uint32_t num_rows,
    const uint32_t cb_id,
    const uint32_t reserve_tiles,
    const uint32_t stats_tile_bytes) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(reserve_tiles);
    uint32_t tile_id = stats_tile_logical.id_of(nb, nq, row_start, 0);
    const uint32_t row_stride = stats_tile_logical.stride2();
    uint32_t addr = cb.get_write_ptr();
    for (uint32_t r = 0; r < num_rows; ++r) {
        noc.async_read(stats_writer, CoreLocalMem<uint32_t>(addr), stats_tile_bytes, {.page_id = tile_id}, {});
        tile_id += row_stride;
        addr += stats_tile_bytes;
    }
}

template <typename TensorAccessorType, typename StatsShapeType>
static __attribute__((noinline, noclone)) void issue_stats_column_writes(
    Noc noc,
    const TensorAccessorType& stats_writer,
    const StatsShapeType& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t row_start,
    const uint32_t num_rows,
    const uint32_t cb_id,
    const uint32_t stats_tile_bytes,
    const uint32_t trid = 0) {
    CircularBuffer cb(cb_id);
    uint32_t tile_id = stats_tile_logical.id_of(nb, nq, row_start, 0);
    const uint32_t row_stride = stats_tile_logical.stride2();
    uint32_t addr = cb.get_read_ptr();
    for (uint32_t r = 0; r < num_rows; ++r) {
        noc.async_write<NocOptions::TXN_ID>(
            CoreLocalMem<uint32_t>(addr), stats_writer, stats_tile_bytes, {}, {.page_id = tile_id}, {.trid = trid});
        tile_id += row_stride;
        addr += stats_tile_bytes;
    }
}

// Non-blocking restore: reserves CB space and issues NOC reads for all 3 accumulators.
// Call complete_restore() later to barrier and push.
// Split from blocking read_prev_accumulators to enable prefetch: issue reads for Q[q+1]
// while Q[q]'s K-loop runs, hiding DRAM read latency behind compute.
template <typename CatAddrGeneratorType, typename TensorAccessorType, typename StatsShapeType>
void issue_restore_reads(
    Noc noc,
    const CatAddrGeneratorType& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const StatsShapeType& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice& out_slice,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_prev_out,
    const uint32_t cb_max_in,
    const uint32_t cb_sum_in,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    // All accumulator tiles are valid (we wrote them) — bypass issue_reads' bound/valid_rows
    // compute and the zero-fill stub; dispatch directly to issue_block_reads. Decouples
    // restore from ring_id, enabling cross-ring prefetch.
    const uint32_t out_rows = out_slice.get_d2_size();
    const uint32_t out_cols = out_slice.get_d3_size();
    const uint32_t out_num_tiles = out_rows * out_cols;
    CircularBuffer cb_prev(cb_prev_out);
    cb_prev.reserve_back(out_num_tiles);
    uint32_t out_barrier_count = 0;
    issue_block_reads(
        cat_out_generator.reader,
        cat_out_generator.tensor_shape.id_of(out_slice.d0, out_slice.d1, out_slice.d2_start, out_slice.d3_start),
        cat_out_generator.tensor_shape.stride2(),
        out_rows,
        out_cols,
        /*dst_row_origin=*/0,
        cb_prev.get_write_ptr(),
        /*outer_stride=*/out_cols * tile_bytes,
        /*inner_stride=*/tile_bytes,
        /*barrier_threshold=*/0,
        out_barrier_count);

    // Stats drains: single-column linear reads. Hoist id_of once per drain; advance by
    // strides[2] per row. All tiles assumed valid (no bounds clamp needed).
    const uint32_t stats_rows = stats_seq_end_tile - stats_seq_start_tile;

    issue_stats_column_reads(
        noc,
        stats_writer,
        stats_tile_logical,
        nb,
        nq,
        stats_seq_start_tile,
        stats_rows,
        cb_max_in,
        Sq_chunk_t,
        stats_tile_bytes);
    issue_stats_column_reads(
        noc,
        stats_writer,
        stats_tile_logical,
        nb,
        nq,
        sum_offset + stats_seq_start_tile,
        stats_rows,
        cb_sum_in,
        Sq_chunk_t,
        stats_tile_bytes);
    // NO barrier, NO push — caller must call complete_restore()
}

// Complete a previously issued restore — single barrier for all 3 CBs, then push.
void complete_restore(
    Noc noc,
    const uint32_t cb_prev_out,
    const uint32_t out_num_tiles,
    const uint32_t cb_max_in,
    const uint32_t cb_sum_in,
    const uint32_t Sq_chunk_t) {
    noc.async_read_barrier();
    CircularBuffer(cb_prev_out).push_back(out_num_tiles);
    CircularBuffer(cb_max_in).push_back(Sq_chunk_t);
    CircularBuffer(cb_sum_in).push_back(Sq_chunk_t);
}

// Three transaction IDs for fine-grained write barrier tracking.
// Q[0] → TRID_FIRST, Q[1..N-2] → TRID_INNER, Q[N-1] → TRID_LAST.
// Only 3 barriers fire per ring iteration (one per TRID):
//   Q[0]: wB(TRID_INNER), Q[N-2]: wB(TRID_LAST), Q[N-1]: wB(TRID_FIRST).
// Start from 1 — TRID 0 is the default for all NOC writes and must not be used
// for per-TRID barriers, as unrelated writes (e.g. write_block_row_grouped_trid
// on last ring iter) would inflate the outstanding count and stall the barrier.
constexpr uint32_t TRID_FIRST = 1;
constexpr uint32_t TRID_INNER = 2;
constexpr uint32_t TRID_LAST = 3;

// Save all 3 accumulators (out, max, sum) to DRAM, tagged with a TRID for prefetch barriers.
// Output is drained row-by-row (overlapping with compute's SALAD pushes); max/sum are bulk-drained.
template <
    bool all_output_rows_valid = false,
    typename CatAddrGeneratorType,
    typename TensorAccessorType,
    typename StatsShapeType>
void save_accumulators_with_trid(
    Noc noc,
    const CatAddrGeneratorType& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const StatsShapeType& stats_tile_logical,
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
    // Each write is tagged per-call with save_trid (no sticky-set state needed). The matching
    // noc.async_write_barrier<NocOptions::TXN_ID>({.trid=save_trid}) downstream waits exactly
    // for these writes — no risk of trid leaking to unrelated writes since the tag is local
    // to each call.

    write_block_row_grouped_trid<all_output_rows_valid>(
        noc, cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes, sbh, save_trid);

    // Bulk drain of max/sum
    CircularBuffer cb_max(cb_max_out);
    CircularBuffer cb_sum(cb_sum_out);
    cb_max.wait_front(Sq_chunk_t);
    cb_sum.wait_front(Sq_chunk_t);

    const uint32_t stats_rows = stats_seq_end_tile - stats_seq_start_tile;
    issue_stats_column_writes(
        noc,
        stats_writer,
        stats_tile_logical,
        nb,
        nq,
        stats_seq_start_tile,
        stats_rows,
        cb_max_out,
        stats_tile_bytes,
        save_trid);
    issue_stats_column_writes(
        noc,
        stats_writer,
        stats_tile_logical,
        nb,
        nq,
        sum_offset + stats_seq_start_tile,
        stats_rows,
        cb_sum_out,
        stats_tile_bytes,
        save_trid);

    noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = save_trid});
    // cb_out was already popped per-group inside write_block_row_grouped_trid.
    cb_max.pop_front(Sq_chunk_t);
    cb_sum.pop_front(Sq_chunk_t);
}

// Eager-path writer: writes normalized output and LSE to DRAM every ring iteration.
// Used by the non-streaming (old sdpa_ring) path.
// Reads from: cb_out and cb_lse_out.
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
// @param cb_out              CB to drain output tiles from
// @param cb_lse_out          CB to drain LSE tiles from
// @param tile_bytes          Output tile size in bytes
// @param stats_tile_bytes    Stats tile size in bytes
template <typename CatAddrGeneratorType, typename TensorAccessorType, typename StatsShapeType>
void write_output_and_lse(
    Noc noc,
    const CatAddrGeneratorType& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const StatsShapeType& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice& out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t cb_out,
    const uint32_t cb_lse_out,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    write_block(noc, cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes);

    CircularBuffer cb_lse(cb_lse_out);
    cb_lse.wait_front(Sq_chunk_t);
    uint32_t lse_addr = cb_lse.get_read_ptr();
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc.async_write(
            CoreLocalMem<uint32_t>(lse_addr),
            stats_writer,
            stats_tile_bytes,
            {},
            {.page_id = stats_tile_logical.id_of(nb, nq, i, 0)});
        lse_addr += stats_tile_bytes;
    }
    noc.async_writes_flushed();
    cb_lse.pop_front(Sq_chunk_t);
}

struct QChunkInfo {
    bool is_joint_q;
    Slice out_slice;
    uint32_t stats_seq_start_tile;
    uint32_t stats_seq_end_tile;
};

// Compute DRAM address info (output slice + stats range) for one Q chunk.
// has_joint_q: when false, joint branch is statically dead and dropped by the compiler.
template <bool has_joint_q>
inline QChunkInfo get_q_chunk_info(
    const uint32_t q_chunk,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t num_local_q_chunks,
    const uint32_t Sq_chunk_t,
    const uint32_t DHt,
    const uint32_t Lt,
    const uint32_t q_local_padded_Nt) {
    QChunkInfo info;
    if constexpr (has_joint_q) {
        info.is_joint_q = q_chunk >= num_local_q_chunks;
        if (info.is_joint_q) {
            const uint32_t joint_out_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
            info.out_slice = Slice(nb, nq, joint_out_row_start_tile, joint_out_row_start_tile + Sq_chunk_t, 0, DHt);
            info.stats_seq_start_tile = q_local_padded_Nt + (q_chunk - num_local_q_chunks) * Sq_chunk_t;
            info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
            info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, q_local_padded_Nt + Lt);
            info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, q_local_padded_Nt + Lt);
            return info;
        }
    } else {
        info.is_joint_q = false;
    }
    const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
    info.out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
    info.stats_seq_start_tile = q_chunk * Sq_chunk_t;
    info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
    info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, q_local_padded_Nt);
    info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, q_local_padded_Nt);
    return info;
}

// Padding boundary for write paths — needs ring_id to know how far the valid sequence extends.
// has_joint_q: when false, joint branch is statically dead.
template <bool has_joint_q>
inline uint32_t get_end_seq_tile(const QChunkInfo& qi, uint32_t ring_id, uint32_t Lt, uint32_t q_local_padded_Nt) {
    if constexpr (has_joint_q) {
        return qi.is_joint_q ? Lt : q_local_padded_Nt * (ring_id + 1);
    } else {
        return q_local_padded_Nt * (ring_id + 1);
    }
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NHK = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(8);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(9);
    constexpr uint32_t logical_n = get_compile_time_arg_val(10);
    // Slot 11 is retained for compile-time arg index stability; live logical_nt is a runtime arg below.
    constexpr uint32_t logical_nt_compile [[maybe_unused]] = get_compile_time_arg_val(11);
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
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(27) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(28);
    constexpr bool chunked_enabled = get_compile_time_arg_val(29) == 1;
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(30);
    // Slots 31-33 are retained for compile-time arg index stability; live ring-work masks
    // are runtime args below.
    constexpr uint32_t active_ring_iter_mask_compile [[maybe_unused]] = get_compile_time_arg_val(31);
    constexpr uint32_t last_active_ring_iter_compile [[maybe_unused]] = get_compile_time_arg_val(32);
    constexpr uint32_t single_valid_kv_chunk_mask_compile [[maybe_unused]] = get_compile_time_arg_val(33);
    // Diagonal-mask tile slot is shared by the kernel's is_causal path and the chunked-prefill
    // path. The program factory masks kernel_is_causal off when chunked is on, so only one of
    // the two paths drives the stamp per program — but they share the CB slot layout.
    constexpr bool diag_tile_enabled = (is_causal == 1) || chunked_enabled;
    // Slot 34: trace-safe KV-pad derivation. When set, the writer reads kv_actual_isl from the
    // kv_actual_isl tensor[0] (common runtime arg 0 = its DRAM addr) and recomputes logical_nt + ring
    // masks on-device (it's a dataflow kernel, can NoC-read), so a captured trace replays across chunks.
    // Output accessors therefore start at compile-arg slot 35.
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(34) == 1;

    // Joint-path compile-time gating. When zero, joint Q/K branches are statically dead
    // and dropped by the compiler, eliminating runtime ternaries and the joint_out_generator.
    constexpr bool has_joint_q = num_joint_q_chunks > 0;
    constexpr bool has_joint_k = num_joint_k_chunks > 0;

    constexpr auto out_args = TensorAccessorArgs<35>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();
    // Metadata accessor (metadata path only) follows the output accessors and precedes the CB compile
    // args; gate the offset on kv_pad_from_metadata so the no-metadata program never names a non-accessor
    // compile arg (fall back to a valid unused accessor offset = out_args' slot 35).
    constexpr uint32_t meta_args_offset = kv_pad_from_metadata ? stats_args.next_compile_time_args_offset() : 35;
    constexpr auto meta_args = TensorAccessorArgs<meta_args_offset>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    // Mutable: on the kv_pad_from_metadata path these are recomputed on-device below from metadata[1].
    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    uint32_t single_valid_kv_chunk_mask = get_arg_val<uint32_t>(argidx++);
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

    // The stats CB is aliased by role: cb_max_* for deferred norm, cb_lse_* for eager norm. The CB
    // compile args start after the metadata accessor when it is present (kv_pad_from_metadata).
    constexpr uint32_t cb_arg_offset =
        kv_pad_from_metadata ? meta_args.next_compile_time_args_offset() : stats_args.next_compile_time_args_offset();
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_scale_in = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_max_in = get_compile_time_arg_val(cb_arg_offset + 6);  // deferred norm: DRAM -> compute
    constexpr uint32_t cb_lse_in = cb_max_in;                                    // eager norm: DRAM -> compute
    constexpr uint32_t cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_sum_out = get_compile_time_arg_val(cb_arg_offset + 10);
    constexpr uint32_t cb_sum_in = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_signal = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_max_out = get_compile_time_arg_val(cb_arg_offset + 14);  // deferred norm: compute -> DRAM
    constexpr uint32_t cb_lse_out = cb_max_out;                                    // eager norm: compute -> DRAM

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_max_in);

    Noc noc;

    // Trace-safe KV-pad derivation: recompute logical_nt + ring masks on-device from kv_actual_isl =
    // kv_actual_isl tensor[0] (the per-core logical_nt/masks args are placeholders on the metadata path
    // and would be frozen by a captured trace). The writer is a dataflow kernel, so it reads the tensor
    // directly via NoC into cb_out's L1 scratch (free before the output write loop). chunk_size_t ==
    // q_chunk_group.
    if constexpr (kv_pad_from_metadata) {
        // kv_actual_isl is a 1-element uint32 DRAM tensor (was metadata[1]); its DRAM address is common
        // runtime arg 0. Read its page 0 (4B).
        const uint32_t kv_actual_isl_addr = get_common_arg_val<uint32_t>(0);
        const auto s_meta = TensorAccessor(meta_args, kv_actual_isl_addr);
        CircularBuffer cb_meta_scratch(cb_out);
        const uint32_t meta_l1 = cb_meta_scratch.get_write_ptr();
        noc.async_read(s_meta, CoreLocalMem<uint32_t>(meta_l1), 4, {.page_id = 0}, {});
        noc.async_read_barrier();
        CoreLocalMem<volatile uint32_t> meta(meta_l1);
        const uint32_t kv_actual_isl = meta[0];
        logical_nt = ring_joint::compute_logical_nt(kv_actual_isl, chunk_size_t * 32, 32);
        const auto masks = ring_joint::build_ring_work_masks_device(
            fused_op_receiver.seq.ring_index,
            ring_size,
            fused_op_receiver.seq.expected[0],  // backward_writes_expected
            fused_op_receiver.seq.expected[1],  // forward_writes_expected
            num_local_k_chunks,
            Sk_chunk_t,
            kv_local_padded_Nt,
            chunked_enabled,
            chunk_size_t,
            q_local_padded_Nt,
            logical_nt,
            num_joint_k_chunks,
            L,
            /*kv_pad_rotation_enabled=*/true,
            is_causal != 0,
            is_balanced != 0);
        active_ring_iter_mask = masks.active_ring_iter_mask;
        single_valid_kv_chunk_mask = masks.single_valid_kv_chunk_mask;
    }

    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr);
    const auto stats_writer = TensorAccessor(stats_args, stats_addr);

    constexpr bool output_has_no_padding = !has_joint_q && (q_local_padded_Nt % Sq_chunk_t == 0);
    using StaticOutputTileShape = StaticTensorTileShape<B, NH, q_local_padded_Nt, vDHt>;
    using OutputTileShape = std::conditional_t<output_has_no_padding, StaticOutputTileShape, TensorTileShape>;

    const auto output_tile_logical = OutputTileShape(B, NH, q_local_padded_Nt, vDHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, vDHt);
    // stats tensor is 2× the sequence length: first half stores max (used by both eager and
    // deferred-norm paths), second half stores sum (deferred-norm only).
    const auto stats_tile_logical = StaticTensorTileShape<B, NH, (q_local_padded_Nt + Lt) * 2, 1>();

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);

    generate_bcast_unary_scalar(CircularBuffer(cb_scale_in), scale_val);
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

    // Lightweight mask: generate all mask tiles once into single CB before the ring loop.
    // Needed when any K/joint dimension has padding, or when causal/chunked masking is active.
    constexpr bool local_n_has_padding = kv_local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool joint_has_padding = L > 0 && L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) || diag_tile_enabled;
    if constexpr (needs_lightweight_mask) {
        generate_lightweight_mask_tiles<global_n_partial_col, joint_l_partial_col, cb_mask_in, diag_tile_enabled>(noc);
    }

    uint32_t ring_index = fused_op_receiver.seq.ring_index;
    uint32_t half_sequence = num_q_chunks / 2;

    // Deferred save: stash params for save_accumulators_with_trid and call it
    // during the next Q chunk's K-loop window to avoid DRAM bank contention.
    struct DeferredWriteContext {
        bool pending = false;
        uint32_t trid = 0;
        uint32_t nb = 0;
        uint32_t nq = 0;
        QChunkInfo qi = {};
    } deferred = {};

    // Track non-skipped iters so the first active iter starts with fresh accumulators (matches compute).
    bool seen_active_iter = false;
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
        // still advances above so writer stays aligned with reader, compute, and all-gather.
        if (((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
            continue;
        }
        const bool do_joint_kv = ring_id == ring_size - 1;
        uint32_t num_kv_chunks = num_local_k_chunks;
        if constexpr (has_joint_k) {
            if (do_joint_kv) {
                num_kv_chunks += num_joint_k_chunks;
            }
        }

        const bool is_first_active_iter = !seen_active_iter;
        seen_active_iter = true;

        // When a ring iteration has one valid K chunk, compute saves to staging on K0,
        // reserving staging CBs immediately. The deferred flush must happen before any
        // prefetch that blocks on cb_prev_out, or the writer and compute deadlock.
        const bool single_valid_kv_chunk = ((single_valid_kv_chunk_mask >> ring_iter) & 1u) != 0;

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

        // GLOBAL N MASK — tile-aligned form. In chunked-prefill mode this whole mask path is
        // disabled: global_n_is_within_ring_iter is gated on !chunked_enabled below, so the
        // skip-by-per-k-chunk-start logic in compute handles the trailing real-region boundary
        // instead.
        const int32_t global_nt_within_ring_iter =
            static_cast<int32_t>(logical_nt) - static_cast<int32_t>(ring_id * kv_local_padded_Nt);
        const bool global_n_is_within_ring_iter =
            !chunked_enabled &&
            (global_nt_within_ring_iter > 0 && global_nt_within_ring_iter <= (int32_t)kv_local_padded_Nt);
        const bool global_n_needs_masking = (global_nt_within_ring_iter % (int32_t)Sk_chunk_t) != 0;
        const bool ring_iter_needs_global_n_mask = global_n_is_within_ring_iter && global_n_needs_masking;

        // LOCAL N MASK
        const bool local_n_needs_masking = kv_local_padded_Nt % Sk_chunk_t != 0;
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
            const bool is_last_ring_iter = is_last_active_ring_iter(active_ring_iter_mask, ring_iter);
            const bool single_q_chunk = (global_q_end - global_q_start == 1);
            constexpr uint32_t sum_offset = q_local_padded_Nt + Lt;
            constexpr uint32_t out_num_tiles = Sq_chunk_t * vDHt;

            const uint32_t q_per_core = global_q_end - global_q_start;
            const uint32_t last_q_index = q_per_core - 1;
            const bool flush_before_prefetch = single_valid_kv_chunk || q_per_core == 2;

            // TRID assignment by Q position: Q[0] -> TRID_FIRST, Q[N-1] -> TRID_LAST,
            // Q[1..N-2] -> TRID_INNER. Used both for tagging the current Q's save and for
            // selecting which TRID to barrier on for the next Q's prefetch.
            auto trid_for_q = [&](uint32_t qi) {
                return qi == 0 ? TRID_FIRST : qi == last_q_index ? TRID_LAST : TRID_INNER;
            };

            // Issue NOC reads to fill staging for Q[pf_q_index] of the current ring_iter (or
            // ring_iter+1 for cross-ring at q==last_q_index, when the caller passes 0). Optionally
            // barriers on pf_trid first to ensure the prior save with that TRID has landed.
            auto prefetch_for = [&](uint32_t pf_q_index, uint32_t pf_trid, bool barrier_first) {
                if (barrier_first) {
                    noc.async_write_barrier<NocOptions::TXN_ID>({.trid = pf_trid});
                }
                const uint32_t gq = remap_q_index(global_q_start + pf_q_index, num_q_chunks, use_zigzag_balancing);
                const uint32_t nb_pf = gq / (NH * num_q_chunks);
                const uint32_t nq_pf = (gq % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t qc_pf = gq % num_q_chunks;
                const auto qi_pf = get_q_chunk_info<has_joint_q>(
                    qc_pf, nb_pf, nq_pf, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, q_local_padded_Nt);
                const auto& gen_pf = [&]() -> const auto& {
                    if constexpr (has_joint_q) {
                        if (qi_pf.is_joint_q) {
                            return joint_out_generator;
                        }
                    }
                    return out_generator;
                }();
                issue_restore_reads(
                    noc,
                    gen_pf,
                    stats_writer,
                    stats_tile_logical,
                    nb_pf,
                    nq_pf,
                    Sq_chunk_t,
                    qi_pf.out_slice,
                    qi_pf.stats_seq_start_tile,
                    qi_pf.stats_seq_end_tile,
                    sum_offset,
                    cb_prev_out,
                    cb_max_in,
                    cb_sum_in,
                    tile_bytes,
                    stats_tile_bytes);
            };

            // Intra-ring prefetch: bounds-check next_q_index, then dispatch with the per-TRID
            // barrier rule. Only barrier when next Q's TRID hasn't been cleared yet this ring
            // iter: Q[0] -> wB(TRID_INNER), Q[N-2] -> wB(TRID_LAST), Q[1..N-3] -> skip
            // (TRID_INNER already cleared at Q[0]).
            auto prefetch_intra_ring = [&](uint32_t next_q_index) {
                if (next_q_index >= q_per_core) {
                    return;
                }
                const uint32_t next_trid = trid_for_q(next_q_index);
                const bool need_barrier = (next_trid != TRID_INNER || next_q_index == 1);
                prefetch_for(next_q_index, next_trid, need_barrier);
            };

            // Drain pending deferred save (raw accumulators -> DRAM) for the prior Q. Called at
            // the early-flush site (before prefetch for single-valid-K iters or q_per_core==2)
            // and the late-flush site (after prefetch in the K-loop window).
            auto flush_deferred_save = [&]() {
                constexpr uint32_t all_tiles_valid = 0xFFFFFFFF;
                const auto& gen = [&]() -> const auto& {
                    if constexpr (has_joint_q) {
                        if (deferred.qi.is_joint_q) {
                            return joint_out_generator;
                        }
                    }
                    return out_generator;
                }();
                save_accumulators_with_trid<output_has_no_padding>(
                    noc,
                    gen,
                    stats_writer,
                    stats_tile_logical,
                    deferred.nb,
                    deferred.nq,
                    Sq_chunk_t,
                    deferred.qi.out_slice,
                    all_tiles_valid,
                    deferred.qi.stats_seq_start_tile,
                    deferred.qi.stats_seq_end_tile,
                    sum_offset,
                    cb_out,
                    cb_max_out,
                    cb_sum_out,
                    tile_bytes,
                    stats_tile_bytes,
                    out_subblock_h,
                    deferred.trid);
                deferred.pending = false;
            };

            for (uint32_t q_index = 0; q_index + global_q_start < global_q_end; ++q_index) {
                uint32_t global_q_chunk = remap_q_index(global_q_start + q_index, num_q_chunks, use_zigzag_balancing);

                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const bool balanced_skip_q = q_chunk < half_sequence && is_balanced && ring_index < ring_id;

                const auto qi = get_q_chunk_info<has_joint_q>(
                    q_chunk, nb, nq, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, q_local_padded_Nt);
                const uint32_t end_seq_tile = get_end_seq_tile<has_joint_q>(qi, ring_id, Lt, q_local_padded_Nt);

                // 1. Complete restore for all Q chunks to keep the prefetch pipeline in sync.
                // For balanced-skip non-last-ring-iter Q chunks, barrier without pushing —
                // compute skips these Q chunks entirely and doesn't need staging data.
                // First active iter has no prior save (matches compute's is_first_kv_for_this_q).
                if (!single_q_chunk && !is_first_active_iter) {
                    if (balanced_skip_q && !is_last_ring_iter) {
                        noc.async_read_barrier();
                    } else {
                        complete_restore(noc, cb_prev_out, out_num_tiles, cb_max_in, cb_sum_in, Sq_chunk_t);
                    }
                }

                // 2. Early flush: drain staging before prefetch when needed.
                // - single valid K chunk: compute saves to staging on K0,
                //   reserving staging CBs immediately — deadlock if they're still full.
                // - q_per_core == 2: next Q == last Q whose deferred data isn't in DRAM yet,
                //   so prefetch would read stale data without flushing first.
                // With >= 2 valid K chunks and q_per_core >= 3, K0 uses ping-pong accumulators
                // (not staging CBs), so we prefetch first and flush later during the K-loop
                // window — spreading DRAM writes to reduce bank contention.
                if (deferred.pending && flush_before_prefetch) {
                    flush_deferred_save();
                }

                // 3. Prefetch next Q chunk's accumulators from DRAM.
                // Skip the intra-ring prefetch when this Q is on the normalize-only path
                // (balanced_skip_q + is_last_ring_iter): normalize produces cb_out incrementally
                // and blocks on cb_out space; cb_out can't drain until the writer reaches
                // write_out below. Reserving space in cb_prev_out here would block until
                // normalize finishes, creating a cycle with cb_out. Deferred prefetch below
                // runs after write_out to break the cycle.
                const bool defer_prefetch = balanced_skip_q && is_last_ring_iter;
                if (!single_q_chunk && !is_first_active_iter && !defer_prefetch) {
                    prefetch_intra_ring(q_index + 1);
                }
                // Cross-ring: Q[N-1] -> Q[0] of next ring iter.
                if (!single_q_chunk && !is_last_ring_iter && q_index == last_q_index) {
                    prefetch_for(/*pf_q_index=*/0, TRID_FIRST, /*barrier_first=*/true);
                }

                // 4. Late flush (>= 2 valid K chunks, q_per_core >= 3): drain during K-loop
                // window after prefetch, spreading DRAM writes to reduce bank contention.
                if (deferred.pending) {
                    flush_deferred_save();
                }

                // Balanced causal skip: on non-last ring iters, compute pops staging and
                // doesn't push the K-loop signal. Writer skips signal wait + save + write.
                // On the last ring iter, compute runs normalize-only and pushes the signal;
                // fall through to signal wait + write (no save — no ping-pong state to save).
                if (balanced_skip_q && !is_last_ring_iter) {
                    continue;
                }

                // === Compute runs K-loop (or normalize-only on last iter) ===

                // Wait for compute to signal last K-chunk start (multi-Q only).
                // Normalize-only path also pushes this signal.
                if (!single_q_chunk) {
                    CircularBuffer cb_sig(cb_signal);
                    cb_sig.wait_front(1);
                    cb_sig.pop_front(1);
                }

                if (is_last_ring_iter) {
                    // Last-iter writes carry default trid (caller never set a non-zero trid here);
                    // pass 0 so the per-group flush waits exactly for these writes.
                    const auto& gen = [&]() -> const auto& {
                        if constexpr (has_joint_q) {
                            if (qi.is_joint_q) {
                                return joint_out_generator;
                            }
                        }
                        return out_generator;
                    }();
                    write_block_row_grouped_trid<output_has_no_padding>(
                        noc, gen, qi.out_slice, end_seq_tile, cb_out, tile_bytes, out_subblock_h, /*flush_trid=*/0);
                } else if (!single_q_chunk) {
                    deferred.pending = true;
                    deferred.trid = trid_for_q(q_index);
                    deferred.nb = nb;
                    deferred.nq = nq;
                    deferred.qi = qi;
                }

                // Delayed intra-ring prefetch for normalize-only Qs: skipped earlier to avoid
                // cycling cb_prev_out <-> cb_out with compute's normalize. Now cb_out has been
                // drained by write_out above, and compute's normalize has fully freed cb_prev_out.
                if (defer_prefetch && !single_q_chunk) {
                    prefetch_intra_ring(q_index + 1);
                }
            }
            // Hoisted DRAM-arrival barrier: on the last ring iter, write_block_row_grouped_trid
            // issued N untagged NOC writes (one per Q on this core). Wait once at the end of the
            // Q loop for all of them to land in DRAM, before the outer ring-iter loop advances
            // or the op teardown runs. Previously this was a per-Q barrier inside the loop.
            if (is_last_ring_iter) {
                noc.async_write_barrier();
            }
        } else {
            for (uint32_t q_iter = 0; q_iter + global_q_start < global_q_end; ++q_iter) {
                uint32_t global_q_chunk = remap_q_index(global_q_start + q_iter, num_q_chunks, use_zigzag_balancing);

                // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi = get_q_chunk_info<has_joint_q>(
                    q_chunk, nb, nq, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, q_local_padded_Nt);
                const uint32_t end_seq_tile = get_end_seq_tile<has_joint_q>(qi, ring_id, Lt, q_local_padded_Nt);

                // Only truly causal case appear in the iteration with local KV
                // Other iterations will just skip the computation with subsequent KV chunks
                bool causality = (ring_iter == 0 ? is_causal : false);

                if (q_chunk < half_sequence && is_balanced && ring_index < ring_id) {
                    continue;
                }

                const auto& gen = [&]() -> const auto& {
                    if constexpr (has_joint_q) {
                        if (qi.is_joint_q) {
                            return joint_out_generator;
                        }
                    }
                    return out_generator;
                }();

                // If not on the first iteration, read LSE and previous output chunk.
                // No race condition because writer kernel writes previous output before reading it again
                if (ring_iter > 0) {
                    read_prev_output_and_lse(
                        noc,
                        gen,
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
                    noc,
                    gen,
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
            noc.async_write_barrier();  // Ensure writes of output and LSE complete before next iteration
        }
    }
}
