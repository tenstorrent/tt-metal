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
#include "chunked_prefill_utils.hpp"
#include "ring_joint_kv_pad_derivation.hpp"
#include "metadata_scalar_read.hpp"
#include "fused_op_receiver.hpp"
#include "ring_utils.hpp"

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

// Maps a Q slot index to the flat chunk id this core processes there. Under the rotated split the
// mapping comes from this ring iteration's runtime-arg list; otherwise the slot index is itself the
// offset into the static flat range. `ordinal` is an ACTIVE ordinal, so ordinal + 1 is the next
// EXECUTED iteration -- what every cross-iteration caller here means.
template <bool Rotated>
struct QSlotMap {
    uint32_t args_base;
    uint32_t iter_stride;
    uint32_t ordinal;
    uint32_t flat_start;

    uint32_t flat_at_ordinal(uint32_t ord, uint32_t slot) const {
        return get_arg_val<uint32_t>(
            rotated_iter_base(args_base, iter_stride, ord) + kRotatedWriterIterHeaderWords + slot);
    }

    // Flat chunk id at `slot` of the current iteration.
    uint32_t at(uint32_t slot) const {
        if constexpr (Rotated) {
            return flat_at_ordinal(ordinal, slot);
        } else {
            return flat_start + slot;
        }
    }

    // Flat chunk id processed first in the next executed iteration. Under rotation that is a base
    // chunk this core owned all along (floats sit last), so it needs no handoff wait.
    uint32_t next_iter_first() const {
        if constexpr (Rotated) {
            return flat_at_ordinal(ordinal + 1, 0);
        } else {
            return flat_start;
        }
    }

    // Does the prefetch issued at `slot` target chunk `flat_q`? That is the next slot of this
    // iteration, or slot 0 of the next executed one when `slot` is the last.
    bool prefetch_targets(uint32_t slot, uint32_t flat_q, uint32_t q_per_core, bool is_last_ring_iter) const {
        if (slot + 1 < q_per_core) {
            return at(slot + 1) == flat_q;
        }
        return !is_last_ring_iter && next_iter_first() == flat_q;
    }
};

// Handoff semaphore slot for active ordinal `ordinal`. Ordinal >= 1 at every call site: the first
// active iteration neither receives a float nor has a deferred save to flush. The modulo keeps an
// underflow in range rather than faulting, so donor and receiver would silently disagree and hang;
// ASSERT is watcher-only, so this documents the host-schedule invariant rather than enforcing it.
inline uint32_t rotated_sem_slot(uint32_t ordinal, uint32_t sem_count) {
    ASSERT(ordinal >= 1);
    return (ordinal - 1) % sem_count;
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
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(34);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    // Slot 35: trace-safe KV-pad derivation. When set, the writer reads kv_actual_isl from the
    // kv_actual_isl tensor[0] (common runtime arg 0 = its DRAM addr) and recomputes logical_nt + ring
    // masks on-device (it's a dataflow kernel, can NoC-read), so a captured trace replays across chunks.
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(35) == 1;
    // Slot 36: sharded-joint flag (appended after upstream's kv_pad_from_metadata). When true, one L/P
    // shard arrives per ring iteration and do_joint_kv fires on every iteration rather than only the
    // last active iteration.
    constexpr bool joint_is_sharded = get_compile_time_arg_val(36) == 1;
    // Slot 37: true (unpadded) joint length in tiles (twins spatial logical_nt). Drives the joint
    // mask-generation gate together with joint_l_partial_col.
    constexpr uint32_t logical_lt = get_compile_time_arg_val(37);
    constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(38) == 1;
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(39));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(40);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(41);
    // Diagonal-mask tile slot is shared by the kernel's is_causal path and the chunked-prefill
    // path. The program factory masks kernel_is_causal off when chunked is on, so only one of
    // the two paths drives the stamp per program — but they share the CB slot layout.
    constexpr bool diag_tile_enabled = ((is_causal == 1) || chunked_enabled) && !has_sliding_window;

    // Joint-path compile-time gating. When zero, joint Q/K branches are statically dead
    // and dropped by the compiler, eliminating runtime ternaries and the joint_out_generator.
    constexpr bool has_joint_q = num_joint_q_chunks > 0;
    constexpr bool has_joint_k = num_joint_k_chunks > 0;
    // Sharded joint: num_joint_k_chunks is per-shard count; process on every ring iteration.
    constexpr bool has_gathered_joint_k = joint_is_sharded && has_joint_k;
    // Effective joint length for masking: per-shard (L_local = L/ring_size) for sharded, full L for replicated.
    constexpr uint32_t L_effective = has_gathered_joint_k ? L / ring_size : L;

    // Slots 38-41 are the rank-mapping descriptor; output accessors start at slot 42.
    constexpr auto out_args = TensorAccessorArgs<42>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();
    // Metadata accessor (metadata path only) follows the output accessors and precedes the CB compile
    // args; gate the offset on kv_pad_from_metadata so the no-metadata program never names a non-accessor
    // compile arg (fall back to a valid unused accessor offset = out_args' slot 42).
    constexpr uint32_t meta_args_offset = kv_pad_from_metadata ? stats_args.next_compile_time_args_offset() : 42;
    constexpr auto meta_args = TensorAccessorArgs<meta_args_offset>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    uint32_t single_valid_kv_chunk_mask = get_arg_val<uint32_t>(argidx++);
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

    // The stats CB is aliased by role: cb_max_* for deferred norm, cb_lse_* for eager norm.
    // Declared here rather than with the other CB args below because the rotated-Q-split sizes are
    // compile-time args off this offset, and the runtime-arg block underneath needs them.
    constexpr uint32_t cb_arg_offset =
        kv_pad_from_metadata ? meta_args.next_compile_time_args_offset() : stats_args.next_compile_time_args_offset();

    // Per-iteration chunk-list length (base chunks plus one float slot), or 0 when the host
    // declined the rotation. The offset clears the whole CB block, not just the CBs used here.
    // The factory pushes rotated_max_slots as the final compile-time arg of every kernel, so
    // read it from there rather than tracking a per-kernel index into the block above.
    constexpr uint32_t rotated_max_slots = get_ct_arg<kernel_compile_time_args.size() - 1>();
    constexpr bool rotated_q_split_enabled = rotated_max_slots > 0;
    constexpr uint32_t rotated_iter_stride = ::rotated_iter_stride(kRotatedWriterIterHeaderWords, rotated_max_slots);
    constexpr uint32_t rotated_sem_count = rotated_handoff_sem_count(ring_size);

    // The factory appends the handoff semaphore ids, then per ring iteration
    // [my_count, float_migrated_in, float_dest, chunk ids x rotated_max_slots]. The donor signals
    // the receiver's semaphore once its accumulator save lands in DRAM; the receiver waits before
    // issuing that float's restore reads.
    uint32_t rotated_sem_ids[rotated_sem_count] = {};
    uint32_t rotated_args_base = 0;
    if constexpr (rotated_q_split_enabled) {
        for (uint32_t sem_slot = 0; sem_slot < rotated_sem_count; ++sem_slot) {
            rotated_sem_ids[sem_slot] = get_arg_val<uint32_t>(argidx++);
        }
        rotated_args_base = argidx;
    }
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

    if constexpr (kv_pad_from_metadata) {
        CircularBuffer cb_meta_scratch(cb_out);
        uint32_t kv_actual_isl = trace_metadata::read_metadata_scalar_u32(
            noc, meta_args, get_common_arg_val<uint32_t>(0), cb_meta_scratch.get_write_ptr());
        kv_actual_isl =
            trace_metadata::bounded_kv_actual_isl(kv_actual_isl, chunk_size_t, kv_local_padded_Nt * ring_size);
        logical_nt = trace_metadata::logical_tile_rows_clamped_to_cache(
            kv_actual_isl, chunk_size_t, kv_local_padded_Nt * ring_size);
        const auto masks = ring_joint::build_ring_work_masks_device<full_mesh_rank_mapping>(
            fused_op_receiver.seq.ring_index,
            ring_size,
            fused_op_receiver.seq.expected[0],
            fused_op_receiver.seq.expected[1],
            num_local_k_chunks,
            Sk_chunk_t,
            kv_local_padded_Nt,
            chunked_enabled,
            chunk_size_t,
            q_local_padded_Nt,
            logical_nt,
            num_joint_k_chunks,
            L,
            true,
            is_causal != 0,
            is_balanced != 0,
            mesh_rows,
            mesh_cols,
            snake_orientation);
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
    // Joint mask generation mirrors spatial's TWO independent flags (like local_n AND global_n).
    //   (local_n analogue) Lt % Sk_chunk_t != 0: the K-chunk is wider than the per-device joint shard
    //     (writer slot 12 Lt is already per-device Lt_local), so every fully-real shard carries
    //     fully-padded trailing tiles (e.g. wadada Lt=2, Sk=16 -> 14 pad tiles per shard).
    //   (global_n analogue) logical_lt % Sk_chunk_t != 0 || joint_l_partial_col != 0: real tokens do
    //     not fill the last real shard's chunk — fully-padded trailing tiles and/or a sub-tile column.
    constexpr bool joint_has_padding =
        L > 0 && ((Lt % Sk_chunk_t != 0) || (logical_lt % Sk_chunk_t != 0) || (joint_l_partial_col != 0));
    constexpr bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) || diag_tile_enabled || has_sliding_window;
    if constexpr (needs_lightweight_mask) {
        generate_lightweight_mask_tiles<
            global_n_partial_col,
            joint_l_partial_col,
            cb_mask_in,
            (is_causal == 1) || chunked_enabled,
            sliding_window_size>(noc);
    }

    const uint32_t ring_index =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            fused_op_receiver.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    uint32_t half_sequence = num_q_chunks / 2;

    // Deferred save: stash params for save_accumulators_with_trid and call it
    // during the next Q chunk's K-loop window to avoid DRAM bank contention.
    struct DeferredWriteContext {
        bool pending = false;
        uint32_t trid = 0;
        uint32_t nb = 0;
        uint32_t nq = 0;
        QChunkInfo qi = {};
        // Rotated split only. Packed physical core owning this chunk next iteration (kRotatedNoDest if
        // it stays put), and the flat chunk id this save belongs to, so the early-flush decision
        // below can compare chunk identity instead of slot position (slots do not map to fixed
        // chunks under rotation).
        uint32_t mig_dest = kRotatedNoDest;
        uint32_t flat_q = 0;
    } deferred = {};

    // Track non-skipped iters so the first active iter starts with fresh accumulators (matches compute).
    bool seen_active_iter = false;
    constexpr uint32_t sdpa_ring_iterations = has_sliding_window ? 1 : ring_size;
    for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
        // Sliding compute consumes all local/halo source ranges in one logical pass, so the
        // writer sees exactly one final output per Q and never enters deferred staging.
        const uint32_t ring_id =
            has_sliding_window
                ? ring_index
                : ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                      fused_op_receiver.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
        // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
        // still advances above so writer stays aligned with reader, compute, and all-gather.
        if (!has_sliding_window && ((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
            continue;
        }
        // Sharded joint: one L/P shard per ring iteration — process joint K/V on every iteration.
        // Replicated joint: process joint when ring_id == ring_size-1.
        const bool do_joint_kv = has_gathered_joint_k ? true : (ring_id == ring_size - 1);
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

        // JOINT L MASK — uses L_effective (per-shard for sharded, full L for replicated).
        constexpr bool joint_n_needs_masking = L_effective % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;

        // Deferred normalization is always paired with streaming compute.
        constexpr bool use_deferred_norm = use_streaming_compute;

        if constexpr (has_sliding_window) {
            // Sliding compute consumes every local/halo source for a Q in one pass. There is
            // therefore no intermediate state to restore or save: wait for the final result,
            // write it once, and advance directly to the next assigned Q.
            const bool single_q_chunk = (global_q_end - global_q_start == 1);
            for (uint32_t q_index = 0; q_index + global_q_start < global_q_end; ++q_index) {
                const auto decoded_q =
                    decompose_global_q_index(global_q_start + q_index, num_q_chunks, NH, use_zigzag_balancing);
                const uint32_t nb = decoded_q.nb;
                const uint32_t nq = decoded_q.nq;
                const uint32_t q_chunk = decoded_q.q_chunk;
                const auto qi = get_q_chunk_info<has_joint_q>(
                    q_chunk, nb, nq, num_local_q_chunks, Sq_chunk_t, vDHt, Lt, q_local_padded_Nt);
                const uint32_t end_seq_tile = get_end_seq_tile<has_joint_q>(qi, ring_id, Lt, q_local_padded_Nt);

                if (!single_q_chunk) {
                    CircularBuffer cb_sig(cb_signal);
                    cb_sig.wait_front(1);
                    cb_sig.pop_front(1);
                }

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
            }
            noc.async_write_barrier();
        } else if constexpr (use_deferred_norm) {
            // Deferred norm: accumulates across ring iterations with exponential rescaling.
            // Single Q-chunk: accumulators persist in L1, write final output on last ring_iter.
            // Multi Q-chunk: raw accumulators round-trip through DRAM between ring iterations.
            const bool is_last_ring_iter = is_last_active_ring_iter(active_ring_iter_mask, ring_iter);
            // Rotated: rotated_max_slots >= 2, so accumulators always round-trip through DRAM.
            // Deriving it from the static range would disagree with compute's per-iteration count.
            const bool single_q_chunk = !rotated_q_split_enabled && (global_q_end - global_q_start == 1);
            constexpr uint32_t sum_offset = q_local_padded_Nt + Lt;
            constexpr uint32_t out_num_tiles = Sq_chunk_t * vDHt;

            // Rotated: q_per_core is the chunks owned THIS iteration (base, or base+1 with the
            // float), shadowing the static meaning of global_q_start/_end.
            uint32_t rotated_ordinal = 0;
            uint32_t q_per_core;
            uint32_t rotated_has_mig_in_float = 0;
            uint32_t rotated_float_dest = kRotatedNoDest;
            if constexpr (rotated_q_split_enabled) {
                rotated_ordinal = rotated_active_ordinal(active_ring_iter_mask, ring_iter);
                const uint32_t rotated_iter_base =
                    ::rotated_iter_base(rotated_args_base, rotated_iter_stride, rotated_ordinal);
                q_per_core = get_arg_val<uint32_t>(rotated_iter_base);
                rotated_has_mig_in_float = get_arg_val<uint32_t>(rotated_iter_base + 1);
                rotated_float_dest = get_arg_val<uint32_t>(rotated_iter_base + 2);
            } else {
                q_per_core = global_q_end - global_q_start;
            }

            const QSlotMap<rotated_q_split_enabled> q_slots{
                rotated_args_base, rotated_iter_stride, rotated_ordinal, global_q_start};

            const uint32_t last_q_index = q_per_core - 1;
            const bool flush_before_prefetch = single_valid_kv_chunk || q_per_core == 2;

            // TRID by Q position: Q[0] -> TRID_FIRST, Q[N-1] -> TRID_LAST, else TRID_INNER. Tags
            // the current Q's save and selects which TRID to barrier before the next Q's restore.
            // Under rotation last_q_index varies per iteration, so slot -> TRID is not stable across
            // the save/restore boundary. Do not "fix" that in isolation: keying off rotated_max_slots
            // makes the map stable but breaks accuracy, because need_barrier below assumes the last
            // slot maps to TRID_LAST and skips its barrier when nothing does.
            auto trid_for_q = [&](uint32_t qi) {
                return qi == 0 ? TRID_FIRST : qi == last_q_index ? TRID_LAST : TRID_INNER;
            };

            // Issue NOC reads to fill staging for Q[pf_q_index] of the current ring_iter (or
            // ring_iter+1 for cross-ring at q==last_q_index, when the caller passes 0). Optionally
            // barriers on pf_trid first to ensure the prior save with that TRID has landed.
            auto prefetch_for = [&](uint32_t pf_flat_q, uint32_t pf_trid, bool barrier_first) {
                if (barrier_first) {
                    noc.async_write_barrier<NocOptions::TXN_ID>({.trid = pf_trid});
                }
                const auto decoded_q = decompose_global_q_index(pf_flat_q, num_q_chunks, NH, use_zigzag_balancing);
                const uint32_t nb_pf = decoded_q.nb;
                const uint32_t nq_pf = decoded_q.nq;
                const uint32_t qc_pf = decoded_q.q_chunk;
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
            // Coupled to trid_for_q above: assumes the last slot maps to TRID_LAST.
            auto prefetch_intra_ring = [&](uint32_t next_q_index) {
                if (next_q_index >= q_per_core) {
                    return;
                }
                // The migrated-in float sits last. Wait for the donor's handoff signal before the
                // restore reads, then reset the semaphore for the next (possibly cached) run.
                if constexpr (rotated_q_split_enabled) {
                    if (rotated_has_mig_in_float != 0 && next_q_index == last_q_index) {
                        // Never set on the first active iteration, so ring_iter >= 1 here.
                        Semaphore<> handoff_sem(rotated_sem_ids[rotated_sem_slot(rotated_ordinal, rotated_sem_count)]);
                        handoff_sem.wait_min(rotated_has_mig_in_float);
                        handoff_sem.set(0);
                    }
                }
                const uint32_t next_trid = trid_for_q(next_q_index);
                const bool need_barrier = (next_trid != TRID_INNER || next_q_index == 1);
                prefetch_for(q_slots.at(next_q_index), next_trid, need_barrier);
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
                // Donor half of the float handoff. Floats sit last, so this flush runs during the
                // receiver's use iteration and both index rotated_sem_slot() with the same ordinal.
                // A core can be donor and receiver in one iteration; that is not a cycle, because
                // the donor's flush at slot 0 always precedes the receiver's wait at last-1 >= 1.
                // Barrier first: flushed-but-not-landed writes must not read as "ready".
                if constexpr (rotated_q_split_enabled) {
                    if (deferred.mig_dest != kRotatedNoDest) {
                        noc.async_write_barrier<NocOptions::TXN_ID>({.trid = deferred.trid});
                        Semaphore<>(rotated_sem_ids[rotated_sem_slot(rotated_ordinal, rotated_sem_count)])
                            .up(noc, rotated_dest_x(deferred.mig_dest), rotated_dest_y(deferred.mig_dest), 1);
                        deferred.mig_dest = kRotatedNoDest;
                    }
                }
                deferred.pending = false;
            };

            for (uint32_t q_index = 0; q_index < q_per_core; ++q_index) {
                const auto decoded_q =
                    decompose_global_q_index(q_slots.at(q_index), num_q_chunks, NH, use_zigzag_balancing);
                const uint32_t nb = decoded_q.nb;
                const uint32_t nq = decoded_q.nq;
                const uint32_t q_chunk = decoded_q.q_chunk;

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
                //
                // q_per_core == 2 is a positional proxy for "the chunk about to be prefetched is
                // the one still in staging", exact only while slots map to fixed chunks. Under
                // rotation they do not, so compare chunk ids directly instead.
                const bool early_flush =
                    rotated_q_split_enabled
                        ? (single_valid_kv_chunk ||
                           q_slots.prefetch_targets(q_index, deferred.flat_q, q_per_core, is_last_ring_iter))
                        : flush_before_prefetch;
                if (deferred.pending && early_flush) {
                    flush_deferred_save();
                }

                // 3. Prefetch next Q chunk's accumulators from DRAM.
                // Skip the intra-ring prefetch when this Q is on the normalize-only path
                // (balanced_skip_q + is_last_ring_iter): normalize produces cb_out incrementally
                // and blocks on cb_out space; cb_out can't drain until the writer reaches
                // write_out below. Reserving space in cb_prev_out here would block until
                // normalize finishes, creating a cycle with cb_out. Deferred prefetch below
                // runs after write_out to break the cycle.
                // At base == 1 the cross-ring prefetch target (slot 0 of t+1) is the chunk being
                // processed now, whose save only exists at the end of this body -- prefetching here
                // would restore the previous iteration's accumulators, so postpone it.
                const bool rotated_postpone_cross_prefetch = rotated_q_split_enabled && !is_last_ring_iter &&
                                                             q_index == last_q_index &&
                                                             q_slots.next_iter_first() == q_slots.at(q_index);
                const bool defer_prefetch = balanced_skip_q && is_last_ring_iter;
                if (!single_q_chunk && !is_first_active_iter && !defer_prefetch) {
                    prefetch_intra_ring(q_index + 1);
                }
                // Cross-ring: Q[N-1] -> Q[0] of next ring iter.
                if (!single_q_chunk && !is_last_ring_iter && q_index == last_q_index &&
                    !rotated_postpone_cross_prefetch) {
                    prefetch_for(q_slots.next_iter_first(), TRID_FIRST, /*barrier_first=*/true);
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
                    if constexpr (rotated_q_split_enabled) {
                        // Only the last (float) slot can migrate; base chunks keep kRotatedNoDest.
                        deferred.mig_dest = (q_index == last_q_index) ? rotated_float_dest : kRotatedNoDest;
                        deferred.flat_q = q_slots.at(q_index);
                    }
                }

                // Postponed cross-ring prefetch: the save now exists, so issue it and read it back.
                // prefetch_for barriers deferred.trid first, making the save visible to the read.
                if (rotated_postpone_cross_prefetch) {
                    const uint32_t save_trid = deferred.trid;
                    flush_deferred_save();
                    prefetch_for(q_slots.next_iter_first(), save_trid, /*barrier_first=*/true);
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
                const auto decoded_q =
                    decompose_global_q_index(global_q_start + q_iter, num_q_chunks, NH, use_zigzag_balancing);
                const uint32_t nb = decoded_q.nb;
                const uint32_t nq = decoded_q.nq;
                const uint32_t q_chunk = decoded_q.q_chunk;

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
