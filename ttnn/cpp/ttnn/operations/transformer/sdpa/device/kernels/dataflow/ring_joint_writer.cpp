// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"
#include "fused_op_receiver.hpp"

template <typename ReaderType, typename TensorAccessorType>
void read_prev_output_and_lse(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& lse_writer,
    const TensorTileShape& lse_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t lse_seq_start_tile,
    const uint32_t lse_seq_end_tile,
    const uint32_t cb_prev_out,
    const uint32_t cb_lse_in,
    const uint32_t tile_bytes,
    const uint32_t lse_tile_bytes) {
    // Read previous output for this Q chunk
    read_block(cat_out_generator, out_slice, end_seq_tile, cb_prev_out, tile_bytes, false);

    // Read previous LSE for this Q chunk
    cb_reserve_back(cb_lse_in, Sq_chunk_t);
    uint32_t lse_addr = get_write_ptr(cb_lse_in);
    for (uint32_t i = lse_seq_start_tile; i < lse_seq_end_tile; i++) {
        noc_async_read_tile(lse_tile_logical.id_of(nb, nq, i, 0), lse_writer, lse_addr);
        lse_addr += lse_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_lse_in, Sq_chunk_t);
}

template <typename ReaderType, typename TensorAccessorType>
void read_prev_accumulators(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& lse_writer,
    const TensorTileShape& lse_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t lse_seq_start_tile,
    const uint32_t lse_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_prev_out,
    const uint32_t cb_lse_in,
    const uint32_t cb_sum_in,
    const uint32_t tile_bytes,
    const uint32_t lse_tile_bytes) {
    // Read previous output
    read_block(cat_out_generator, out_slice, end_seq_tile, cb_prev_out, tile_bytes, false);

    // Read max from LSE DRAM (first half)
    cb_reserve_back(cb_lse_in, Sq_chunk_t);
    uint32_t max_addr = get_write_ptr(cb_lse_in);
    for (uint32_t i = lse_seq_start_tile; i < lse_seq_end_tile; i++) {
        noc_async_read_tile(lse_tile_logical.id_of(nb, nq, i, 0), lse_writer, max_addr);
        max_addr += lse_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_lse_in, Sq_chunk_t);

    // Read sum from LSE DRAM (second half, offset by sum_offset)
    cb_reserve_back(cb_sum_in, Sq_chunk_t);
    uint32_t sum_addr = get_write_ptr(cb_sum_in);
    for (uint32_t i = lse_seq_start_tile; i < lse_seq_end_tile; i++) {
        noc_async_read_tile(lse_tile_logical.id_of(nb, nq, sum_offset + i, 0), lse_writer, sum_addr);
        sum_addr += lse_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_sum_in, Sq_chunk_t);
}

/**
 * Write max and sum statistics to DRAM (deferred norm path).
 * Max occupies the first half of the stats tensor, sum the second half (at sum_offset).
 * Waits for compute to push tiles, writes them to DRAM, and pops the CBs.
 */
template <typename TensorAccessorType>
void write_max_and_sum(
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_max_out,
    const uint32_t cb_sum_out,
    const uint32_t stats_tile_bytes) {
    // Write max to stats DRAM (first half)
    cb_wait_front(cb_max_out, Sq_chunk_t);
    uint32_t max_write_addr = get_read_ptr(cb_max_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, max_write_addr);
        max_write_addr += stats_tile_bytes;
    }
    cb_pop_front(cb_max_out, Sq_chunk_t);

    // Write sum to stats DRAM (second half, offset by sum_offset)
    cb_wait_front(cb_sum_out, Sq_chunk_t);
    uint32_t sum_write_addr = get_read_ptr(cb_sum_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, sum_offset + i, 0), stats_writer, sum_write_addr);
        sum_write_addr += stats_tile_bytes;
    }
    cb_pop_front(cb_sum_out, Sq_chunk_t);
}

template <typename ReaderType, typename TensorAccessorType>
void write_accumulators(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& lse_writer,
    const TensorTileShape& lse_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t lse_seq_start_tile,
    const uint32_t lse_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_out,
    const uint32_t cb_lse_out,
    const uint32_t cb_sum_out,
    const uint32_t tile_bytes,
    const uint32_t lse_tile_bytes) {
    write_block(cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes);
    write_max_and_sum(
        lse_writer,
        lse_tile_logical,
        nb,
        nq,
        Sq_chunk_t,
        lse_seq_start_tile,
        lse_seq_end_tile,
        sum_offset,
        cb_lse_out,
        cb_sum_out,
        lse_tile_bytes);
}

/**
 * Write output tiles to DRAM in row groups of subblock_h rows.
 * Each group: wait for tiles from compute, write to DRAM, barrier, pop.
 * Frees cb_out space incrementally so compute can start saving the next Q chunk sooner.
 */
template <typename ReaderType>
void write_output_row_by_row(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t cb_out,
    const uint32_t tile_bytes,
    const uint32_t Sq_chunk_t,
    const uint32_t DHt,
    const uint32_t subblock_h) {
    for (uint32_t g = 0; g < Sq_chunk_t; g += subblock_h) {
        const uint32_t group_rows = std::min(subblock_h, Sq_chunk_t - g);
        const uint32_t group_tiles = group_rows * DHt;

        cb_wait_front(cb_out, group_tiles);
        uint32_t read_ptr = get_read_ptr(cb_out);
        for (uint32_t row = 0; row < group_rows; ++row) {
            for (uint32_t col = 0; col < DHt; ++col) {
                cat_out_generator.maybe_write_tile(
                    out_slice.d0,
                    out_slice.d1,
                    out_slice.d2_start + g + row,
                    out_slice.d3_start + col,
                    end_seq_tile,
                    read_ptr);
                read_ptr += tile_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, group_tiles);
    }
}

/**
 * Read previous ring iteration's accumulators (max, sum, output) from DRAM into CBs.
 * Max and sum are read all at once (small: Sq_chunk_t tiles each).
 * Output is read row-by-row (subblock_h rows per group) so compute can start
 * consuming tiles as soon as the first group arrives — overlapping DRAM reads
 * with K chunk 0's compute (which doesn't need prev_out).
 */
template <typename ReaderType, typename TensorAccessorType>
void read_prev_accumulators_prefetch(
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
    const uint32_t stats_tile_bytes,
    const uint32_t DHt,
    const uint32_t subblock_h) {
    // Read max from stats DRAM (first half) — all at once
    cb_reserve_back(cb_max_in, Sq_chunk_t);
    uint32_t max_addr = get_write_ptr(cb_max_in);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_read_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, max_addr);
        max_addr += stats_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_max_in, Sq_chunk_t);

    // Read sum from stats DRAM (second half) — all at once
    cb_reserve_back(cb_sum_in, Sq_chunk_t);
    uint32_t sum_addr = get_write_ptr(cb_sum_in);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_read_tile(stats_tile_logical.id_of(nb, nq, sum_offset + i, 0), stats_writer, sum_addr);
        sum_addr += stats_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_sum_in, Sq_chunk_t);

    // Read prev_out row-by-row
    const uint32_t src_rows = out_slice.get_d2_size();
    const uint32_t src_cols = out_slice.get_d3_size();
    for (uint32_t g = 0; g < src_rows; g += subblock_h) {
        const uint32_t group_rows = std::min(subblock_h, src_rows - g);
        const uint32_t group_tiles = group_rows * src_cols;

        cb_reserve_back(cb_prev_out, group_tiles);
        uint32_t write_ptr = get_write_ptr(cb_prev_out);
        for (uint32_t row = 0; row < group_rows; ++row) {
            for (uint32_t col = 0; col < src_cols; ++col) {
                cat_out_generator.maybe_read_tile(
                    out_slice.d0,
                    out_slice.d1,
                    out_slice.d2_start + g + row,
                    out_slice.d3_start + col,
                    end_seq_tile,
                    write_ptr);
                write_ptr += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_prev_out, group_tiles);
    }
}

template <typename ReaderType, typename TensorAccessorType>
void write_output_and_lse(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& lse_writer,
    const TensorTileShape& lse_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t lse_seq_start_tile,
    const uint32_t lse_seq_end_tile,
    const uint32_t cb_out,
    const uint32_t cb_lse_out,
    const uint32_t tile_bytes,
    const uint32_t lse_tile_bytes) {
    write_block(cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes);

    cb_wait_front(cb_lse_out, Sq_chunk_t);
    uint32_t lse_addr = get_read_ptr(cb_lse_out);
    for (uint32_t i = lse_seq_start_tile; i < lse_seq_end_tile; i++) {
        noc_async_write_tile(lse_tile_logical.id_of(nb, nq, i, 0), lse_writer, lse_addr);
        lse_addr += lse_tile_bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_lse_out, Sq_chunk_t);
}

/**
 * Per-Q-chunk DRAM addressing for output and statistics (max/sum) tensors.
 *
 * Naming conventions:
 *   "Nt" suffix = tile count (N in tiles). E.g., local_padded_Nt = local padded sequence length in tiles.
 *   "global_q_chunk" = flattened index across [B, NH, num_q_chunks] assigned to this core.
 *   "local" = per-device (this device's portion of the sequence, before ring rotation).
 *   "joint" = joint/cross-attention tokens appended after local tokens.
 *   nb, nq = batch index, head index extracted from the flattened global_q_chunk.
 */
struct QChunkAddr {
    uint32_t nb;                    // Batch index
    uint32_t nq;                    // Head index
    bool is_joint_q;                // True if this Q chunk is from the joint tensor (not local)
    Slice out_slice;                // DRAM slice for output: [nb, nq, row_start:row_end, 0:DHt]
    uint32_t end_seq_tile;          // Boundary tile for PaddedAddrGenerator (local vs joint routing)
    uint32_t stats_seq_start_tile;  // Start tile in stats DRAM tensor (max/sum) for this Q chunk
    uint32_t stats_seq_end_tile;    // End tile (exclusive) in stats DRAM tensor
};

/**
 * Compute DRAM addressing for a given Q chunk.
 *
 * @param global_q_chunk  Flattened index into [B * NH * num_q_chunks] for this core's Q work.
 *                        Decomposed into (nb, nq, q_chunk) where q_chunk is the local chunk index.
 * @param ring_id         Which device's K/V shard is currently being processed (determines
 *                        end_seq_tile for the PaddedAddrGenerator to route local vs joint output).
 *
 * Template params:
 *   NH              = number of attention heads per device
 *   num_q_chunks    = total Q chunks per (batch, head) = num_local_q_chunks + num_joint_q_chunks
 *   num_local_q_chunks = Q chunks from the local sequence
 *   Sq_chunk_t      = Q chunk size in tiles (e.g., 288/32 = 9)
 *   DHt             = head dimension in tiles (e.g., 128/32 = 4)
 *   Lt              = joint sequence length in tiles
 *   local_padded_Nt = local sequence length in tiles, padded to tile boundary
 */
template <
    uint32_t NH,
    uint32_t num_q_chunks,
    uint32_t num_local_q_chunks,
    uint32_t Sq_chunk_t,
    uint32_t DHt,
    uint32_t Lt,
    uint32_t local_padded_Nt>
QChunkAddr compute_q_chunk_addr(uint32_t global_q_chunk, uint32_t ring_id) {
    QChunkAddr a;
    a.nb = global_q_chunk / (NH * num_q_chunks);
    a.nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
    const uint32_t q_chunk = global_q_chunk % num_q_chunks;  // Local chunk index within this (batch, head)
    a.is_joint_q = q_chunk >= num_local_q_chunks;

    if (a.is_joint_q) {
        // Joint Q chunk: indexes into the joint output tensor
        const uint32_t jrow = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
        a.out_slice = Slice(a.nb, a.nq, jrow, jrow + Sq_chunk_t, 0, DHt);
        a.end_seq_tile = Lt;
        // Stats tile range: joint stats are stored after local stats in the DRAM tensor
        a.stats_seq_start_tile = local_padded_Nt + (q_chunk - num_local_q_chunks) * Sq_chunk_t;
        a.stats_seq_end_tile = a.stats_seq_start_tile + Sq_chunk_t;
        a.stats_seq_start_tile = std::min(a.stats_seq_start_tile, local_padded_Nt + Lt);
        a.stats_seq_end_tile = std::min(a.stats_seq_end_tile, local_padded_Nt + Lt);
    } else {
        // Local Q chunk: indexes into the main output tensor
        const uint32_t row = q_chunk * Sq_chunk_t;
        a.out_slice = Slice(a.nb, a.nq, row, row + Sq_chunk_t, 0, DHt);
        // end_seq_tile tells PaddedAddrGenerator where local output ends (for zero-padding beyond)
        a.end_seq_tile = local_padded_Nt * (ring_id + 1);
        // Stats tile range: local stats in the first half of the DRAM tensor
        a.stats_seq_start_tile = q_chunk * Sq_chunk_t;
        a.stats_seq_end_tile = a.stats_seq_start_tile + Sq_chunk_t;
        a.stats_seq_start_tile = std::min(a.stats_seq_start_tile, local_padded_Nt);
        a.stats_seq_end_tile = std::min(a.stats_seq_end_tile, local_padded_Nt);
    }
    return a;
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    // Naming conventions:
    //   _t suffix = in tiles (e.g., Sq_chunk_t = Q chunk size in tiles)
    //   _N suffix = in elements (e.g., local_padded_N = local padded seq len in elements)
    //   "local"   = this device's portion of the sequence
    //   "global"  = across all devices in the ring (logical_n = true total seq length)
    //   "padded"  = rounded up to tile boundary
    //   "joint"   = joint/cross-attention tokens (appended after local)
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);       // Q chunk size in tiles
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);       // K chunk size in tiles
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(5);   // Per-device seq len (elements), padded
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(6);  // Per-device seq len (tiles), padded
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(7);        // Total seq len across ring (tiles), padded
    constexpr uint32_t logical_n = get_compile_time_arg_val(8);        // True total seq len (elements) before padding
    constexpr uint32_t logical_nt = get_compile_time_arg_val(9);  // True total seq len (tiles) = ceil(logical_n/32)
    constexpr uint32_t Lt = get_compile_time_arg_val(10);         // Joint seq len (tiles)
    constexpr uint32_t L = get_compile_time_arg_val(11);          // Joint seq len (elements)
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(16);  // num_local_q_chunks + num_joint_q_chunks
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(17);
    constexpr uint32_t scale_val = get_compile_time_arg_val(18);
    constexpr uint32_t ring_size = get_compile_time_arg_val(19);  // Number of devices in ring
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(20);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(21);
    constexpr bool use_deferred_norm = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t subblock_h = get_compile_time_arg_val(23);  // Row group size for row-by-row I/O

    constexpr auto out_args = TensorAccessorArgs<24>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);        // DRAM addr: local output tensor
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);  // DRAM addr: joint output tensor
    const uint32_t lse_addr = get_arg_val<uint32_t>(argidx++);        // DRAM addr: stats tensor (max + sum)
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);  // First Q chunk index for this core
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);    // One past last Q chunk index

    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

    constexpr uint32_t cb_lse_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_sum_out = tt::CBIndex::c_10;
    constexpr uint32_t cb_sum_in = tt::CBIndex::c_11;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t lse_tile_bytes = get_tile_size(cb_lse_in);

    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr, tile_bytes);
    const auto lse_writer = TensorAccessor(lse_args, lse_addr, lse_tile_bytes);

    const auto output_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, DHt);
    // LSE tensor is 2× size: first half for max/LSE, second half for sum.
    const auto lse_tile_logical = TensorTileShape(B, NH, (local_padded_Nt + Lt) * 2, 1);

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);
    const auto lse_generator = PaddedAddrGenerator(lse_writer, lse_tile_logical);

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

        if constexpr (use_deferred_norm) {
            // Deferred norm: accumulates across ring iterations with exponential rescaling.
            // Single Q-chunk: accumulators persist in L1, write final output on last ring_iter.
            // Multi Q-chunk: raw accumulators round-trip through DRAM between ring iterations.
            const bool is_last_ring_iter = (ring_iter == ring_size - 1);
            const bool single_q_chunk = (global_q_end - global_q_start == 1);

            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                const auto a =
                    compute_q_chunk_addr<NH, num_q_chunks, num_local_q_chunks, Sq_chunk_t, DHt, Lt, local_padded_Nt>(
                        global_q_chunk, ring_id);
                constexpr uint32_t sum_offset = local_padded_Nt + Lt;
                const auto& gen = a.is_joint_q ? joint_out_generator : out_generator;

                // Read accumulators: first Q chunk reads fresh; rest were prefetched.
                if (!single_q_chunk && ring_iter > 0 && global_q_chunk == global_q_start) {
                    read_prev_accumulators_prefetch(
                        gen,
                        lse_writer,
                        lse_tile_logical,
                        a.nb,
                        a.nq,
                        Sq_chunk_t,
                        a.out_slice,
                        a.end_seq_tile,
                        a.stats_seq_start_tile,
                        a.stats_seq_end_tile,
                        sum_offset,
                        cb_prev_out,
                        cb_lse_in,
                        cb_sum_in,
                        tile_bytes,
                        lse_tile_bytes,
                        DHt,
                        subblock_h);
                }

                // Write current Q chunk's output (row-by-row to overlap with compute).
                if (is_last_ring_iter) {
                    write_output_row_by_row(
                        gen, a.out_slice, a.end_seq_tile, cb_out, tile_bytes, Sq_chunk_t, DHt, subblock_h);
                } else if (!single_q_chunk) {
                    write_output_row_by_row(
                        gen, a.out_slice, a.end_seq_tile, cb_out, tile_bytes, Sq_chunk_t, DHt, subblock_h);
                    write_max_and_sum(
                        lse_writer,
                        lse_tile_logical,
                        a.nb,
                        a.nq,
                        Sq_chunk_t,
                        a.stats_seq_start_tile,
                        a.stats_seq_end_tile,
                        sum_offset,
                        cb_lse_out,
                        cb_sum_out,
                        lse_tile_bytes);
                }

                // Prefetch next Q chunk's accumulators from DRAM.
                // Overlaps with compute's K chunk 0 (is_first=true, no prev_out needed).
                if (!single_q_chunk && ring_iter > 0 && global_q_chunk + 1 < global_q_end) {
                    const auto next = compute_q_chunk_addr<
                        NH,
                        num_q_chunks,
                        num_local_q_chunks,
                        Sq_chunk_t,
                        DHt,
                        Lt,
                        local_padded_Nt>(global_q_chunk + 1, ring_id);
                    const auto& next_gen = next.is_joint_q ? joint_out_generator : out_generator;
                    read_prev_accumulators_prefetch(
                        next_gen,
                        lse_writer,
                        lse_tile_logical,
                        next.nb,
                        next.nq,
                        Sq_chunk_t,
                        next.out_slice,
                        next.end_seq_tile,
                        next.stats_seq_start_tile,
                        next.stats_seq_end_tile,
                        sum_offset,
                        cb_prev_out,
                        cb_lse_in,
                        cb_sum_in,
                        tile_bytes,
                        lse_tile_bytes,
                        DHt,
                        subblock_h);
                }
            }
            noc_async_write_barrier();
        } else {
            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const bool is_joint_q = q_chunk >= num_local_q_chunks;
                Slice out_slice;
                uint32_t end_seq_tile;
                if (is_joint_q) {
                    const uint32_t joint_out_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                    out_slice = Slice(nb, nq, joint_out_row_start_tile, joint_out_row_start_tile + Sq_chunk_t, 0, DHt);
                    end_seq_tile = Lt;
                } else {
                    const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
                    out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
                    end_seq_tile = local_padded_Nt * (ring_id + 1);
                }

                // If not on the first iteration, read LSE input and previous output chunk.
                // No race condition because writer kernel writes previous output before reading it again

                uint32_t lse_seq_start_tile;
                uint32_t lse_seq_end_tile;
                if (is_joint_q) {
                    lse_seq_start_tile = local_padded_Nt + (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                    lse_seq_end_tile = lse_seq_start_tile + Sq_chunk_t;
                    lse_seq_start_tile = std::min(lse_seq_start_tile, local_padded_Nt + Lt);
                    lse_seq_end_tile = std::min(lse_seq_end_tile, local_padded_Nt + Lt);
                } else {
                    lse_seq_start_tile = q_chunk * Sq_chunk_t;
                    lse_seq_end_tile = lse_seq_start_tile + Sq_chunk_t;
                    lse_seq_start_tile = std::min(lse_seq_start_tile, local_padded_Nt);
                    lse_seq_end_tile = std::min(lse_seq_end_tile, local_padded_Nt);
                }

                if (ring_iter > 0) {
                    read_prev_output_and_lse(
                        is_joint_q ? joint_out_generator : out_generator,
                        lse_writer,
                        lse_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        out_slice,
                        end_seq_tile,
                        lse_seq_start_tile,
                        lse_seq_end_tile,
                        cb_prev_out,
                        cb_lse_in,
                        tile_bytes,
                        lse_tile_bytes);
                }

                write_output_and_lse(
                    is_joint_q ? joint_out_generator : out_generator,
                    lse_writer,
                    lse_tile_logical,
                    nb,
                    nq,
                    Sq_chunk_t,
                    out_slice,
                    end_seq_tile,
                    lse_seq_start_tile,
                    lse_seq_end_tile,
                    cb_out,
                    cb_lse_out,
                    tile_bytes,
                    lse_tile_bytes);
            }
            noc_async_write_barrier();  // Ensure writes of output and LSE complete before next iteration
        }
    }
}
