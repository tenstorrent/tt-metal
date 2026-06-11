// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Reader for the fused Wan2.2 distributed RMSNorm op.
 *
 * Streams the core's tile-row slice of (input, weight, rope_cos, rope_sin)
 * from DRAM into local CBs, block-by-block — matches the existing
 * rms_post_allgather_reader streaming model so compute can start consuming
 * the first chunk while the reader continues filling.
 *
 * Differences from the existing post-allgather reader:
 *   - We do NOT read stats from DRAM; stats come from the compute kernel's
 *     own pre phase via stats_local_cb, are forwarded by the AG kernel, and
 *     are delivered to compute via stats_gathered_cb.
 *   - We generate TWO reduce scalars: SUM (for pre phase) and AVG (for post).
 *   - Optional weight / RoPE / trans_mat loading is preserved verbatim.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <tt-metalium/constants.hpp>
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(1);
    constexpr uint32_t reduce_scalar_sum_cb = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_scalar_avg_cb = get_compile_time_arg_val(3);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(4);
    constexpr uint32_t transformation_mat_cb = get_compile_time_arg_val(5);
    constexpr uint32_t rope_cos_cb = get_compile_time_arg_val(6);
    constexpr uint32_t rope_sin_cb = get_compile_time_arg_val(7);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(8);
    constexpr uint32_t block_size = get_compile_time_arg_val(9);
    constexpr uint32_t reduce_factor = get_compile_time_arg_val(10);  // == stats_tiles_cols
    constexpr uint32_t epsilon_value = get_compile_time_arg_val(11);
    constexpr uint32_t has_weight = get_compile_time_arg_val(12);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(13);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t chunk_size_rows = get_compile_time_arg_val(15);
    constexpr uint32_t per_head_rope = get_compile_time_arg_val(16);
    constexpr uint32_t rope_seqlen_tiles = get_compile_time_arg_val(17);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(18);
    constexpr uint32_t has_bias = get_compile_time_arg_val(19);
    // Per-token weight/bias: shape [N, H] (vs broadcast [1, H]). Read pattern
    // is per-row (after each row's input is pushed) using noc_async_read_tile
    // for full 4 KB/tile (vs face_row_bytes for the broadcast case). Compute
    // uses mul_tiles / add_tiles directly (no _bcast_rows).
    constexpr uint32_t per_token_weight = get_compile_time_arg_val(20);
    constexpr uint32_t per_token_bias = get_compile_time_arg_val(21);
    // Streaming low-L1: input_cb is block-sized, so the row is read in two
    // passes (PRE sum-of-squares, then a POST re-read for x*(1/rms)) in
    // block_size-tile pushes that compute pops as it consumes. The resident
    // fast path reads the whole row once. See program_factory.
    constexpr uint32_t streaming_low_l1 = get_compile_time_arg_val(22);
    // When set (MUX path), the WRITER populates reduce_scalar_*/epsilon/
    // transformation_mat CBs instead of the reader, so the reader's first NoC
    // op is the input read (it starts streaming input ASAP).
    constexpr uint32_t scalars_in_writer = get_compile_time_arg_val(23);
    // Barrier-read threshold: push+barrier the input read every this many tiles
    // (caps in-flight DRAM reads to limit NoC/DRAM contention; also sets PRE's
    // consumption granularity). Host heuristic = f(num readers, tile bytes), or
    // WAN_BARRIER_TILES override. Subsumes the old per-block (RoPE) / whole-row
    // (no-rope) input-read structure: threshold == block_size reproduces the
    // RoPE finer push; threshold == num_tile_cols reproduces the whole-row push.
    constexpr uint32_t input_barrier_tiles = get_compile_time_arg_val(24);
    constexpr auto input_args = TensorAccessorArgs<25>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto transformation_mat_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto rope_cos_args = TensorAccessorArgs<transformation_mat_args.next_compile_time_args_offset()>();
    constexpr auto rope_sin_args = TensorAccessorArgs<rope_cos_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t transformation_mat_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rope_cos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rope_sin_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t input_tile_bytes = get_tile_size(input_cb);
    const uint32_t weight_tile_bytes = get_tile_size(weight_cb);
    const uint32_t bias_tile_bytes = get_tile_size(bias_cb);
    const uint32_t rope_cos_tile_bytes = get_tile_size(rope_cos_cb);
    const uint32_t rope_sin_tile_bytes = get_tile_size(rope_sin_cb);

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr);
    const auto transformation_mat_accessor = TensorAccessor(transformation_mat_args, transformation_mat_addr);
    const auto rope_cos_accessor = TensorAccessor(rope_cos_args, rope_cos_addr);
    const auto rope_sin_accessor = TensorAccessor(rope_sin_args, rope_sin_addr);

    // Row-broadcast weight / bias live in a TILE-layout [1, H] tensor where
    // only the first face-row of each face carries data — the rest is zero.
    // Reading just face_row_bytes per face avoids paying the full 4 KB/tile
    // bandwidth cost (measured ~13% e2e win on the N=2368 Wan config).
    constexpr uint32_t bf16_datum_size_bytes = 2;
    constexpr uint32_t face_row_bytes = tt::constants::FACE_WIDTH * bf16_datum_size_bytes;
    constexpr uint32_t face_bytes = tt::constants::FACE_HW * bf16_datum_size_bytes;

    // Generate reduce scalars (SUM for pre uses 1.0, AVG for post uses 1/H_full)
    // and the eps tile. Skipped when the writer populates them (MUX path) so the
    // reader can start the input read immediately.
    if constexpr (scalars_in_writer == 0) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            reduce_scalar_sum_cb,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>();
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            reduce_scalar_avg_cb,
            ckernel::PoolType::AVG,
            ckernel::ReduceDim::REDUCE_ROW,
            reduce_factor>();
        generate_bcast_col_scalar(epsilon_cb, epsilon_value);

        if constexpr (fuse_rope) {
            cb_reserve_back(transformation_mat_cb, 1);
            uint32_t transformation_mat_wr_ptr = get_write_ptr(transformation_mat_cb);
            noc_async_read_tile(0, transformation_mat_accessor, transformation_mat_wr_ptr);
            noc_async_read_barrier();
            cb_push_back(transformation_mat_cb, 1);
        }
    }

    // Weight + bias are consumed in the POST phase (sub-phases 2 / 2.5) which
    // only start after chunk 0's AG completes. So both reads can be deferred
    // until chunk 0's input rows are all pushed — the latency then hides
    // behind chunk 0's pre compute + fabric mcast + fabric wait. Issued in
    // `block_size`-sized pushes so the compute kernel can consume cumulatively.
    bool weight_pushed = (has_weight == 0);
    bool bias_pushed = (has_bias == 0);

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        // Deep input read: issue the whole row's tiles, then ONE barrier, so
        // num_tile_cols reads are in flight at once instead of block_size. The
        // per-2-tile barrier this replaces was the dominant DRAM-read-latency
        // bottleneck — the op ran at ~44% of BH's 512 GB/s peak because only 2
        // reads were ever outstanding and each barrier exposed full round-trip
        // latency. input_cb is sized to 2 * chunk_size_rows full rows, an
        // integer multiple of num_tile_cols, so a row's reservation never wraps
        // the ring (wr_ptr stays contiguous). Compute consumes cumulatively, so
        // a coarser push granularity is transparent to it.
        // Issue cos/sin reads FIRST (few tiles), then the input row, under a
        // SINGLE barrier. cos/sin reads overlap with the input burst, and input
        // is pushed first (immediately after the barrier) so compute can begin
        // the PRE sum-of-squares without waiting on cos/sin — those aren't
        // consumed until the much-later POST RoPE sub-phase. This removes the
        // second per-row barrier (one latency exposure instead of two) without
        // delaying input availability, recovering the NoC-contention regression
        // the deep-read change introduced on compute-bound RoPE shapes.
        // Idea A (read-overlap): issue the INPUT row FIRST and barrier it alone
        // (no cos/sin in flight yet), so compute's PRE sum-of-squares starts as
        // soon as input lands. cos/sin are issued AFTER the input push below, so
        // their DRAM read latency overlaps compute's PRE pass instead of gating
        // it (cos/sin aren't consumed until the post-AG RoPE phase). Previously
        // a single shared barrier made input wait for the cos/sin reads too.
        const uint32_t input_tile_idx = tile_row * num_tile_cols;
        if constexpr (streaming_low_l1) {
            // Stream the row twice in block_size-tile pushes: pass 0 feeds
            // compute's PRE sum-of-squares, pass 1 feeds the POST x*(1/rms)
            // re-read. input_cb is block-sized so compute pops each block,
            // keeping the resident footprint constant in num_tile_cols.
            DeviceZoneScopedN("R_INPUT");
            for (uint32_t pass = 0; pass < 2; pass++) {
                for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                    const uint32_t tiles_in_block =
                        ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                    cb_reserve_back(input_cb, tiles_in_block);
                    uint32_t input_wr_ptr = get_write_ptr(input_cb);
                    for (uint32_t i = 0; i < tiles_in_block; i++) {
#ifndef WAN_ABL_SKIP_INPUT_READ
                        noc_async_read_tile(input_tile_idx + col_tile + i, input_accessor, input_wr_ptr);
#endif
                        input_wr_ptr += input_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(input_cb, tiles_in_block);
                }
            }
        } else {
            // Barrier-read-threshold input read: push + barrier every
            // input_barrier_tiles tiles. Caps in-flight DRAM reads (contention)
            // and sets PRE's consumption granularity (compute overlap). The host
            // heuristic picks the threshold from the active-reader count + tile
            // bytes (Wormhole reference formula); WAN_BARRIER_TILES overrides.
            DeviceZoneScopedN("R_INPUT");
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += input_barrier_tiles) {
                const uint32_t grp = ((num_tile_cols - col_tile) >= input_barrier_tiles) ? input_barrier_tiles
                                                                                         : (num_tile_cols - col_tile);
                cb_reserve_back(input_cb, grp);
                uint32_t input_wr_ptr = get_write_ptr(input_cb);
                for (uint32_t i = 0; i < grp; i++) {
#ifndef WAN_ABL_SKIP_INPUT_READ
                    noc_async_read_tile(input_tile_idx + col_tile + i, input_accessor, input_wr_ptr);
#endif
                    input_wr_ptr += input_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(input_cb, grp);
            }
        }

        // (cos/sin moved BELOW the weight/bias reads — compute consumes weight
        // (POST sub-phase 2) before rope (sub-phase 3), so weight is read first.)

        // Per-token weight / bias: push this row's slice now, in block_size
        // tiles. Full-tile reads since per-token data isn't face-row sparse.
        // Compute kernel pops these per-row.
        if constexpr (per_token_weight != 0) {
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_reserve_back(weight_cb, tiles_in_block);
                uint32_t weight_wr_ptr = get_write_ptr(weight_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    const uint32_t w_idx = tile_row * num_tile_cols + col_tile + i;
#ifndef WAN_ABL_SKIP_WEIGHT_READ
                    noc_async_read_tile(w_idx, weight_accessor, weight_wr_ptr);
#endif
                    weight_wr_ptr += weight_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(weight_cb, tiles_in_block);
            }
        }
        if constexpr (per_token_bias != 0) {
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_reserve_back(bias_cb, tiles_in_block);
                uint32_t bias_wr_ptr = get_write_ptr(bias_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    const uint32_t b_idx = tile_row * num_tile_cols + col_tile + i;
#ifndef WAN_ABL_SKIP_WEIGHT_READ
                    noc_async_read_tile(b_idx, bias_accessor, bias_wr_ptr);
#endif
                    bias_wr_ptr += bias_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(bias_cb, tiles_in_block);
            }
        }

        // Broadcast weight + bias: after chunk 0's rows are pushed (or at
        // end-of-worker if the worker has fewer rows than chunk_size_rows),
        // issue the reads once for the whole worker. Latency hides behind
        // chunk 0's pre compute + fabric mcast + fabric wait.
        const uint32_t rows_pushed = tile_row + 1 - tile_row_start;
        const bool first_chunk_done = (rows_pushed >= chunk_size_rows);
        const bool last_row = (tile_row + 1 == tile_row_end);
        const bool should_issue_side_inputs = first_chunk_done || last_row;
        if constexpr (per_token_weight == 0) {
            if (!weight_pushed && should_issue_side_inputs) {
                for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                    const uint32_t tiles_in_block =
                        ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                    cb_reserve_back(weight_cb, tiles_in_block);
                    uint32_t weight_wr_ptr = get_write_ptr(weight_cb);
                    for (uint32_t i = 0; i < tiles_in_block; i++) {
                        uint64_t weight_noc_addr = get_noc_addr(col_tile + i, weight_accessor);
#ifndef WAN_ABL_SKIP_WEIGHT_READ
                        noc_async_read(weight_noc_addr, weight_wr_ptr, face_row_bytes);
                        noc_async_read(weight_noc_addr + face_bytes, weight_wr_ptr + face_bytes, face_row_bytes);
#endif
                        weight_wr_ptr += weight_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(weight_cb, tiles_in_block);
                }
                weight_pushed = true;
            }
        }
        if constexpr (per_token_bias == 0) {
            if (!bias_pushed && should_issue_side_inputs) {
                for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                    const uint32_t tiles_in_block =
                        ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                    cb_reserve_back(bias_cb, tiles_in_block);
                    uint32_t bias_wr_ptr = get_write_ptr(bias_cb);
                    for (uint32_t i = 0; i < tiles_in_block; i++) {
                        uint64_t bias_noc_addr = get_noc_addr(col_tile + i, bias_accessor);
#ifndef WAN_ABL_SKIP_WEIGHT_READ
                        noc_async_read(bias_noc_addr, bias_wr_ptr, face_row_bytes);
                        noc_async_read(bias_noc_addr + face_bytes, bias_wr_ptr + face_bytes, face_row_bytes);
#endif
                        bias_wr_ptr += bias_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(bias_cb, tiles_in_block);
                }
                bias_pushed = true;
            }
        }

        // cos/sin for the WHOLE chunk, read AFTER the chunk's input rows AND the
        // weight/bias reads (compute's POST consumes weight in sub-phase 2 and
        // rope in sub-phase 3, so weight is read first; matches the compute loop
        // order). Keeping these reads out from between the input rows lets the
        // input flow uninterrupted on the NoC. cos/sin aren't consumed until the
        // post-AG RoPE phase, so reading them here leaves them ready in time and
        // overlaps the AG wait + POST. Still pushed per row so POST can start as
        // each row lands.
        if constexpr (fuse_rope) {
            const uint32_t pos_in_chunk = (tile_row - tile_row_start) % chunk_size_rows;
            const bool chunk_input_done = (pos_in_chunk + 1 == chunk_size_rows) || (tile_row + 1 == tile_row_end);
            if (chunk_input_done) {
                const uint32_t chunk_first = tile_row - pos_in_chunk;
                for (uint32_t r = chunk_first; r <= tile_row; r++) {
                    // Per-head RoPE: cos/sin shape [B, num_heads, N, head_dim] — all
                    // heads' head_dim_tiles tiles for row r, idx
                    // h*rope_seqlen_tiles*head_dim_tiles + r*head_dim_tiles + c.
                    // Broadcast (per_head_rope=0): [B,1,N,head_dim], idx r*head_dim_tiles+c.
                    // This row's cos/sin tiles, laid out contiguously in the CB as
                    // [head0: head_dim_tiles, head1: ...] for per-head RoPE (total
                    // num_tile_cols), or head_dim_tiles for broadcast. Read them in
                    // block_size-tile groups — the SAME granularity P_ROPE consumes
                    // them at — barriering + pushing each group, instead of issuing
                    // the whole row (2*num_tile_cols reads) under a single barrier.
                    // This caps outstanding rope reads at 2*block_size (cos+sin) so the
                    // deep reader can't oversubscribe the NoC read-response queue under
                    // many chunks (the selfattn_qk_s2 traced hang: ~2300 reads issued
                    // but unreturned). Compute's cumulative cb_wait_front absorbs the
                    // block-wise pushes (same pattern as the input read above).
                    const uint32_t rope_tiles_this_row = (per_head_rope != 0) ? num_tile_cols : head_dim_tiles;
                    for (uint32_t t0 = 0; t0 < rope_tiles_this_row; t0 += block_size) {
                        const uint32_t grp =
                            ((rope_tiles_this_row - t0) >= block_size) ? block_size : (rope_tiles_this_row - t0);
                        cb_reserve_back(rope_cos_cb, grp);
                        cb_reserve_back(rope_sin_cb, grp);
                        uint32_t rope_cos_wr_ptr = get_write_ptr(rope_cos_cb);
                        uint32_t rope_sin_wr_ptr = get_write_ptr(rope_sin_cb);
                        for (uint32_t j = 0; j < grp; j++) {
                            const uint32_t t = t0 + j;
                            uint32_t src_idx;
                            if constexpr (per_head_rope != 0) {
                                // tile t -> head (t / head_dim_tiles), within-head (t % head_dim_tiles)
                                const uint32_t h = t / head_dim_tiles;
                                const uint32_t within = t - h * head_dim_tiles;
                                src_idx = h * rope_seqlen_tiles * head_dim_tiles + r * head_dim_tiles + within;
                            } else {
                                src_idx = r * head_dim_tiles + t;
                            }
#ifndef WAN_ABL_SKIP_ROPE_READ
                            noc_async_read_tile(src_idx, rope_cos_accessor, rope_cos_wr_ptr);
                            noc_async_read_tile(src_idx, rope_sin_accessor, rope_sin_wr_ptr);
#endif
                            rope_cos_wr_ptr += rope_cos_tile_bytes;
                            rope_sin_wr_ptr += rope_sin_tile_bytes;
                        }
                        noc_async_read_barrier();
                        cb_push_back(rope_cos_cb, grp);
                        cb_push_back(rope_sin_cb, grp);
                    }
                }
            }
        }
    }
}
