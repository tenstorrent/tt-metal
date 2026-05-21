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
#include <tt-metalium/constants.hpp>
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(1);
    constexpr uint32_t rope_cos_cb = get_compile_time_arg_val(2);
    constexpr uint32_t rope_sin_cb = get_compile_time_arg_val(3);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);
    constexpr uint32_t has_weight = get_compile_time_arg_val(6);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(7);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t chunk_size_rows = 1u;  // chunk size is always 1 (no CT arg)
    constexpr uint32_t per_head_rope = get_compile_time_arg_val(9);
    constexpr uint32_t rope_seqlen_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(11);
    constexpr uint32_t has_bias = get_compile_time_arg_val(12);
    // Per-token weight/bias: shape [N, H] (vs broadcast [1, H]). Read pattern
    // is per-row (after each row's input is pushed) using noc_async_read_tile
    // for full 4 KB/tile (vs face_row_bytes for the broadcast case). Compute
    // uses mul_tiles / add_tiles directly (no _bcast_rows).
    constexpr uint32_t per_token_weight = get_compile_time_arg_val(13);
    constexpr uint32_t per_token_bias = get_compile_time_arg_val(14);
    // Streaming low-L1: input_cb is block-sized, so the row is read in two
    // passes (PRE sum-of-squares, then a POST re-read for x*(1/rms)) in
    // block_size-tile pushes that compute pops as it consumes. The resident
    // fast path reads the whole row once. See program_factory.
    constexpr uint32_t streaming_low_l1 = get_compile_time_arg_val(15);
    // input_schedule: WHERE the streaming input passes are read relative to the
    // resident weight/bias/cos pushes. The block-major POST consumes weight/bias/cos
    // mid-(POST pass), so they must be pushed BEFORE the POST pass — but PRE needs the
    // PRE pass, and on the AG path the local stats (from PRE) must be produced ASAP so
    // the ring gather isn't delayed. Three schedules:
    //   0 INPUT_FIRST: read all input at the top (resident: whole row once; streaming:
    //     both passes). Used when the POST is resident (no mid-pass side-input wait) or
    //     non-streaming. Streaming block-major would deadlock here (POST waits weight,
    //     reader can't finish pushing the POST pass to reach the weight push).
    //   1 DEFER_ALL: push side inputs first, then BOTH passes (is_tp_1 block-major). No
    //     AG between PRE and POST, so delaying PRE is free; weight is resident first.
    //   2 SPLIT: read the PRE pass at the top (stats produced ASAP -> ring gather starts),
    //     push side inputs, then read the POST pass (weight now resident). The AG
    //     (ring>1) block-major path: avoids the deadlock without delaying the gather.
    constexpr uint32_t input_schedule = get_compile_time_arg_val(16);
    constexpr uint32_t SCHED_INPUT_FIRST = 0u;
    constexpr uint32_t SCHED_DEFER_ALL = 1u;
    constexpr uint32_t SCHED_SPLIT = 2u;
    // Welford reciprocal LUT (LayerNorm). When use_recip_lut, read the reciprocals DRAM
    // tensor once into recip_lut_cb at the top so compute reads it as the LLK's
    // reciprocal_lut (array load vs soft-float 1/(N+1) per sample). recip accessor is the
    // last TensorAccessorArgs; recip DRAM addr is reader RT arg 7.
    constexpr uint32_t use_recip_lut = get_compile_time_arg_val(17);
    constexpr uint32_t recip_lut_cb = get_compile_time_arg_val(18);
    // Broadcast affine read count (CT 19/20): num_tile_cols — only TRUE broadcast [1,1,H]
    // weight/bias uses the one-shot resident face-row read. per-batch adaLN streams per row.
    constexpr uint32_t weight_bcast_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t bias_bcast_tiles = get_compile_time_arg_val(20);
    // Batched RoPE (CT 21): per-batch stride into cos/sin, in tiles (== one batch's whole cos/sin
    // block). 0 -> broadcast the same cos/sin to every input batch. The reader always indexes
    // cos/sin by the WITHIN-batch seq row (global_row % rope_seqlen_tiles) and adds
    // (global_row / rope_seqlen_tiles) * rope_batch_stride_tiles; at batch=1 both terms collapse
    // to the original single-batch indexing.
    constexpr uint32_t rope_batch_stride_tiles = get_compile_time_arg_val(21);
    // Per-batch adaLN weight/bias (CT 22/23/24): [batch,1,H] — broadcast over seq but distinct per
    // batch. Streamed like per-token (per-row push + compute pops per row), but the read is the
    // face-row broadcast read at wbatch*num_tile_cols, wbatch = tile_row / rows_per_batch_tiles.
    constexpr uint32_t per_batch_weight = get_compile_time_arg_val(22);
    constexpr uint32_t per_batch_bias = get_compile_time_arg_val(23);
    constexpr uint32_t rows_per_batch_tiles = get_compile_time_arg_val(24);
    // The WRITER always populates the reduce_scalar_* / epsilon / trans_mat CBs,
    // so the reader's first NoC op is the input read (starts streaming ASAP).
    constexpr auto input_args = TensorAccessorArgs<25>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto rope_cos_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto rope_sin_args = TensorAccessorArgs<rope_cos_args.next_compile_time_args_offset()>();
    constexpr auto recip_args = TensorAccessorArgs<rope_sin_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rope_cos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rope_sin_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recip_addr = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t input_tile_bytes = get_tile_size(input_cb);
    const uint32_t weight_tile_bytes = get_tile_size(weight_cb);
    const uint32_t bias_tile_bytes = get_tile_size(bias_cb);
    const uint32_t rope_cos_tile_bytes = get_tile_size(rope_cos_cb);
    const uint32_t rope_sin_tile_bytes = get_tile_size(rope_sin_cb);

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr);
    const auto rope_cos_accessor = TensorAccessor(rope_cos_args, rope_cos_addr);
    const auto rope_sin_accessor = TensorAccessor(rope_sin_args, rope_sin_addr);

    // Welford reciprocal LUT: read the whole [1, reduce_width] fp32 page (one DRAM page,
    // reduce_width == num_tile_cols * 32 -> num_tile_cols * 128 bytes) into recip_lut_cb
    // ONCE, before any row work, so compute's first welford_update has it. Compute reads
    // it as std::array<uint32_t, reduce_width>; absent -> compute uses runtime division.
    if constexpr (use_recip_lut) {
        const auto recip_accessor = TensorAccessor(recip_args, recip_addr);
        constexpr uint32_t recip_bytes = num_tile_cols * 128u;  // reduce_width * sizeof(float)
        cb_reserve_back(recip_lut_cb, 1);
        noc_async_read(get_noc_addr(0, recip_accessor), get_write_ptr(recip_lut_cb), recip_bytes);
        noc_async_read_barrier();
        cb_push_back(recip_lut_cb, 1);
    }

    // Row-broadcast weight / bias live in a TILE-layout [1, H] tensor where
    // only the first face-row of each face carries data — the rest is zero.
    // Reading just face_row_bytes per face avoids paying the full tile bandwidth
    // cost (measured ~13% e2e win on the N=2368 Wan config). The datum size is
    // derived from the CB tile size (tile_bytes / TILE_HW) so the read works for
    // both bf16 (2 B) and fp32 (4 B) weight/bias.
    constexpr uint32_t kTileHW = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;  // 1024
    const uint32_t weight_datum_bytes = weight_tile_bytes / kTileHW;
    const uint32_t weight_face_row_bytes = tt::constants::FACE_WIDTH * weight_datum_bytes;
    const uint32_t weight_face_bytes = tt::constants::FACE_HW * weight_datum_bytes;
    const uint32_t bias_datum_bytes = bias_tile_bytes / kTileHW;
    const uint32_t bias_face_row_bytes = tt::constants::FACE_WIDTH * bias_datum_bytes;
    const uint32_t bias_face_bytes = tt::constants::FACE_HW * bias_datum_bytes;

    // Weight + bias are consumed in the POST phase (sub-phases 2 / 2.5) which
    // only start after chunk 0's AG completes. So both reads can be deferred
    // until chunk 0's input rows are all pushed — the latency then hides
    // behind chunk 0's pre compute + fabric mcast + fabric wait. Issued in
    // `block_size`-sized pushes so the compute kernel can consume cumulatively.
    bool weight_pushed = (has_weight == 0);
    bool bias_pushed = (has_bias == 0);

    // Read one full pass over a tile-row's input (num_tile_cols tiles) in block_size
    // pushes, deep-barriered per block. Streaming reads this twice (PRE then a POST
    // re-read); resident reads it once (compute holds the whole row). The schedule
    // logic below decides WHEN each pass runs relative to the side-input pushes.
    auto read_input_pass = [&](uint32_t input_tile_idx) {
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const uint32_t tiles_in_block =
                ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
            cb_reserve_back(input_cb, tiles_in_block);
            uint32_t input_wr_ptr = get_write_ptr(input_cb);
            for (uint32_t i = 0; i < tiles_in_block; i++) {
                noc_async_read_tile(input_tile_idx + col_tile + i, input_accessor, input_wr_ptr);
                input_wr_ptr += input_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(input_cb, tiles_in_block);
        }
    };

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        // Deep input read: issue the whole row's tiles, then ONE barrier, so
        // num_tile_cols reads are in flight at once (keeps DRAM-read latency hidden;
        // a per-block barrier would cap outstanding reads at block_size and expose
        // the round-trip each time). input_cb is sized to 2 * chunk_size_rows full
        // rows, an integer multiple of num_tile_cols, so a row's reservation never
        // wraps the ring (wr_ptr stays contiguous). Compute consumes cumulatively,
        // so the coarser push granularity is transparent to it.
        // Order: issue + push the INPUT row FIRST, barriered alone, so compute's PRE
        // sum-of-squares starts as soon as input lands; cos/sin are issued AFTER (see
        // below) so their DRAM read latency overlaps PRE — they aren't consumed until
        // the POST RoPE phase.
        const uint32_t input_tile_idx = tile_row * num_tile_cols;
        // Input read placement is schedule-driven (see input_schedule above):
        //   INPUT_FIRST: read everything HERE so PRE starts ASAP — streaming = both
        //     passes (PRE + POST re-read), resident = the whole row once.
        //   SPLIT: read ONLY the PRE pass here (streaming) so the local stats are
        //     produced ASAP and the ring gather isn't delayed; the POST pass is read
        //     below, after the side inputs are resident.
        //   DEFER_ALL: read nothing here; both passes are read below, after side inputs.
        if constexpr (input_schedule == SCHED_INPUT_FIRST) {
            DeviceZoneScopedN("R_INPUT");
            read_input_pass(input_tile_idx);
            if constexpr (streaming_low_l1) {
                read_input_pass(input_tile_idx);  // POST re-read pass
            }
        } else if constexpr (input_schedule == SCHED_SPLIT) {
            DeviceZoneScopedN("R_INPUT");
            read_input_pass(input_tile_idx);  // PRE pass only; POST pass deferred below
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
                    noc_async_read_tile(w_idx, weight_accessor, weight_wr_ptr);
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
                    noc_async_read_tile(b_idx, bias_accessor, bias_wr_ptr);
                    bias_wr_ptr += bias_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(bias_cb, tiles_in_block);
            }
        }

        // Per-batch adaLN weight / bias: this row's batch slice, pushed per row (compute pops per
        // row, mul_bcast_rows). Broadcast over seq -> face-row read (one real row per batch), at
        // wbatch*num_tile_cols where wbatch = tile_row / rows_per_batch_tiles. Streamed (not
        // all-batches-resident), so weight_cb stays 1 row and a wide per-batch shard fits L1.
        if constexpr (per_batch_weight != 0) {
            const uint32_t w_base =
                ((rows_per_batch_tiles != 0) ? (tile_row / rows_per_batch_tiles) : 0u) * num_tile_cols;
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_reserve_back(weight_cb, tiles_in_block);
                uint32_t weight_wr_ptr = get_write_ptr(weight_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    uint64_t weight_noc_addr = get_noc_addr(w_base + col_tile + i, weight_accessor);
                    noc_async_read(weight_noc_addr, weight_wr_ptr, weight_face_row_bytes);
                    noc_async_read(
                        weight_noc_addr + weight_face_bytes, weight_wr_ptr + weight_face_bytes, weight_face_row_bytes);
                    weight_wr_ptr += weight_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(weight_cb, tiles_in_block);
            }
        }
        if constexpr (per_batch_bias != 0) {
            const uint32_t b_base =
                ((rows_per_batch_tiles != 0) ? (tile_row / rows_per_batch_tiles) : 0u) * num_tile_cols;
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_reserve_back(bias_cb, tiles_in_block);
                uint32_t bias_wr_ptr = get_write_ptr(bias_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    uint64_t bias_noc_addr = get_noc_addr(b_base + col_tile + i, bias_accessor);
                    noc_async_read(bias_noc_addr, bias_wr_ptr, bias_face_row_bytes);
                    noc_async_read(bias_noc_addr + bias_face_bytes, bias_wr_ptr + bias_face_bytes, bias_face_row_bytes);
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
        if constexpr (per_token_weight == 0 && per_batch_weight == 0) {
            if (!weight_pushed && should_issue_side_inputs) {
                for (uint32_t col_tile = 0; col_tile < weight_bcast_tiles; col_tile += block_size) {
                    const uint32_t tiles_in_block =
                        ((weight_bcast_tiles - col_tile) >= block_size) ? block_size : (weight_bcast_tiles - col_tile);
                    cb_reserve_back(weight_cb, tiles_in_block);
                    uint32_t weight_wr_ptr = get_write_ptr(weight_cb);
                    for (uint32_t i = 0; i < tiles_in_block; i++) {
                        uint64_t weight_noc_addr = get_noc_addr(col_tile + i, weight_accessor);
                        noc_async_read(weight_noc_addr, weight_wr_ptr, weight_face_row_bytes);
                        noc_async_read(
                            weight_noc_addr + weight_face_bytes,
                            weight_wr_ptr + weight_face_bytes,
                            weight_face_row_bytes);
                        weight_wr_ptr += weight_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(weight_cb, tiles_in_block);
                }
                weight_pushed = true;
            }
        }
        if constexpr (per_token_bias == 0 && per_batch_bias == 0) {
            if (!bias_pushed && should_issue_side_inputs) {
                for (uint32_t col_tile = 0; col_tile < bias_bcast_tiles; col_tile += block_size) {
                    const uint32_t tiles_in_block =
                        ((bias_bcast_tiles - col_tile) >= block_size) ? block_size : (bias_bcast_tiles - col_tile);
                    cb_reserve_back(bias_cb, tiles_in_block);
                    uint32_t bias_wr_ptr = get_write_ptr(bias_cb);
                    for (uint32_t i = 0; i < tiles_in_block; i++) {
                        uint64_t bias_noc_addr = get_noc_addr(col_tile + i, bias_accessor);
                        noc_async_read(bias_noc_addr, bias_wr_ptr, bias_face_row_bytes);
                        noc_async_read(
                            bias_noc_addr + bias_face_bytes, bias_wr_ptr + bias_face_bytes, bias_face_row_bytes);
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
                    // Batched RoPE: fold the global row r into (batch b, within-batch seq row
                    // r_seq) so each input batch reuses cos/sin at seq row r_seq. b selects the
                    // per-batch cos/sin block via rope_batch_stride_tiles (0 => broadcast, all
                    // batches share block 0). At batch=1: r_seq==r, rope_batch_off==0 (unchanged).
                    const uint32_t r_seq = (rope_seqlen_tiles != 0) ? (r % rope_seqlen_tiles) : r;
                    const uint32_t rope_batch_off =
                        (rope_seqlen_tiles != 0) ? ((r / rope_seqlen_tiles) * rope_batch_stride_tiles) : 0u;
                    // Per-head RoPE: cos/sin shape [B, num_heads, N, head_dim] — all
                    // heads' head_dim_tiles tiles for row r_seq, idx (+ per-batch offset)
                    // h*rope_seqlen_tiles*head_dim_tiles + r_seq*head_dim_tiles + c.
                    // Broadcast (per_head_rope=0): [B,1,N,head_dim], idx r_seq*head_dim_tiles+c.
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
                                src_idx = rope_batch_off + h * rope_seqlen_tiles * head_dim_tiles +
                                          r_seq * head_dim_tiles + within;
                            } else {
                                src_idx = rope_batch_off + r_seq * head_dim_tiles + t;
                            }
                            noc_async_read_tile(src_idx, rope_cos_accessor, rope_cos_wr_ptr);
                            noc_async_read_tile(src_idx, rope_sin_accessor, rope_sin_wr_ptr);
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

        // Deferred block-major input: now that weight/bias/cos are resident, stream the
        // POST re-read pass(es) in block_size pushes. The block-major POST consumes them
        // with its side inputs already resident (no reader<->compute deadlock).
        //   DEFER_ALL (is_tp_1): both passes here — PRE pass 0, then POST pass 1.
        //   SPLIT (AG): only the POST pass here (the PRE pass already ran at the top, so
        //     the local stats / ring gather started before this side-input wait).
        if constexpr (input_schedule == SCHED_DEFER_ALL) {
            DeviceZoneScopedN("R_INPUT");
            read_input_pass(input_tile_idx);  // PRE pass
            read_input_pass(input_tile_idx);  // POST re-read pass
        } else if constexpr (input_schedule == SCHED_SPLIT) {
            DeviceZoneScopedN("R_INPUT");
            read_input_pass(input_tile_idx);  // POST re-read pass
        }
    }
}
