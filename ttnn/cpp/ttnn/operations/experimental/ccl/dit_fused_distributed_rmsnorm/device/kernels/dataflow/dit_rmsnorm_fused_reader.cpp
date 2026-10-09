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
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include <tt-metalium/constants.hpp>
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr std::uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr std::uint32_t weight_cb = get_compile_time_arg_val(1);
    constexpr std::uint32_t rope_cos_cb = get_compile_time_arg_val(2);
    constexpr std::uint32_t rope_sin_cb = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_tile_cols = get_compile_time_arg_val(4);
    constexpr std::uint32_t block_size = get_compile_time_arg_val(5);
    constexpr std::uint32_t has_weight = get_compile_time_arg_val(6);
    constexpr std::uint32_t fuse_rope = get_compile_time_arg_val(7);
    constexpr std::uint32_t head_dim_tiles = get_compile_time_arg_val(8);
    constexpr std::uint32_t chunk_size_rows = 1u;  // chunk size is always 1 (no CT arg)
    constexpr std::uint32_t per_head_rope = get_compile_time_arg_val(9);
    constexpr std::uint32_t rope_seqlen_tiles = get_compile_time_arg_val(10);
    constexpr std::uint32_t bias_cb = get_compile_time_arg_val(11);
    constexpr std::uint32_t has_bias = get_compile_time_arg_val(12);
    // Per-token weight/bias: shape [N, H] (vs broadcast [1, H]). Read pattern
    // is per-row (after each row's input is pushed) using a full-page
    // noc.async_read for full 4 KB/tile (vs face_row_bytes for the broadcast
    // case). Compute uses mul_tiles / add_tiles directly (no _bcast_rows).
    constexpr std::uint32_t per_token_weight = get_compile_time_arg_val(13);
    constexpr std::uint32_t per_token_bias = get_compile_time_arg_val(14);
    // Streaming low-L1: input_cb is block-sized, so the row is read in two
    // passes (PRE sum-of-squares, then a POST re-read for x*(1/rms)) in
    // block_size-tile pushes that compute pops as it consumes. The resident
    // fast path reads the whole row once. See program_factory.
    constexpr std::uint32_t streaming_low_l1 = get_compile_time_arg_val(15);
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
    constexpr std::uint32_t input_schedule = get_compile_time_arg_val(16);
    constexpr std::uint32_t SCHED_INPUT_FIRST = 0u;
    constexpr std::uint32_t SCHED_DEFER_ALL = 1u;
    constexpr std::uint32_t SCHED_SPLIT = 2u;
    // Welford reciprocal LUT (LayerNorm). When use_recip_lut, read the reciprocals DRAM
    // tensor once into recip_lut_cb at the top so compute reads it as the LLK's
    // reciprocal_lut (array load vs soft-float 1/(N+1) per sample). recip accessor is the
    // last TensorAccessorArgs; recip DRAM addr is reader RT arg 7.
    constexpr std::uint32_t use_recip_lut = get_compile_time_arg_val(17);
    constexpr std::uint32_t recip_lut_cb = get_compile_time_arg_val(18);
    // Broadcast affine read count (CT 19/20): num_tile_cols — only TRUE broadcast [1,1,H]
    // weight/bias uses the one-shot resident face-row read. per-batch adaLN streams per row.
    constexpr std::uint32_t weight_bcast_tiles = get_compile_time_arg_val(19);
    constexpr std::uint32_t bias_bcast_tiles = get_compile_time_arg_val(20);
    // Batched RoPE (CT 21): per-batch stride into cos/sin, in tiles (== one batch's whole cos/sin
    // block). 0 -> broadcast the same cos/sin to every input batch. The reader always indexes
    // cos/sin by the WITHIN-batch seq row (global_row % rope_seqlen_tiles) and adds
    // (global_row / rope_seqlen_tiles) * rope_batch_stride_tiles; at batch=1 both terms collapse
    // to the original single-batch indexing.
    constexpr std::uint32_t rope_batch_stride_tiles = get_compile_time_arg_val(21);
    // Per-batch adaLN weight/bias (CT 22/23/24): [batch,1,H] — broadcast over seq but distinct per
    // batch. Streamed like per-token (per-row push + compute pops per row), but the read is the
    // face-row broadcast read at wbatch*num_tile_cols, wbatch = tile_row / rows_per_batch_tiles.
    constexpr std::uint32_t per_batch_weight = get_compile_time_arg_val(22);
    constexpr std::uint32_t per_batch_bias = get_compile_time_arg_val(23);
    constexpr std::uint32_t rows_per_batch_tiles = get_compile_time_arg_val(24);
    // The WRITER always populates the reduce_scalar_* / epsilon / trans_mat CBs,
    // so the reader's first NoC op is the input read (starts streaming ASAP).
    constexpr auto input_args = TensorAccessorArgs<25>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto rope_cos_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto rope_sin_args = TensorAccessorArgs<rope_cos_args.next_compile_time_args_offset()>();
    constexpr auto recip_args = TensorAccessorArgs<rope_sin_args.next_compile_time_args_offset()>();
    // Broadcast gamma read by the worker writer (BRISC) at kernel start instead of here, so the
    // NCRISC read queue carries only the input row (see the factory's writer_reads_weight).
    constexpr std::uint32_t weight_from_writer = get_compile_time_arg_val(recip_args.next_compile_time_args_offset());
    // Two-wave column split: the DRAM row stride (the worker reads num_tile_cols tiles from col_offset),
    // and the start semaphore a wave-B worker waits on. Its wave-A partner ups it once its own read is
    // kWaveSignalLeadBlocks blocks from landing, so B's reads queue right behind A's and each wave reads
    // with the DRAM to itself.
    constexpr std::uint32_t row_stride_tiles = get_compile_time_arg_val(recip_args.next_compile_time_args_offset() + 1);
    constexpr std::uint32_t start_sem_id = get_compile_time_arg_val(recip_args.next_compile_time_args_offset() + 2);
    constexpr std::uint32_t kWaveSignal = 1u;
    constexpr std::uint32_t kWaveWait = 2u;
    constexpr std::uint32_t kWaveSignalLeadBlocks = 2u;

    std::uint32_t arg_idx = 0;
    const std::uint32_t input_addr = get_common_arg_val<std::uint32_t>(0);
    const std::uint32_t weight_addr = get_common_arg_val<std::uint32_t>(1);
    const std::uint32_t bias_addr = get_common_arg_val<std::uint32_t>(2);
    const std::uint32_t rope_cos_addr = get_common_arg_val<std::uint32_t>(3);
    const std::uint32_t rope_sin_addr = get_common_arg_val<std::uint32_t>(4);
    const std::uint32_t tile_row_start = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t tile_row_end = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t recip_addr = get_common_arg_val<std::uint32_t>(5);
    const std::uint32_t col_offset = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t wave_role = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t partner_x = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t partner_y = get_arg_val<std::uint32_t>(arg_idx++);

    Noc noc;

    CircularBuffer cb_input(input_cb);
    CircularBuffer cb_weight(weight_cb);
    CircularBuffer cb_bias(bias_cb);
    CircularBuffer cb_rope_cos(rope_cos_cb);
    CircularBuffer cb_rope_sin(rope_sin_cb);

    const std::uint32_t input_tile_bytes = cb_input.get_tile_size();
    const std::uint32_t weight_tile_bytes = cb_weight.get_tile_size();
    const std::uint32_t bias_tile_bytes = cb_bias.get_tile_size();
    const std::uint32_t rope_cos_tile_bytes = cb_rope_cos.get_tile_size();
    const std::uint32_t rope_sin_tile_bytes = cb_rope_sin.get_tile_size();

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr);
    const auto rope_cos_accessor = TensorAccessor(rope_cos_args, rope_cos_addr);
    const auto rope_sin_accessor = TensorAccessor(rope_sin_args, rope_sin_addr);

    // Full-page read sizes. The face-row reads further down deliberately do NOT use
    // these — they read a fraction of a page.
    const std::uint32_t input_page_bytes = input_accessor.get_aligned_page_size();
    const std::uint32_t weight_page_bytes = weight_accessor.get_aligned_page_size();
    const std::uint32_t bias_page_bytes = bias_accessor.get_aligned_page_size();
    const std::uint32_t rope_cos_page_bytes = rope_cos_accessor.get_aligned_page_size();
    const std::uint32_t rope_sin_page_bytes = rope_sin_accessor.get_aligned_page_size();

    // Welford reciprocal LUT: read the whole [1, reduce_width] fp32 page (one DRAM page,
    // reduce_width == num_tile_cols * 32 -> num_tile_cols * 128 bytes) into recip_lut_cb
    // ONCE, before any row work, so compute's first welford_update has it. Compute reads
    // it as std::array<uint32_t, reduce_width>; absent -> compute uses runtime division.
    if constexpr (use_recip_lut) {
        const auto recip_accessor = TensorAccessor(recip_args, recip_addr);
        CircularBuffer cb_recip_lut(recip_lut_cb);
        constexpr std::uint32_t recip_bytes = num_tile_cols * 128u;  // reduce_width * sizeof(float)
        cb_recip_lut.reserve_back(1);
        // Size is recip_bytes, not the accessor's page size: the LUT is one DRAM page but
        // we only want the [1, reduce_width] prefix.
        noc.async_read(recip_accessor, cb_recip_lut, recip_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
        cb_recip_lut.push_back(1);
    }

    // Row-broadcast weight / bias live in a TILE-layout [1, H] tensor where
    // only the first face-row of each face carries data — the rest is zero.
    // Reading just face_row_bytes per face avoids paying the full tile bandwidth
    // cost (measured ~13% e2e win on the N=2368 Wan config). The datum size is
    // derived from the CB tile size (tile_bytes / TILE_HW) so the read works for
    // both bf16 (2 B) and fp32 (4 B) weight/bias.
    constexpr std::uint32_t kTileHW = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;  // 1024
    const std::uint32_t weight_datum_bytes = weight_tile_bytes / kTileHW;
    const std::uint32_t weight_face_row_bytes = tt::constants::FACE_WIDTH * weight_datum_bytes;
    const std::uint32_t weight_face_bytes = tt::constants::FACE_HW * weight_datum_bytes;
    const std::uint32_t bias_datum_bytes = bias_tile_bytes / kTileHW;
    const std::uint32_t bias_face_row_bytes = tt::constants::FACE_WIDTH * bias_datum_bytes;
    const std::uint32_t bias_face_bytes = tt::constants::FACE_HW * bias_datum_bytes;

    // Weight + bias are consumed in the POST phase (sub-phases 2 / 2.5) which
    // only start after chunk 0's AG completes. So both reads can be deferred
    // until chunk 0's input rows are all pushed — the latency then hides
    // behind chunk 0's pre compute + fabric mcast + fabric wait. Issued in
    // `block_size`-sized pushes so the compute kernel can consume cumulatively.
    bool weight_pushed = (has_weight == 0) || (weight_from_writer != 0);
    bool bias_pushed = (has_bias == 0);

    // Read one full pass over a tile-row's input (num_tile_cols tiles) in block_size
    // pushes, deep-barriered per block. Streaming reads this twice (PRE then a POST
    // re-read); resident reads it once (compute holds the whole row). The schedule
    // logic below decides WHEN each pass runs relative to the side-input pushes.
    auto read_input_pass = [&](std::uint32_t input_tile_idx) {
        for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const std::uint32_t tiles_in_block =
                ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
            cb_input.reserve_back(tiles_in_block);
            std::uint32_t input_wr_ptr = cb_input.get_write_ptr();
            for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                noc.async_read(
                    input_accessor,
                    CoreLocalMem<std::uint32_t>(input_wr_ptr),
                    input_page_bytes,
                    {.page_id = input_tile_idx + col_tile + i},
                    {});
                input_wr_ptr += input_tile_bytes;
            }
            noc.async_read_barrier();
            cb_input.push_back(tiles_in_block);
        }
    };

    // Trid-pipelined resident input pass. Each block_size-tile block is tagged with its own
    // NoC read transaction id, so up to kInputLookahead blocks are in flight while we wait
    // on (and push) only the oldest one: PRE still consumes block by block, but the DRAM
    // round-trips overlap instead of serialising per block. When `with_weight`, the broadcast
    // gamma face-row reads for each block are issued right behind its input reads (one
    // shared trid), so gamma is resident as soon as the input row is, instead of a second
    // per-block-barriered pass after it.
    constexpr std::uint32_t kInputTrids = 14u;  // trids 1..14 for input blocks
    constexpr std::uint32_t kWeightTrid = 15u;
    constexpr std::uint32_t kInputLookahead = 4u;
    constexpr std::uint32_t num_input_blocks = (num_tile_cols + block_size - 1u) / block_size;
    constexpr std::uint32_t wave_signal_block =
        (num_input_blocks > kWaveSignalLeadBlocks) ? (num_input_blocks - 1u - kWaveSignalLeadBlocks) : 0u;
    auto read_input_pass_pipelined = [&](std::uint32_t input_tile_idx, bool with_weight) {
        cb_input.reserve_back(num_tile_cols);
        const std::uint32_t input_base = cb_input.get_write_ptr();
        std::uint32_t weight_base = 0;
        if (with_weight) {
            cb_weight.reserve_back(weight_bcast_tiles);
            weight_base = cb_weight.get_write_ptr();
        }
        for (std::uint32_t k = 0; k < num_input_blocks + kInputLookahead; k++) {
            if (k < num_input_blocks) {
                const std::uint32_t col_tile = k * block_size;
                const std::uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                const std::uint32_t trid = 1u + (k % kInputTrids);
                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                    noc.async_read<NocOptions::TXN_ID>(
                        input_accessor,
                        CoreLocalMem<std::uint32_t>(input_base + (col_tile + i) * input_tile_bytes),
                        input_page_bytes,
                        {.page_id = input_tile_idx + col_tile + i},
                        {},
                        {.trid = trid});
                }
                if (with_weight) {
                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                        const std::uint32_t w_page = col_tile + i;
                        const std::uint32_t w_dst = weight_base + w_page * weight_tile_bytes;
                        noc.async_read<NocOptions::TXN_ID>(
                            weight_accessor,
                            CoreLocalMem<std::uint32_t>(w_dst),
                            weight_face_row_bytes,
                            {.page_id = w_page},
                            {},
                            {.trid = kWeightTrid});
                        noc.async_read<NocOptions::TXN_ID>(
                            weight_accessor,
                            CoreLocalMem<std::uint32_t>(w_dst + weight_face_bytes),
                            weight_face_row_bytes,
                            {.page_id = w_page, .offset_bytes = weight_face_bytes},
                            {},
                            {.trid = kWeightTrid});
                    }
                }
            }
            if (k >= kInputLookahead) {
                const std::uint32_t kb = k - kInputLookahead;
                const std::uint32_t col_tile = kb * block_size;
                const std::uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1u + (kb % kInputTrids)});
                cb_input.push_back(tiles_in_block);
                if (kb == wave_signal_block && wave_role == kWaveSignal) {
                    Semaphore<>(start_sem_id).up(noc, partner_x, partner_y, 1);
                }
            }
        }
        if (with_weight) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = kWeightTrid});
            cb_weight.push_back(weight_bcast_tiles);
        }
        // Restore the default read trid for the plain reads below / later kernels.
        noc_async_read_set_trid(0, noc.get_noc_id());
    };
    static_assert(kInputLookahead < kInputTrids, "lookahead must not reuse an in-flight trid");

    for (std::uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
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
        const std::uint32_t input_tile_idx = tile_row * row_stride_tiles + col_offset;
        if (wave_role == kWaveWait && tile_row == tile_row_start) {
            DeviceZoneScopedN("R_WAVEWAIT");
            Semaphore<> start_sem(start_sem_id);
            start_sem.wait_min(1);
            start_sem.set(0);  // trace replay doesn't re-run the host semaphore init
        }
        // Input read placement is schedule-driven (see input_schedule above):
        //   INPUT_FIRST: read everything HERE so PRE starts ASAP — streaming = both
        //     passes (PRE + POST re-read), resident = the whole row once.
        //   SPLIT: read ONLY the PRE pass here (streaming) so the local stats are
        //     produced ASAP and the ring gather isn't delayed; the POST pass is read
        //     below, after the side inputs are resident.
        //   DEFER_ALL: read nothing here; both passes are read below, after side inputs.
        if constexpr (input_schedule == SCHED_INPUT_FIRST) {
            DeviceZoneScopedN("R_INPUT");
            if constexpr (streaming_low_l1) {
                read_input_pass(input_tile_idx);
                read_input_pass(input_tile_idx);  // POST re-read pass
            } else {
                // Broadcast gamma rides along with the first row's input (the deferred
                // weight read below then sees weight_pushed and skips).
                constexpr bool bcast_weight = (has_weight != 0) && (per_token_weight == 0) && (per_batch_weight == 0);
                const bool with_weight = bcast_weight && !weight_pushed;
                read_input_pass_pipelined(input_tile_idx, with_weight);
                if (with_weight) {
                    weight_pushed = true;
                }
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
            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const std::uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_weight.reserve_back(tiles_in_block);
                std::uint32_t weight_wr_ptr = cb_weight.get_write_ptr();
                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                    const std::uint32_t w_idx = tile_row * num_tile_cols + col_tile + i;
                    noc.async_read(
                        weight_accessor,
                        CoreLocalMem<std::uint32_t>(weight_wr_ptr),
                        weight_page_bytes,
                        {.page_id = w_idx},
                        {});
                    weight_wr_ptr += weight_tile_bytes;
                }
                noc.async_read_barrier();
                cb_weight.push_back(tiles_in_block);
            }
        }
        if constexpr (per_token_bias != 0) {
            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const std::uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_bias.reserve_back(tiles_in_block);
                std::uint32_t bias_wr_ptr = cb_bias.get_write_ptr();
                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                    const std::uint32_t b_idx = tile_row * num_tile_cols + col_tile + i;
                    noc.async_read(
                        bias_accessor,
                        CoreLocalMem<std::uint32_t>(bias_wr_ptr),
                        bias_page_bytes,
                        {.page_id = b_idx},
                        {});
                    bias_wr_ptr += bias_tile_bytes;
                }
                noc.async_read_barrier();
                cb_bias.push_back(tiles_in_block);
            }
        }

        // Per-batch adaLN weight / bias: this row's batch slice, pushed per row (compute pops per
        // row, mul_bcast_rows). Broadcast over seq -> face-row read (one real row per batch), at
        // wbatch*num_tile_cols where wbatch = tile_row / rows_per_batch_tiles. Streamed (not
        // all-batches-resident), so weight_cb stays 1 row and a wide per-batch shard fits L1.
        if constexpr (per_batch_weight != 0) {
            const std::uint32_t w_base =
                ((rows_per_batch_tiles != 0) ? (tile_row / rows_per_batch_tiles) : 0u) * num_tile_cols;
            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const std::uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_weight.reserve_back(tiles_in_block);
                std::uint32_t weight_wr_ptr = cb_weight.get_write_ptr();
                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                    // Two sub-page reads per tile: face_00 row 0, then face_01 row 0.
                    const std::uint32_t w_page = w_base + col_tile + i;
                    noc.async_read(
                        weight_accessor,
                        CoreLocalMem<std::uint32_t>(weight_wr_ptr),
                        weight_face_row_bytes,
                        {.page_id = w_page},
                        {});
                    noc.async_read(
                        weight_accessor,
                        CoreLocalMem<std::uint32_t>(weight_wr_ptr + weight_face_bytes),
                        weight_face_row_bytes,
                        {.page_id = w_page, .offset_bytes = weight_face_bytes},
                        {});
                    weight_wr_ptr += weight_tile_bytes;
                }
                noc.async_read_barrier();
                cb_weight.push_back(tiles_in_block);
            }
        }
        if constexpr (per_batch_bias != 0) {
            const std::uint32_t b_base =
                ((rows_per_batch_tiles != 0) ? (tile_row / rows_per_batch_tiles) : 0u) * num_tile_cols;
            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const std::uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_bias.reserve_back(tiles_in_block);
                std::uint32_t bias_wr_ptr = cb_bias.get_write_ptr();
                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                    // face_00 row 0 + face_01 row 0, as for weight above.
                    const std::uint32_t b_page = b_base + col_tile + i;
                    noc.async_read(
                        bias_accessor,
                        CoreLocalMem<std::uint32_t>(bias_wr_ptr),
                        bias_face_row_bytes,
                        {.page_id = b_page},
                        {});
                    noc.async_read(
                        bias_accessor,
                        CoreLocalMem<std::uint32_t>(bias_wr_ptr + bias_face_bytes),
                        bias_face_row_bytes,
                        {.page_id = b_page, .offset_bytes = bias_face_bytes},
                        {});
                    bias_wr_ptr += bias_tile_bytes;
                }
                noc.async_read_barrier();
                cb_bias.push_back(tiles_in_block);
            }
        }

        // Broadcast weight + bias: after chunk 0's rows are pushed (or at
        // end-of-worker if the worker has fewer rows than chunk_size_rows),
        // issue the reads once for the whole worker. Latency hides behind
        // chunk 0's pre compute + fabric mcast + fabric wait.
        const std::uint32_t rows_pushed = tile_row + 1 - tile_row_start;
        const bool first_chunk_done = (rows_pushed >= chunk_size_rows);
        const bool last_row = (tile_row + 1 == tile_row_end);
        const bool should_issue_side_inputs = first_chunk_done || last_row;
        if constexpr (per_token_weight == 0 && per_batch_weight == 0) {
            if (!weight_pushed && should_issue_side_inputs) {
                for (std::uint32_t col_tile = 0; col_tile < weight_bcast_tiles; col_tile += block_size) {
                    const std::uint32_t tiles_in_block =
                        ((weight_bcast_tiles - col_tile) >= block_size) ? block_size : (weight_bcast_tiles - col_tile);
                    cb_weight.reserve_back(tiles_in_block);
                    std::uint32_t weight_wr_ptr = cb_weight.get_write_ptr();
                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                        // face_00 row 0 + face_01 row 0, as for the per-batch read above.
                        const std::uint32_t w_page = col_tile + i;
                        noc.async_read(
                            weight_accessor,
                            CoreLocalMem<std::uint32_t>(weight_wr_ptr),
                            weight_face_row_bytes,
                            {.page_id = w_page},
                            {});
                        noc.async_read(
                            weight_accessor,
                            CoreLocalMem<std::uint32_t>(weight_wr_ptr + weight_face_bytes),
                            weight_face_row_bytes,
                            {.page_id = w_page, .offset_bytes = weight_face_bytes},
                            {});
                        weight_wr_ptr += weight_tile_bytes;
                    }
                    noc.async_read_barrier();
                    cb_weight.push_back(tiles_in_block);
                }
                weight_pushed = true;
            }
        }
        if constexpr (per_token_bias == 0 && per_batch_bias == 0) {
            if (!bias_pushed && should_issue_side_inputs) {
                for (std::uint32_t col_tile = 0; col_tile < bias_bcast_tiles; col_tile += block_size) {
                    const std::uint32_t tiles_in_block =
                        ((bias_bcast_tiles - col_tile) >= block_size) ? block_size : (bias_bcast_tiles - col_tile);
                    cb_bias.reserve_back(tiles_in_block);
                    std::uint32_t bias_wr_ptr = cb_bias.get_write_ptr();
                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                        // face_00 row 0 + face_01 row 0, as for the per-batch read above.
                        const std::uint32_t b_page = col_tile + i;
                        noc.async_read(
                            bias_accessor,
                            CoreLocalMem<std::uint32_t>(bias_wr_ptr),
                            bias_face_row_bytes,
                            {.page_id = b_page},
                            {});
                        noc.async_read(
                            bias_accessor,
                            CoreLocalMem<std::uint32_t>(bias_wr_ptr + bias_face_bytes),
                            bias_face_row_bytes,
                            {.page_id = b_page, .offset_bytes = bias_face_bytes},
                            {});
                        bias_wr_ptr += bias_tile_bytes;
                    }
                    noc.async_read_barrier();
                    cb_bias.push_back(tiles_in_block);
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
            const std::uint32_t pos_in_chunk = (tile_row - tile_row_start) % chunk_size_rows;
            const bool chunk_input_done = (pos_in_chunk + 1 == chunk_size_rows) || (tile_row + 1 == tile_row_end);
            if (chunk_input_done) {
                const std::uint32_t chunk_first = tile_row - pos_in_chunk;
                for (std::uint32_t r = chunk_first; r <= tile_row; r++) {
                    // Batched RoPE: fold the global row r into (batch b, within-batch seq row
                    // r_seq) so each input batch reuses cos/sin at seq row r_seq. b selects the
                    // per-batch cos/sin block via rope_batch_stride_tiles (0 => broadcast, all
                    // batches share block 0). At batch=1: r_seq==r, rope_batch_off==0 (unchanged).
                    const std::uint32_t r_seq = (rope_seqlen_tiles != 0) ? (r % rope_seqlen_tiles) : r;
                    const std::uint32_t rope_batch_off =
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
                    // but unreturned). Compute's cumulative wait_front absorbs the
                    // block-wise pushes (same pattern as the input read above).
                    const std::uint32_t rope_tiles_this_row = (per_head_rope != 0) ? num_tile_cols : head_dim_tiles;
                    for (std::uint32_t t0 = 0; t0 < rope_tiles_this_row; t0 += block_size) {
                        const std::uint32_t grp =
                            ((rope_tiles_this_row - t0) >= block_size) ? block_size : (rope_tiles_this_row - t0);
                        cb_rope_cos.reserve_back(grp);
                        cb_rope_sin.reserve_back(grp);
                        std::uint32_t rope_cos_wr_ptr = cb_rope_cos.get_write_ptr();
                        std::uint32_t rope_sin_wr_ptr = cb_rope_sin.get_write_ptr();
                        for (std::uint32_t j = 0; j < grp; j++) {
                            const std::uint32_t t = t0 + j;
                            std::uint32_t src_idx;
                            if constexpr (per_head_rope != 0) {
                                // tile t -> head (t / head_dim_tiles), within-head (t % head_dim_tiles)
                                const std::uint32_t h = t / head_dim_tiles;
                                const std::uint32_t within = t - h * head_dim_tiles;
                                src_idx = rope_batch_off + h * rope_seqlen_tiles * head_dim_tiles +
                                          r_seq * head_dim_tiles + within;
                            } else {
                                src_idx = rope_batch_off + r_seq * head_dim_tiles + t;
                            }
                            noc.async_read(
                                rope_cos_accessor,
                                CoreLocalMem<std::uint32_t>(rope_cos_wr_ptr),
                                rope_cos_page_bytes,
                                {.page_id = src_idx},
                                {});
                            noc.async_read(
                                rope_sin_accessor,
                                CoreLocalMem<std::uint32_t>(rope_sin_wr_ptr),
                                rope_sin_page_bytes,
                                {.page_id = src_idx},
                                {});
                            rope_cos_wr_ptr += rope_cos_tile_bytes;
                            rope_sin_wr_ptr += rope_sin_tile_bytes;
                        }
                        noc.async_read_barrier();
                        cb_rope_cos.push_back(grp);
                        cb_rope_sin.push_back(grp);
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
    if (wave_role == kWaveSignal) {
        noc.async_atomic_barrier();  // the partner's start_sem inc
    }
}
