// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

// This kernel reconfigs ~30x; inlining the LLK Src zero-flag DEFAULT configurator at each site pushes
// the program over the kernel-config buffer. Force it out-of-line here.
#define LLK_ZEROFLAG_OUTLINE 1

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/tile_constants.h"
#include "compute_common.hpp"
#include "sparse_ring_streaming.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"

// Sparse ring attention compute: every core keeps the Q chunks of its current pass and their flash state in L1, and
// applies each K/V piece the reader streams to every resident chunk that keeps a block of it (sdpa_ring_resident,
// once per ring iteration). The core's chunks are split evenly into num_q_passes passes; each pass runs the ring
// order from the start.
void kernel_main() {
    // Positional compile-time args (shared layout with the reader and writer; unused slots are not read).
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    // Mask block size in tiles; the Q block is the q chunk (one mask row each).
    constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
    static_assert(Sq_chunk_t == block_tiles, "one Q block (q chunk) per mask row");
    static_assert(Sk_chunk_t % block_tiles == 0, "a K/V piece holds whole blocks");
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(5);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t logical_n_compile = get_compile_time_arg_val(7);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t ring_size = get_compile_time_arg_val(15);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(26);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(27) == 1;
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(28);
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(34);
    // One KV region per ring rank (no split KV).
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");
    static_assert(use_streaming_compute, "the resident step runs on the streaming compute path (fp32_dest_acc_en off)");

    // Padding of the last K chunk of a shard (local padded tail) and of the shard holding logical_n.
    constexpr bool local_n_has_padding = kv_local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n_compile % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr uint32_t local_n_padded_tiles = local_n_has_padding ? Sk_chunk_t - (kv_local_padded_Nt % Sk_chunk_t) : 0;

    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_size_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t forward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t backward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    argidx += 4;  // KV-pad rotation words (unused: no KV cache)
    const uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);

    // CB indices: the block starts at positional slot 49.
    constexpr uint32_t cb_arg_offset = 49;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_max_in = get_compile_time_arg_val(cb_arg_offset + 6);    // slot state: running max
    constexpr uint32_t cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 7);  // slot state: output accumulator
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_recip_scratch =
        get_compile_time_arg_val(cb_arg_offset + 9);  // 1-tile scratch for normalize_row_streaming
    constexpr uint32_t cb_sum_in = get_compile_time_arg_val(cb_arg_offset + 11);  // slot state: running sum
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(cb_arg_offset + 15);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 18);  // new-max scratch of the step
    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(cb_arg_offset + 22);

    // Sparse-specific named args.
    constexpr uint32_t cb_sparse_plan = get_named_compile_time_arg_val("cb_sparse_plan");
    constexpr uint32_t cb_pooled_bias = get_named_compile_time_arg_val("cb_pooled_bias");
    constexpr uint32_t pooled_chunks = get_named_compile_time_arg_val("pooled_chunks");
    constexpr uint32_t pooled_last_tiles = get_named_compile_time_arg_val("pooled_last_tiles");
    constexpr bool pooled_route = get_named_compile_time_arg_val("pooled_route") == 1;
    constexpr uint32_t resident_slots = get_named_compile_time_arg_val("resident_slots");
    constexpr uint32_t plan_block_words = get_named_compile_time_arg_val("plan_block_words");
    // Plan layout: one plan page per ring iteration (record base word), or 0 for a page per resident chunk. The
    // reader takes the same arg.
    constexpr uint32_t plan_rec_base = get_named_compile_time_arg_val("plan_rec_base");
    constexpr uint32_t num_q_passes = get_named_compile_time_arg_val("num_q_passes");
    // Pair steps: one step over two consecutive pieces of a head (halves the per-step fixed cost); needs the K/V CB
    // depth to keep a pair from wrapping the CB end.
    constexpr bool pair_steps = get_named_compile_time_arg_val("pair_steps") == 1;
    constexpr uint32_t kv_cb_pages = get_named_compile_time_arg_val("kv_cb_pages");

    SPARSE_TOTALS_INIT()  // DEBUG (dev branch): whole-op totals (sparse_prof_window.hpp)
    SPARSE_TOTAL(t_kernel, "KERNEL")
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    matmul_init(cb_q_in, cb_k_in);

    // The identity scale stays fronted for the whole op.
    CircularBuffer(cb_identity_scale_in).wait_front(1);

    const uint32_t q_per_core = global_q_end - global_q_start;
    // The K/V CBs' read position in entries, carried across ring iterations and passes (the CBs are never reset).
    uint32_t kv_pos = 0;
    for (uint32_t q_pass = 0; q_pass < num_q_passes; ++q_pass) {
        const uint32_t pass_q_start = global_q_start + (q_pass * q_per_core) / num_q_passes;
        const uint32_t pass_q_end = global_q_start + ((q_pass + 1) * q_per_core) / num_q_passes;
        RingSDPAOpIndexer fused_op_indexer(
            ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);
        // Which resident chunks have state in their slot, carried across the pass's ring iterations.
        uint32_t initialized_slots = 0;
        // The pass's first active iteration starts with fresh slots.
        bool seen_active_iter = false;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
            // Host precomputes which ring iterations have useful work; the ring-id sequencing above still advances so
            // compute stays aligned with the reader, the writer and the all-gather.
            if (((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
                continue;
            }
            sparse_prof::begin_iter(q_pass, ring_iter);  // DEBUG: profile window (sparse_prof_window.hpp)
            const bool is_first_active_iter = !seen_active_iter;
            seen_active_iter = true;
            const bool is_last_ring_iter = is_last_active_ring_iter(active_ring_iter_mask, ring_iter);

            // Global logical_n tail: the shard-relative K chunk holding the last real tile and the fully padded tiles
            // after it, on the shard where logical_n lands. Anchored on the last real tile, so a chunk-final sub-tile
            // partial column also counts as a masked chunk.
            const int32_t global_nt_within_ring_iter =
                static_cast<int32_t>(logical_nt) - static_cast<int32_t>(ring_id * kv_local_padded_Nt);
            const bool global_n_is_within_ring_iter =
                global_nt_within_ring_iter > 0 && global_nt_within_ring_iter <= (int32_t)kv_local_padded_Nt;
            const uint32_t global_n_mask_chunk_id =
                global_n_is_within_ring_iter ? (uint32_t)(global_nt_within_ring_iter - 1) / Sk_chunk_t : 0;
            const uint32_t global_n_valid_tiles_in_chunk =
                global_n_is_within_ring_iter
                    ? (uint32_t)global_nt_within_ring_iter - global_n_mask_chunk_id * Sk_chunk_t
                    : 0;
            const uint32_t global_n_padded_tiles_iter =
                global_n_is_within_ring_iter ? Sk_chunk_t - global_n_valid_tiles_in_chunk : 0;
            const bool ring_iter_needs_global_n_mask =
                global_n_is_within_ring_iter && (global_n_padded_tiles_iter > 0 || global_n_partial_col > 0);
            const uint32_t global_n_padded_tiles = ring_iter_needs_global_n_mask ? global_n_padded_tiles_iter : 0;
            // Local padded tail of the shard.
            const bool local_n_needs_masking = local_n_has_padding;
            const uint32_t local_n_mask_chunk_id = kv_local_padded_Nt / Sk_chunk_t;

            SPARSE_TOTAL(t_iter, "C_ITER")  // DEBUG (dev branch): whole-op totals
            sdpa_ring_resident<
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                vDHt,
                NH,
                scale_fp32,
                qk_subblock_w,
                cb_q_in,
                cb_k_in,
                cb_v_in,
                cb_qk_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_max_A,
                cb_col_identity,
                cb_recip_scratch,
                cb_max_in,
                cb_prev_out,
                cb_out,
                cb_sum_in,
                kv_local_padded_Nt,
                kv_region_Nt,
                chunk_size_t,
                vDHt,  // V CB row width in tiles (V is a separate tensor)
                global_n_has_padding,
                local_n_has_padding,
                local_n_padded_tiles,
                cb_sparse_plan,
                cb_pooled_bias,
                pooled_chunks,
                pooled_last_tiles,
                pooled_route,
                resident_slots,
                plan_block_words,
                plan_rec_base,
                num_q_passes,
                pair_steps,
                kv_cb_pages>(
                pass_q_start,
                pass_q_end,
                num_q_chunks,
                ring_id,
                num_local_k_chunks,
                logical_nt,
                ring_iter_needs_global_n_mask,
                local_n_needs_masking,
                global_n_mask_chunk_id,
                local_n_mask_chunk_id,
                global_n_padded_tiles,
                initialized_slots,
                kv_pos,
                is_last_ring_iter,
                is_first_active_iter);
        }
    }
    SPARSE_TOTAL_END(t_kernel)  // DEBUG (dev branch)
    SPARSE_TOTALS_FLUSH()
}
