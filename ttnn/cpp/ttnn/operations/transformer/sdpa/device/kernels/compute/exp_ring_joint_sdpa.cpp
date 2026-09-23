// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef SDPA_RECIPE_EXP_RING
// Named precision recipes B/C/D/E: the shared streaming recipe continuation (recipe_ring.hpp) replaces
// sdpa_ring_v2. It reuses the ring_joint recipe hooks (valid-row key tail masking), which are keyed on
// SDPA_RECIPE_RING. Kernel-size settings mirror ring_joint_sdpa.cpp's recipe block.
#define SDPA_RECIPE_RING 1
#define LLK_ZEROFLAG_OUTLINE 1
#if defined(WATCHER_ENABLED) || !defined(SDPA_RECIPE_FP32)
#pragma GCC optimize("Os")
#else
#pragma GCC optimize("O2")
#endif
#ifdef SDPA_RECIPE_LOFI
#include "streaming/lofi_scaling.hpp"
#endif
#endif

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include <tt-metalium/constants.hpp>
#include "compute_common.hpp"
#ifdef SDPA_RECIPE_EXP_RING
#include "streaming/recipe_tail.hpp"
#include "streaming/recipe_sfpu.hpp"
#include "streaming/recipe_streaming.hpp"
#include "streaming/recipe_ring.hpp"
#else
#include "compute_streaming.hpp"
#endif
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_fused_op_indexer.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(3);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(4);
    constexpr uint32_t logical_n_ct = get_compile_time_arg_val(5);
    constexpr uint32_t logical_nt_ct = get_compile_time_arg_val(6);
    constexpr uint32_t Lt = get_compile_time_arg_val(7);
    constexpr uint32_t L = get_compile_time_arg_val(8);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(10);
    constexpr uint32_t ring_size = get_compile_time_arg_val(11);

    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(12);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(14);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(15);

    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(16);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(18);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(19);
    // Streamed Q (host fallback when resident Q does not fit L1): cb_q_in holds one chunk; every
    // pass reads at offset 0 and pops its chunk at pass end so the reader can load the next one.
    constexpr bool stream_q = get_compile_time_arg_val(20) == 1;
    // Multi-pass programs keep per-pass accumulator state in the L1 FIFO; single-pass programs
    // use the original persist-in-scratch path (no FIFO entry/exit cost per ring iteration).
    constexpr bool use_state_fifo = get_compile_time_arg_val(21) == 1;
    // When set, logical_n_ct/logical_nt_ct are worst-case placeholders; live values arrive via the
    // reader's derived CB.
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t cb_derived = tt::CBIndex::c_13;

    // Lightweight mask: all mask tiles live in cb_mask_in (c_3).
    // Layout: [neginf(0)] [global_n_partial?(1)] [joint_l_partial?(1 or 2)]
    // Only needed when any K/joint dimension has padding that doesn't fill a chunk.
    constexpr bool local_n_has_padding = local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding =
        has_logical_n_tensor || (logical_n_ct % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0);
    constexpr bool joint_has_padding = L > 0 && L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool needs_lightweight_mask = local_n_has_padding || global_n_has_padding || joint_has_padding;

    // Tile presence must match the factory's CB sizing and the writer's generation order
    // (partial_tile_present is the one definition of that rule).
    constexpr bool has_global_n_partial_tile =
        ring_joint::partial_tile_present(global_n_partial_col, has_logical_n_tensor);
    constexpr uint32_t neginf_tile_idx = 0;
    constexpr uint32_t global_n_partial_tile_idx = has_global_n_partial_tile ? 1 : 0;
    constexpr uint32_t joint_l_partial_tile_idx =
        (joint_l_partial_col > 0) ? (1 + (has_global_n_partial_tile ? 1 : 0)) : 0;
    constexpr uint32_t total_mask_tiles = 1 + (has_global_n_partial_tile ? 1 : 0) + (joint_l_partial_col > 0 ? 1 : 0);

    uint32_t argidx = 0;
    // Head-serial passes: this core owns flat Q chunks q_base + p * q_stride for p in [0, q_count),
    // i.e. one chunk of head (p * grid_rows + my_row) per pass. See the program factory.
    const uint32_t q_base = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_stride = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_count = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpIndexer fused_op_indexer = RingSDPAOpIndexer(argidx);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_max_in = tt::CBIndex::c_6;  // deferred norm: running max
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_7;
    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_max_out = tt::CBIndex::c_17;  // deferred norm: running max

    // Streaming compute uses c_9 as 1-tile recip scratch for normalize_row_streaming.
    // (c_4 is used by cb_scale_in in ring joint SDPA, unlike regular SDPA.)
    constexpr uint32_t cb_recip_scratch = tt::CBIndex::c_9;

    // Deferred norm: sum save/restore CBs for multi Q-chunk DRAM round-trip.
    constexpr uint32_t cb_sum_out = tt::CBIndex::c_10;
    constexpr uint32_t cb_sum_in = tt::CBIndex::c_11;
    constexpr uint32_t cb_signal = tt::CBIndex::c_12;

    // div_up: the last local chunk may be partial (padded Q rows). Matches the factory's
    // num_q_chunks so flat-id decoding stays consistent across host and kernels.
    constexpr uint32_t num_q_chunks =
        (local_padded_Nt + Sq_chunk_t - 1) / Sq_chunk_t + (Lt + Sq_chunk_t - 1) / Sq_chunk_t;

    // Live length from the reader's derived CB, read ONCE so this kernel and the reader derive their
    // chunk-skip decisions from the same single DRAM read. Must precede compute_kernel_hw_startup:
    // read_tile_value rendezvouses UNPACK -> MATH/PACK through the mailboxes, and reading it after
    // hw startup / matmul_init returns garbage.
    uint32_t logical_n = logical_n_ct;
    uint32_t logical_nt = logical_nt_ct;
    uint32_t global_n_partial_col_live = global_n_partial_col;
    if constexpr (has_logical_n_tensor) {
        CircularBuffer cb_derived_obj(cb_derived);
        cb_derived_obj.wait_front(1);
        constexpr uint32_t kDerivedTile = 0;
        logical_nt = ckernel::read_tile_value(cb_derived, kDerivedTile, ring_joint::kDerivedLogicalNt);
        global_n_partial_col_live =
            ckernel::read_tile_value(cb_derived, kDerivedTile, ring_joint::kDerivedGlobalNPartialCol);
        cb_derived_obj.pop_front(1);
        // Recover the element count: exact inverse of the reader's (logical_nt, partial_col) derivation.
        logical_n = (logical_nt == 0)
                        ? 0u
                        : ((logical_nt - 1) * ring_joint::kTileHeight +
                           (global_n_partial_col_live == 0 ? ring_joint::kTileHeight : global_n_partial_col_live));
    }

#ifdef SDPA_RECIPE_EXP_RING
    {
        // Recipe CB layout (sdpa_recipe.cpp): Q=0, K=1, V=2, reduce scaler=3, column identity=4,
        // qk_im=6, state {sum,max,out} = {12,10,8}/{13,11,9}, out=16. The logical_n tensor is rejected on
        // the host, so logical_n/logical_nt are the compile-time values.
        constexpr uint32_t recipe_cb_q = 0, recipe_cb_k = 1, recipe_cb_qk_im = 6;
        constexpr uint32_t recipe_cb_identity_scale = 3, recipe_cb_col_identity = 4;
#ifdef SDPA_RECIPE_FP32
        constexpr uint32_t recipe_subblock_h = 1;
#else
        constexpr uint32_t recipe_subblock_h = 2;
#endif
        static_assert(Sk_chunk_t == 16 && DHt == 4, "Named exp ring recipes require K512/D128");
        compute_kernel_hw_startup<SrcOrder::Reverse>(recipe_cb_q, recipe_cb_k, recipe_cb_qk_im);
        matmul_init(recipe_cb_q, recipe_cb_k);
        init_sdpa_streaming_semaphores();
        CircularBuffer(recipe_cb_identity_scale).wait_front(1);
        CircularBuffer(recipe_cb_col_identity).wait_front(1);

        // One pass per core row (host-validated): a single Q chunk whose recurrent state stays resident
        // in L1 across every active ring iteration. Q is released and the state normalized only on the
        // last KV chunk of the last active iteration.
        ASSERT(q_count == 1);
        RecipeAccumulatorState resident = {{12, 10, 8}, {13, 11, 9}};
        const uint32_t last_active_ring_iter =
            find_last_active_ring_iter(fused_op_indexer.seq, local_padded_Nt, logical_nt, L);
        bool seen_active_iter = false;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
            const bool do_joint_kv = ring_id == ring_size - 1;
            const uint32_t num_kv_chunks =
                do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;
            // Same activity predicate as the reader/writer (and find_last_active_ring_iter).
            const bool ring_iter_processes_KV_chunks = ring_id * local_padded_Nt < logical_nt;
            if (!ring_iter_processes_KV_chunks && !(do_joint_kv && L != 0)) {
                continue;
            }
            // Valid key rows of this shard: local shard padding and the global logical_n tail. The
            // recipe skips chunks whose origin is at/after the valid rows -- exactly the reader's
            // kv_chunk_is_beyond_logical_n skip (chunk kc is sent iff kc*512 < primary_rows) -- and masks
            // the remaining partial chunk columns. Joint chunks are never skipped (joint_rows = L).
            const uint32_t n_origin = ring_id * local_padded_N;
            const uint32_t primary_rows = logical_n <= n_origin ? 0
                                          : logical_n - n_origin < local_padded_N ? logical_n - n_origin
                                                                                  : local_padded_N;
            const uint32_t joint_rows = do_joint_kv ? L : 0;
            // Consumes the reader's per-pass phase-alignment K/V pair (sent iff the chunk count is even).
            sdpa_recipe_ring_segment<Sq_chunk_t, scale_fp32, recipe_subblock_h>(
                resident,
                0,
                1,
                num_local_k_chunks,
                num_kv_chunks,
                primary_rows,
                joint_rows,
                !seen_active_iter,
                ring_iter == last_active_ring_iter);
            seen_active_iter = true;
        }
    }
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    matmul_init(cb_q_in, cb_k_in);

    CircularBuffer cb_identity_scale_in_obj(cb_identity_scale_in);
    CircularBuffer cb_mask_in_obj(cb_mask_in);

    // Wait once for identity scale; streaming v2 removes per-call waits inside reduce_c_row_group.
    cb_identity_scale_in_obj.wait_front(1);

    // Wait for all lightweight mask tiles once before the ring loop.
    // Writer generates them once and they stay permanently fronted.
    if constexpr (needs_lightweight_mask) {
        cb_mask_in_obj.wait_front(total_mask_tiles);
    }

    // Precompute padded tile counts that are constant across ring iterations
    constexpr uint32_t local_n_padded_tiles =
        (local_padded_Nt % Sk_chunk_t != 0) ? (Sk_chunk_t - (local_padded_Nt % Sk_chunk_t)) : 0;
    constexpr uint32_t joint_n_padded_tiles = (Lt % Sk_chunk_t != 0) ? (Sk_chunk_t - (Lt % Sk_chunk_t)) : 0;

    // Fixed scratch halves for the within-pass ping-pong. Unlike the single-head path these are
    // never rewritten: cross-ring-iteration state lives in the L1 state FIFO
    // ({cb_sum_in, cb_max_in, cb_prev_out} = {c_11, c_6, c_7}), one entry per pass, so every pass
    // starts from the same scratch roles. See sdpa_ring_v2's use_l1_state_fifo.
    RingAccumulatorState scratch_state = {
        {cb_sum_A, cb_max_A, cb_out_im_A},  // prev-scratch
        {cb_sum_B, cb_max_B, cb_out_im_B},  // cur-scratch
    };

    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_indexer.seq, local_padded_Nt, logical_nt, L);

    // First iteration that does work inits accumulators — not necessarily iter 0 (a fully-pad own
    // shard skips it).
    bool seen_active_iter = false;

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        // First, find out if this ring iter processes any KV chunks.
        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile < logical_nt;
        const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);

        if (!ring_iter_does_work) {
            continue;
        }

        const int32_t global_n_within_ring_iter = logical_n - ring_id * local_padded_N;
        // Note the > and <=. This means there is real length of logical_n within this ring iter.
        const bool global_n_is_within_ring_iter =
            global_n_within_ring_iter > 0 && global_n_within_ring_iter <= (int32_t)local_padded_N;
        const bool global_n_needs_masking = global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_global_n_mask = global_n_is_within_ring_iter && global_n_needs_masking;
        const uint32_t global_n_mask_chunk_id = global_n_within_ring_iter / (Sk_chunk_t * tt::constants::TILE_HEIGHT);

        // LOCAL N MASK
        const bool local_n_needs_masking = local_padded_Nt % Sk_chunk_t != 0;
        const uint32_t local_n_mask_chunk_id = local_padded_Nt / Sk_chunk_t;

        // JOINT L MASK
        const bool joint_n_needs_masking = L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;
        const uint32_t joint_n_mask_chunk_id = L / (Sk_chunk_t * tt::constants::TILE_HEIGHT);

        // Build lightweight mask context for this ring iteration
        LightweightMaskContext lw_mask;
        lw_mask.neginf_tile_idx = neginf_tile_idx;
        lw_mask.local_n_padded_tiles = local_n_padded_tiles;
        lw_mask.joint_n_padded_tiles = joint_n_padded_tiles;
        lw_mask.global_n_partial_col = global_n_partial_col_live;
        lw_mask.joint_l_partial_col = joint_l_partial_col;
        lw_mask.global_n_partial_tile_idx = global_n_partial_tile_idx;
        lw_mask.joint_l_partial_tile_idx = joint_l_partial_tile_idx;
        if (ring_iter_needs_global_n_mask) {
            const uint32_t unpadded_in_chunk = global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT);
            const uint32_t valid_tiles =
                (unpadded_in_chunk + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
            lw_mask.global_n_padded_tiles = Sk_chunk_t - valid_tiles;
        }

        const bool is_last_ring_iter = (ring_iter == last_active_ring_iter);
        const bool is_first_active_iter = !seen_active_iter;
        static_assert(use_streaming_compute, "Streaming compute must be enabled for ring joint SDPA");

        // Serial passes over this row's heads, same order as the reader and writer. Each pass is a
        // single-Q-chunk sdpa_ring_v2 call (q_per_core == 1), so every L1-residency property of the
        // single-head path holds per pass; the only per-pass state is which Q chunk it reads from
        // cb_q_in (q_base_tiles) and which L1 FIFO entry it merges (handled inside sdpa_ring_v2).
        for (uint32_t pass = 0; pass < q_count; ++pass) {
            const uint32_t global_q_chunk = q_base + pass * q_stride;
            sdpa_ring_v2<
                Sq_chunk_t,
                Sk_chunk_t,
                0,  // Skt — not used for ring
                DHt,
                DHt,  // vDHt = DHt for ring
                scale_fp32,
                qk_subblock_h,
                qk_subblock_w,
                out_subblock_h,  // qktv_subblock_h
                out_subblock_w,  // qktv_subblock_w
                cb_q_in,
                cb_k_in,
                cb_v_in,
                cb_qk_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                cb_mask_in,
                cb_scale_in,
                cb_max_in,
                cb_max_out,
                cb_prev_out,
                cb_out,
                cb_out,  // cb_normalized_out — output goes directly to cb_out
                cb_sum_out,
                cb_sum_in,
                cb_signal,
                needs_lightweight_mask,
                false,  // is_causal_sdpa
                false,  // is_balanced_sdpa
                false,  // chunked_enabled
                local_padded_Nt,
                local_padded_Nt,  // q_local_padded_Nt
                0,                // chunk_size_t
                global_n_has_padding,
                local_n_has_padding,
                joint_has_padding,
                false,            // straddle_mask_enabled
                use_state_fifo>(  // use_l1_state_fifo — single-pass programs keep the scratch path
                global_q_chunk,
                global_q_chunk + 1,
                num_kv_chunks,
                num_q_chunks,
                ring_iter,
                ring_id,
                num_local_k_chunks,
                logical_nt,
                ring_iter_needs_global_n_mask,
                ring_iter_needs_joint_n_mask,
                local_n_needs_masking,
                global_n_mask_chunk_id,
                local_n_mask_chunk_id,
                joint_n_mask_chunk_id,
                scratch_state,
                is_last_ring_iter,
                /*q_per_core=*/1,
                lw_mask,
                /*skip_first_half_q=*/false,
                /*use_zigzag_balancing=*/false,
                ChunkedContext{},
                is_first_active_iter,
                /*logical_lt=*/0,
                /*q_base_tiles=*/stream_q ? 0u : pass * q_chunk_tiles);

            if constexpr (stream_q) {
                // This pass's chunk is spent; free the slot so the reader can load the next pass's
                // chunk. Uniform on every core (q_count == 1 rows re-read too).
                sdpa_cb_pop_front_out_of_line(cb_q_in, q_chunk_tiles);
            }
        }

        if constexpr (!stream_q && use_state_fifo) {
            // All q_count Q chunks stay resident in cb_q_in for the whole op (each read once, on the
            // first active ring iteration) and are popped together once the last pass has consumed
            // them. On the scratch path (!use_state_fifo) sdpa_ring_v2 pops the single resident
            // chunk itself on the last ring iteration — popping here too would double-pop.
            if (is_last_ring_iter) {
                sdpa_cb_pop_front_out_of_line(cb_q_in, q_count * q_chunk_tiles);
            }
        }

        seen_active_iter = true;
    }
#endif  // SDPA_RECIPE_EXP_RING
}
