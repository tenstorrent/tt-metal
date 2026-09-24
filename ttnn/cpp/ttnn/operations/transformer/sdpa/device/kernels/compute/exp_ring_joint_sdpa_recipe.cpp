// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exp ring joint SDPA compute for the named precision recipes B/C/D/E (FAST keeps exp_ring_joint_sdpa.cpp).
// The shared streaming recipe continuation (recipe_ring.hpp) keeps one recurrent state per pass across the
// ring. Compile-time arguments share the exp-ring layout (ExpRingJointSDPARecipeProgramFactory).

#include <cstdint>

// The recipe headers key their ring hooks (valid-row key tail masking) on SDPA_RECIPE_RING. -Os only for
// Watcher or the paired recipes' odd-chunk builds (SDPA_RECIPE_SIZE_OPTIMIZED, as in the dense kernel); O2
// otherwise. -Os on the even-chunk BF16 recipes cost ~35% of trace wall. Do not drop to the default compute
// level: the runtime key-tail mask hook (recipe_tail.hpp) then corrupts masked chunks (~55% L2 on every
// sub-K512 tail case).
#define SDPA_RECIPE_RING 1
#define LLK_ZEROFLAG_OUTLINE 1
// Size-limited builds (odd Q chunks) size-optimize only the pack thread: exp-ring Q224
// single-pass B 1.08 -> 0.87 ms, E_bf16 1.06 -> 0.77 ms on 1x2.
#if defined(WATCHER_ENABLED) || (defined(SDPA_RECIPE_SIZE_OPTIMIZED) && defined(TRISC_PACK))
#pragma GCC optimize("Os")
#else
#pragma GCC optimize("O2")
#endif
#ifdef SDPA_RECIPE_LOFI
#include "streaming/lofi_scaling.hpp"
#endif

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include <tt-metalium/constants.hpp>
#include "compute_common.hpp"
#include "streaming/recipe_tail.hpp"
#include "streaming/recipe_sfpu.hpp"
#include "streaming/recipe_streaming.hpp"
#include "streaming/recipe_ring.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_fused_op_indexer.hpp"

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(3);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(4);
    // The logical_n tensor is rejected on the host, so logical_n/logical_nt are the compile-time values.
    constexpr uint32_t logical_n = get_compile_time_arg_val(5);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(6);
    constexpr uint32_t L = get_compile_time_arg_val(8);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(10);
    constexpr uint32_t ring_size = get_compile_time_arg_val(11);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(16);
    static_assert(get_compile_time_arg_val(22) == 0, "Named exp ring recipes reject a logical_n tensor");

    uint32_t argidx = 0;
    // Head-serial passes: this core owns flat Q chunks q_base + p * q_stride for p in [0, q_count). The
    // recipe reads each pass's Q chunk from offset 0 of its single-slot Q CB, so only the count is used.
    [[maybe_unused]] const uint32_t q_base = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const uint32_t q_stride = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_count = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpIndexer fused_op_indexer = RingSDPAOpIndexer(argidx);

    // Recipe CB layout (sdpa_recipe.cpp): Q=0, K=1, V=2, reduce scaler=3, column identity=4,
    // qk_im=6, state {sum,max,out} = {12,10,8}/{13,11,9}, out=16.
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

    // Pass-outer, ring-inner (matching the reader and writer): each pass owns one Q chunk whose
    // recurrent state stays resident in L1 across every active ring iteration. Q is released and the
    // state normalized only on the last KV chunk of the pass's last active ring iteration, so the
    // next pass starts from fresh state and the single-slot Q CB.
    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_indexer.seq, local_padded_Nt, logical_nt, L);
    for (uint32_t pass = 0; pass < q_count; ++pass) {
        RecipeAccumulatorState resident = {{12, 10, 8}, {13, 11, 9}};
        RingIdSequencer pass_seq = fused_op_indexer.seq;
        bool seen_active_iter = false;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = pass_seq.get_next_ring_id([](uint32_t, uint32_t) {});
            const bool do_joint_kv = ring_id == ring_size - 1;
            const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;
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
            const uint32_t primary_rows = logical_n <= n_origin                   ? 0
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
}
