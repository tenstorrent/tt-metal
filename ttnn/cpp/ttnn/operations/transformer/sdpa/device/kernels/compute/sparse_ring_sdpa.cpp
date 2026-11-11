// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#define LLK_ZEROFLAG_OUTLINE 1

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/tile_constants.h"
#include "compute_common.hpp"
#include "sparse_ring_streaming.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"

void kernel_main() {
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
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
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");
    static_assert(use_streaming_compute, "the resident step runs on the streaming compute path (fp32_dest_acc_en off)");

    constexpr bool local_n_has_padding = kv_local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n_compile % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr uint32_t local_n_padded_tiles = local_n_has_padding ? Sk_chunk_t - (kv_local_padded_Nt % Sk_chunk_t) : 0;

    uint32_t argidx = 0;
    uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_size_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t forward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t backward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    argidx += 4;
    const uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t cb_arg_offset = 49;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_max_in = get_compile_time_arg_val(cb_arg_offset + 6);
    constexpr uint32_t cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(cb_arg_offset + 9);
    constexpr uint32_t cb_sum_in = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(cb_arg_offset + 15);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 18);
    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(cb_arg_offset + 22);

    constexpr uint32_t cb_sparse_plan = get_named_compile_time_arg_val("cb_sparse_plan");
    constexpr uint32_t cb_pooled_bias = get_named_compile_time_arg_val("cb_pooled_bias");
    constexpr uint32_t pooled_chunks = get_named_compile_time_arg_val("pooled_chunks");
    constexpr uint32_t pooled_last_tiles = get_named_compile_time_arg_val("pooled_last_tiles");
    constexpr bool pooled_route = get_named_compile_time_arg_val("pooled_route") == 1;
    constexpr uint32_t resident_slots = get_named_compile_time_arg_val("resident_slots");
    constexpr uint32_t plan_block_words = get_named_compile_time_arg_val("plan_block_words");
    constexpr uint32_t plan_rec_base = get_named_compile_time_arg_val("plan_rec_base");
    constexpr uint32_t num_q_passes = get_named_compile_time_arg_val("num_q_passes");
    constexpr bool pair_steps = get_named_compile_time_arg_val("pair_steps") == 1;
    constexpr uint32_t kv_cb_pages = get_named_compile_time_arg_val("kv_cb_pages");
    constexpr bool has_block_counts = get_named_compile_time_arg_val("has_block_counts") == 1;
    constexpr uint32_t cb_block_counts = get_named_compile_time_arg_val("cb_block_counts");
    constexpr uint32_t cb_key_bias = get_named_compile_time_arg_val("cb_key_bias");
    constexpr uint32_t key_bias_tiles = get_named_compile_time_arg_val("key_bias_tiles");
    constexpr uint32_t cb_sum_identity = get_named_compile_time_arg_val("cb_sum_identity");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    matmul_init(cb_q_in, cb_k_in);
    // PACK_DONE capacity 15: the resident step posts one token per S5 batch. Once per kernel: a reset between ring
    // iterations could drop a token UNPACK has not yet taken.
    init_sdpa_streaming_semaphores<15>();

    CircularBuffer(cb_identity_scale_in).wait_front(1);
    [[maybe_unused]] const volatile uint8_t* block_counts = nullptr;
    constexpr uint32_t dyn_mode = get_named_compile_time_arg_val("dyn_mode");
    if constexpr (has_block_counts || dyn_mode == 2) {
        CircularBuffer(cb_block_counts).wait_front(1);
        block_counts = reinterpret_cast<const volatile uint8_t*>(ckernel::get_tile_address(cb_block_counts, 0));
        invalidate_l1_cache();
    }
    if constexpr (dyn_mode == 2) {
        constexpr uint32_t dyn_range_offset = get_named_compile_time_arg_val("dyn_range_offset");
        const volatile uint32_t* range = reinterpret_cast<const volatile uint32_t*>(block_counts + dyn_range_offset);
        global_q_start = range[0];
        global_q_end = range[1];
    }

    const uint32_t q_per_core = global_q_end - global_q_start;
    uint32_t kv_pos = 0;
    uint32_t kb_pos = 0;
    for (uint32_t q_pass = 0; q_pass < num_q_passes; ++q_pass) {
        const uint32_t pass_q_start = global_q_start + (q_pass * q_per_core) / num_q_passes;
        const uint32_t pass_q_end = global_q_start + ((q_pass + 1) * q_per_core) / num_q_passes;
        RingSDPAOpIndexer fused_op_indexer(
            ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);
        uint32_t initialized_slots = 0;
        bool seen_active_iter = false;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
            if (((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
                continue;
            }
            const bool is_first_active_iter = !seen_active_iter;
            seen_active_iter = true;
            const bool is_last_ring_iter = is_last_active_ring_iter(active_ring_iter_mask, ring_iter);

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
            const bool local_n_needs_masking = local_n_has_padding;
            const uint32_t local_n_mask_chunk_id = kv_local_padded_Nt / Sk_chunk_t;

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
                vDHt,
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
                kv_cb_pages,
                has_block_counts,
                cb_key_bias,
                key_bias_tiles,
                cb_sum_identity>(
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
                kb_pos,
                is_last_ring_iter,
                is_first_active_iter,
                block_counts);
        }
    }
    if constexpr (has_block_counts || dyn_mode == 2) {
        CircularBuffer(cb_block_counts).pop_front(1);
    }
}
