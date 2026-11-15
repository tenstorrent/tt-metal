// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-mode compute (num_rows R > 1). The SDPA pipeline keeps one (8 heads x 1 row) running state in dst, so a core's
// rows (its row group's) run one after another: row j is the one-row SDPA of q tile j over its chunks, each read from
// L1 (a shared chunk stays in cb_k_shared for every row; row j's own chunks stream through cb_k_own, or all stay in
// cb_k_shared with one row per core), then row j's tail merge, sink and inverse RoPE with row j's table. S-block 0 of
// each row group finally tiles its heads' gathered [hpr * R, head_dim] rows into (R, 32) tiles.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/experimental/rope_sfpu.h"
#include "api/compute/experimental/sdpa.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/flash_mqa_decode_rows_common.hpp"

namespace {

// compute_sdpa_chunk (api/compute/experimental/sdpa.h) for one chunk whose K == V tiles start at tile k_tile_index of
// cb_k (already waited for; the caller pops), with the Q tile row at q_tile_index of cb_q. Fixed configuration:
// single-CB K == V, signal granularities 1, output granularity num_tiles_v, no pack init.
template <
    std::uint32_t chunk_size,
    std::uint32_t num_tiles_k,
    std::uint32_t num_tiles_v,
    std::uint32_t scale_fp32,
    std::uint32_t packed_tile_size>
void rows_sdpa_chunk(
    std::uint32_t cb_q,
    std::uint32_t q_tile_index,
    std::uint32_t cb_k,
    std::uint32_t k_tile_index,
    std::uint32_t cb_mask,
    std::uint32_t cb_out,
    std::uint32_t mm1_dst_offset,
    std::uint32_t mm2_dst_offset,
    std::uint32_t max_dst_offset,
    std::uint32_t sum_dst_offset,
    std::uint32_t corr_exp_dst_offset,
    bool first_chunk,
    bool last_chunk,
    bool mask_chunk) {
    constexpr bool transpose_k = true;
    constexpr bool transpose_v = false;
    constexpr bool exp_approx_mode = false;
    constexpr std::uint16_t scale_bf16 = scale_fp32 >> 16;
    constexpr std::uint32_t output_granularity = num_tiles_v;
    static_assert(chunk_size >= 2 && num_tiles_k % 2 == 0);
    static_assert(chunk_size + 1 <= 15 && num_tiles_v / output_granularity + 1 <= 15);
    PACK((ckernel::sfpu::_init_sdpa_reduce_max_row_8x32_replay_buffers_()));
    sdpa_custom_mm_block_init_short<transpose_k, false>(cb_q, cb_k, cb_out, chunk_size);
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
    sdpa_custom_mm_block<transpose_k, 1, false, num_tiles_k>(
        cb_q, cb_k, cb_mask, q_tile_index, k_tile_index, mm1_dst_offset, num_tiles_k, chunk_size, mask_chunk);

    PACK((llk_math_sfpu_sdpa_reduce_max_row<false, DST_ACCUM_MODE, DataFormat::Float16_b, chunk_size, false, 1>(
        mm1_dst_offset, max_dst_offset, !first_chunk)));
    sdpa_sub_bcast_col_srca_srcb_reuse_tiles_init<chunk_size>(cb_q);
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
    sdpa_bcast_col_srca_srcb_reuse_preamble(max_dst_offset);
    sdpa_sub_bcast_col_srca_srcb_reuse_tiles<chunk_size, false, 1>(mm1_dst_offset);
    if (!first_chunk) {
        PACK((non_approx_exp_mul_prev<exp_approx_mode, scale_bf16>(sum_dst_offset, corr_exp_dst_offset)));
        PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
#ifdef TRISC_MATH
        constexpr bool skip_addrmod = !is_high_fidelity(MATH_FIDELITY);
#else
        constexpr bool skip_addrmod = false;
#endif
        sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v, skip_addrmod>(cb_q);
        MATH((t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)));
        sdpa_bcast_col_srca_srcb_reuse_preamble(corr_exp_dst_offset);
        sdpa_mul_bcast_col_srca_srcb_reuse_tiles<num_tiles_v, true, 1>(mm2_dst_offset);
        MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
        MATH((t6_semaphore_get<p_stall::NONE>(SFPU_FPU)));
    } else {
        MATH((t6_semaphore_post<p_stall::NONE>(semaphore::FPU_SFPU)));
    }
    for (std::uint32_t i = 0; i < chunk_size; i++) {
        PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
        PACK((fast_approx_exp(mm1_dst_offset + i * packed_tile_size)));
        PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
        PACK((t6_semaphore_post<p_stall::NONE>(SFPU_FPU)));
    }

    sdpa_custom_mm_reuse_dest_srcb_block_init_short(cb_q, cb_k, cb_out, transpose_v, chunk_size, num_tiles_v);
    sdpa_custom_mm_reuse_dest_srcb_block<output_granularity, 1>(
        cb_q,
        cb_k,
        0,
        k_tile_index,
        mm1_dst_offset,
        mm2_dst_offset,
        transpose_v,
        chunk_size,
        num_tiles_v,
        num_tiles_k,
        last_chunk);

    PACK((ckernel::sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_()));
    PACK((llk_math_sfpu_sdpa_reduce_sum_row<false, DST_ACCUM_MODE, DataFormat::Float16_b, chunk_size, true>(
        mm1_dst_offset, sum_dst_offset, !first_chunk)));
    PACK((llk_math_sdpa_sfpu_signal_chunk_done()));
}

}  // namespace

void kernel_main() {
    constexpr uint32_t DHt = get_named_compile_time_arg_val("DHt");
    constexpr uint32_t vDHt = get_named_compile_time_arg_val("vDHt");
    constexpr uint32_t Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t");
    constexpr uint32_t k_chunk_size = get_named_compile_time_arg_val("k_chunk_size");
    constexpr uint32_t num_s_blocks = get_named_compile_time_arg_val("num_s_blocks");
    constexpr uint32_t num_tree_steps = get_named_compile_time_arg_val("num_tree_steps");
    constexpr uint32_t heads_per_core = get_named_compile_time_arg_val("heads_per_core");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr uint32_t row_groups = get_named_compile_time_arg_val("row_groups");
    // One row per core: the reader keeps every chunk of this core in cb_k_shared (see the reader).
    constexpr bool share_reads = get_named_compile_time_arg_val("share_reads") == 1;
    constexpr uint32_t group_rows = num_rows / row_groups;
    constexpr bool dense = get_named_compile_time_arg_val("dense_selected") == 1;
    constexpr uint32_t ring_capacity = get_named_compile_time_arg_val("ring_capacity");
    constexpr uint32_t selected_capacity = get_named_compile_time_arg_val("selected_capacity");
    constexpr uint32_t scale_fp32 = get_named_compile_time_arg_val("scale_fp32");
    constexpr uint32_t dst_size = get_named_compile_time_arg_val("dst_size");
    constexpr uint32_t rope_tiles = get_named_compile_time_arg_val("rope_tiles");
    constexpr uint32_t cb_q_in = get_named_compile_time_arg_val("cb_q_in");
    constexpr uint32_t cb_q_rm = get_named_compile_time_arg_val("cb_q_rm");
    constexpr uint32_t cb_k_shared = get_named_compile_time_arg_val("cb_k_shared");
    constexpr uint32_t cb_k_own = get_named_compile_time_arg_val("cb_k_own");
    constexpr uint32_t cb_mask = get_named_compile_time_arg_val("cb_mask");
    constexpr uint32_t cb_ms_in = get_named_compile_time_arg_val("cb_ms_in");
    constexpr uint32_t cb_out_in = get_named_compile_time_arg_val("cb_out_in");
    constexpr uint32_t cb_out_o = get_named_compile_time_arg_val("cb_out_o");
    constexpr uint32_t cb_out_ms = get_named_compile_time_arg_val("cb_out_ms");
    constexpr uint32_t cb_interm_out = get_named_compile_time_arg_val("cb_interm_out");
    constexpr uint32_t cb_interm_ms = get_named_compile_time_arg_val("cb_interm_ms");
    constexpr uint32_t cb_out_rm = get_named_compile_time_arg_val("cb_out_rm");
    constexpr uint32_t cb_lengths = get_named_compile_time_arg_val("cb_lengths");
    constexpr uint32_t cb_cs = get_named_compile_time_arg_val("cb_cs");
    constexpr uint32_t cb_tile_rm = get_named_compile_time_arg_val("cb_tile_rm");
    constexpr uint32_t cb_out_tiles = get_named_compile_time_arg_val("cb_out_tiles");

    const uint32_t s_block = get_arg_val<uint32_t>(0);
    const uint32_t row_group = get_arg_val<uint32_t>(1);

    constexpr uint32_t q_chunk_tiles = DHt;
    constexpr uint32_t out_chunk_tiles = vDHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    static_assert(vDHt % dst_size == 0, "vDHt must be divisible by dst_size");
    constexpr uint32_t num_blocks = vDHt / dst_size;
    constexpr uint32_t block_size = vDHt / num_blocks;

    deepseek_compute_kernel_init();

    cb_wait_front(cb_lengths, 1);
    const uint32_t p = read_tile_value(cb_lengths, 0, 0);
    uint32_t n[num_rows];
    for (uint32_t j = 0; j < num_rows; ++j) {
        n[j] = read_tile_value(cb_lengths, 0, 1 + j);
    }
    cb_pop_front(cb_lengths, 1);
    const flash_mqa_decode_rows::Plan plan = flash_mqa_decode_rows::make_plan(
        num_rows, ring_capacity, k_chunk_size, selected_capacity, dense, p, n, row_group, row_groups);
    uint32_t max_len = 0;
    for (uint32_t j = row_group; j < num_rows; j += row_groups) {
        const uint32_t len = flash_mqa_decode_rows::row_len(plan, j);
        max_len = len > max_len ? len : max_len;
    }
    if (s_block >= max_len) {
        return;
    }

    compute_kernel_lib::tilize<DHt, cb_q_rm, cb_q_in>(group_rows);
    cb_wait_front(cb_q_in, group_rows * q_chunk_tiles);

    constexpr bool exp_approx_mode = false;
    constexpr uint32_t packed_tile_size = 8 * 2;
    constexpr uint32_t mm2_dst_offset = 0;
    constexpr uint32_t mm2_dst_tile_offset = mm2_dst_offset / packed_tile_size;
    constexpr uint32_t max_dst_offset = mm2_dst_offset + packed_tile_size * vDHt;
    constexpr uint32_t max_dst_tile_offset = max_dst_offset / packed_tile_size;
    constexpr uint32_t sum_dst_offset = max_dst_offset + 2;
    constexpr uint32_t corr_exp_dst_offset = max_dst_offset + packed_tile_size;
    constexpr uint32_t mm1_dst_offset = corr_exp_dst_offset + packed_tile_size;

    for (uint32_t j = row_group, jl = 0; j < num_rows; j += row_groups, ++jl) {
        const uint32_t len = flash_mqa_decode_rows::row_len(plan, j);
        const uint32_t active = len < num_s_blocks ? len : num_s_blocks;
        if (s_block >= active) {
            continue;
        }
        const uint32_t block_chunks = flash_mqa_decode::num_block_chunks(s_block, len, num_s_blocks);
        const uint32_t num_rounds =
            flash_mqa_decode::num_received(s_block, active, num_tree_steps) + (s_block == 0 ? 1 : 0);

        reconfig_full_operand(cb_k_shared, cb_q_in);
        pack_reconfig_data_format<true>(cb_out_o);
        PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
        PACK(SFPU_UNARY_INIT_FN(
            exponential,
            sfpu::exp_init,
            (true /*APPROXIMATION_MODE*/, scale_fp32, true /*CLAMP_NEGATIVE*/, DST_ACCUM_MODE)));
        sdpa_custom_mm_block_init_pack_short();

        const uint32_t sdpa_output_cb = num_rounds == 0 ? cb_out_o : cb_interm_out;
        const uint32_t sdpa_ms_cb = num_rounds == 0 ? cb_out_ms : cb_interm_ms;
        pack_block_contiguous_init(sdpa_output_cb);
        cb_reserve_back(sdpa_output_cb, out_chunk_tiles);
        cb_reserve_back(sdpa_ms_cb, 1);
        {
            tile_regs_acquire();
            for (uint32_t c = 0; c < block_chunks; ++c) {
                const uint32_t g = s_block + c * num_s_blocks;
                const bool mask_chunk = flash_mqa_decode_rows::valid_rows(plan, j, g) < k_chunk_size;
                const bool shared = share_reads || g < plan.shared_len;
                const uint32_t cb_k = shared ? cb_k_shared : cb_k_own;
                if (shared) {
                    cb_wait_front(cb_k_shared, (c + 1) * k_chunk_tiles);
                } else {
                    cb_wait_front(cb_k_own, k_chunk_tiles);
                }
                if (mask_chunk) {
                    cb_wait_front(cb_mask, 1);
                }
                rows_sdpa_chunk<Sk_chunk_t, q_chunk_tiles, out_chunk_tiles, scale_fp32, packed_tile_size>(
                    cb_q_in,
                    jl * q_chunk_tiles,
                    cb_k,
                    shared ? c * k_chunk_tiles : 0,
                    cb_mask,
                    sdpa_output_cb,
                    mm1_dst_offset,
                    mm2_dst_offset,
                    max_dst_offset,
                    sum_dst_offset,
                    corr_exp_dst_offset,
                    c == 0,
                    c == block_chunks - 1,
                    mask_chunk);
                if (mask_chunk) {
                    cb_pop_front(cb_mask, 1);
                }
                if (!shared) {
                    cb_pop_front(cb_k_own, k_chunk_tiles);
                }
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_block_contiguous(max_dst_tile_offset, sdpa_ms_cb, 1);
            cb_push_back(sdpa_ms_cb, 1);
            PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
            pack_block_contiguous(mm2_dst_tile_offset, sdpa_output_cb, out_chunk_tiles);
            PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
            cb_push_back(sdpa_output_cb, out_chunk_tiles);
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
            sdpa_custom_mm_block_uninit();
            MATH(t6_semaphore_wait_on_max<p_stall::STALL_SFPU>(semaphore::FPU_SFPU));
        }

        if (num_rounds == 0) {
            continue;
        }
        reconfig_full_operand_srca(cb_ms_in);
        exp_tile_init<exp_approx_mode, scale_fp32>();
        for (uint32_t r = 0; r + 1 < num_rounds; ++r) {
            sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                cb_ms_in, cb_interm_ms, cb_interm_ms, cb_out_in, cb_interm_out, cb_interm_out);
        }
        if (s_block == 0) {
            constexpr uint32_t rope_wt = rope_tiles;
            constexpr uint32_t rope_tail_block = 4;
            constexpr uint32_t rope_tail_blocks = vDHt / rope_tail_block;
            static_assert(vDHt % rope_tail_block == 0 && rope_wt <= rope_tail_block);
            static_assert(rope_tail_block + rope_wt <= dst_size);

            sdpa_tail_ms_reduce<exp_approx_mode, true, rope_tail_block, scale_fp32, VectorMode::C, true, false>(
                cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in);
            pack_untilize_dest_init<rope_tail_block, vDHt, false, TILE_C_DIM, false, false>(cb_out_rm);
            cb_reserve_back(cb_out_rm, vDHt);
            for (uint32_t b = 0; b < rope_tail_blocks; ++b) {
                if (b > 0) {
                    tile_regs_acquire();
                }
                cb_wait_front(cb_interm_out, rope_tail_block);
                cb_wait_front(cb_out_in, rope_tail_block);
                sdpa_mul_bcast_col_reuse_tiles<rope_tail_block>(cb_interm_out, cb_out_in, 0, 0);
                cb_pop_front(cb_interm_out, rope_tail_block);
                cb_pop_front(cb_out_in, rope_tail_block);
                if (b + 1 == rope_tail_blocks) {
                    sdpa_bcast_col_reuse_postamble();
                    cb_wait_front(cb_cs, rope_wt);
                    copy_init(cb_cs);
                    for (uint32_t t = 0; t < rope_wt; ++t) {
                        copy_tile(cb_cs, t, rope_tail_block + t);
                    }
                    cb_pop_front(cb_cs, rope_wt);
                    rope_sfpu_fused_init();
                    rope_sfpu_init();
                    rope_sfpu_inplace_fused_rows<
                        1,
                        rope_wt,
                        (rope_tail_block - rope_wt) * ROPE_SFPU_TILE_ROWS,
                        ROPE_SFPU_TILE_ROWS,
                        rope_tail_block * ROPE_SFPU_TILE_ROWS,
                        ROPE_SFPU_TILE_ROWS,
                        false,
                        heads_per_core,
                        false>(0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<rope_tail_block, vDHt, false, false, TILE_C_DIM, 0, false>(cb_out_rm, 1, b);
                tile_regs_release();
            }
            cb_push_back(cb_out_rm, vDHt);
            pack_untilize_uninit(cb_out_rm);
        } else {
            sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_o);
        }
    }

    const uint32_t shared_chunks =
        flash_mqa_decode::num_block_chunks(s_block, share_reads ? max_len : plan.shared_len, num_s_blocks);
    cb_wait_front(cb_k_shared, shared_chunks * k_chunk_tiles);
    cb_pop_front(cb_k_shared, shared_chunks * k_chunk_tiles);
    cb_pop_front(cb_q_in, group_rows * q_chunk_tiles);

    if (s_block == 0) {
        // This root's [hpr * R, head_dim] rows (row h_local * R + j) -> (R, 32) tiles, head-major (output order).
        compute_kernel_lib::tilize<vDHt, cb_tile_rm, cb_out_tiles>(heads_per_core / row_groups);
    }
}
