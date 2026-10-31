// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/experimental/sdpa.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tilize.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/flash_mqa_decode_common.hpp"

void kernel_main() {
    constexpr uint32_t DHt = get_named_compile_time_arg_val("DHt");
    constexpr uint32_t vDHt = get_named_compile_time_arg_val("vDHt");
    constexpr uint32_t Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t");
    constexpr uint32_t k_chunk_size = get_named_compile_time_arg_val("k_chunk_size");
    constexpr uint32_t num_s_blocks = get_named_compile_time_arg_val("num_s_blocks");
    constexpr uint32_t num_tree_steps = get_named_compile_time_arg_val("num_tree_steps");
    constexpr uint32_t scale_fp32 = get_named_compile_time_arg_val("scale_fp32");
    constexpr uint32_t dst_size = get_named_compile_time_arg_val("dst_size");
    constexpr uint32_t cb_q_in = get_named_compile_time_arg_val("cb_q_in");
    constexpr uint32_t cb_q_rm = get_named_compile_time_arg_val("cb_q_rm");
    constexpr uint32_t cb_k_in = get_named_compile_time_arg_val("cb_k_in");
    constexpr uint32_t cb_mask = get_named_compile_time_arg_val("cb_mask");
    constexpr uint32_t cb_ms_in = get_named_compile_time_arg_val("cb_ms_in");
    constexpr uint32_t cb_out_in = get_named_compile_time_arg_val("cb_out_in");
    constexpr uint32_t cb_out_o = get_named_compile_time_arg_val("cb_out_o");
    constexpr uint32_t cb_out_ms = get_named_compile_time_arg_val("cb_out_ms");
    constexpr uint32_t cb_interm_out = get_named_compile_time_arg_val("cb_interm_out");
    constexpr uint32_t cb_interm_ms = get_named_compile_time_arg_val("cb_interm_ms");
    constexpr uint32_t cb_out_rm = get_named_compile_time_arg_val("cb_out_rm");
    constexpr uint32_t cb_lengths = get_named_compile_time_arg_val("cb_lengths");

    const uint32_t s_block = get_arg_val<uint32_t>(0);

    constexpr uint32_t q_chunk_tiles = DHt;
    constexpr uint32_t out_chunk_tiles = vDHt;
    static_assert(out_chunk_tiles % 2 == 0, "out_chunk_tiles must be even");
    static_assert(vDHt % dst_size == 0, "vDHt must be divisible by dst_size");
    constexpr uint32_t num_blocks = vDHt / dst_size;
    constexpr uint32_t block_size = vDHt / num_blocks;

    deepseek_compute_kernel_init();

    cb_wait_front(cb_lengths, 1);
    const flash_mqa_decode::Lengths lengths{read_tile_value(cb_lengths, 0, 0), read_tile_value(cb_lengths, 0, 1)};
    cb_pop_front(cb_lengths, 1);
    const uint32_t total_chunks = flash_mqa_decode::num_chunks(lengths, k_chunk_size);
    const uint32_t block_chunks = flash_mqa_decode::num_block_chunks(s_block, total_chunks, num_s_blocks);
    if (block_chunks == 0) {
        return;
    }
    const uint32_t num_active = total_chunks < num_s_blocks ? total_chunks : num_s_blocks;
    // S-block 0 also merges the sink peer.
    const uint32_t num_rounds =
        flash_mqa_decode::num_received(s_block, num_active, num_tree_steps) + (s_block == 0 ? 1 : 0);

    compute_kernel_lib::tilize<DHt, cb_q_rm, cb_q_in>(1);

    reconfig_full_operand(cb_k_in, cb_q_in);
    pack_reconfig_data_format<true>(cb_out_o);
    PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
    PACK(SFPU_UNARY_INIT_FN(
        exponential,
        sfpu::exp_init,
        (true /*APPROXIMATION_MODE*/, scale_fp32, true /*CLAMP_NEGATIVE*/, DST_ACCUM_MODE)));
    sdpa_custom_mm_block_init_pack_short();

    constexpr bool transpose_k = true;
    constexpr bool transpose_v = false;
    constexpr bool exp_approx_mode = false;
    constexpr uint32_t packed_tile_size = 8 * 2;
    constexpr uint32_t mm2_dst_offset = 0;
    constexpr uint32_t mm2_dst_tile_offset = mm2_dst_offset / packed_tile_size;
    constexpr uint32_t max_dst_offset = mm2_dst_offset + packed_tile_size * vDHt;
    constexpr uint32_t max_dst_tile_offset = max_dst_offset / packed_tile_size;
    constexpr uint32_t sum_dst_offset = max_dst_offset + 2;
    constexpr uint32_t corr_exp_dst_offset = max_dst_offset + packed_tile_size;
    constexpr uint32_t mm1_dst_offset = corr_exp_dst_offset + packed_tile_size;
    constexpr uint32_t output_granularity = out_chunk_tiles;

    // A leaf sends its partial straight from cb_out_o / cb_out_ms; everyone else merges into interm.
    const uint32_t sdpa_output_cb = num_rounds == 0 ? cb_out_o : cb_interm_out;
    const uint32_t sdpa_ms_cb = num_rounds == 0 ? cb_out_ms : cb_interm_ms;

    pack_block_contiguous_init(sdpa_output_cb);
    cb_wait_front(cb_q_in, q_chunk_tiles);
    cb_reserve_back(sdpa_output_cb, out_chunk_tiles);
    cb_reserve_back(sdpa_ms_cb, 1);
    tile_regs_acquire();
    for (uint32_t j = 0; j < block_chunks; ++j) {
        const flash_mqa_decode::Chunk chunk =
            flash_mqa_decode::chunk_at(s_block + j * num_s_blocks, lengths, k_chunk_size);
        const bool mask_chunk = chunk.valid_rows < k_chunk_size;
        if (mask_chunk) {
            cb_wait_front(cb_mask, 1);
        }
        compute_sdpa_chunk<
            Sk_chunk_t,
            q_chunk_tiles,
            out_chunk_tiles,
            scale_fp32,
            transpose_k,
            transpose_v,
            packed_tile_size,
            exp_approx_mode,
            /*qk_signal_granularity=*/1,
            /*exp_signal_granularity=*/1,
            output_granularity,
            false>(
            cb_q_in,
            cb_k_in,
            cb_k_in,
            cb_mask,
            sdpa_output_cb,
            mm1_dst_offset,
            mm2_dst_offset,
            max_dst_offset,
            sum_dst_offset,
            corr_exp_dst_offset,
            j == 0,
            j == block_chunks - 1,
            mask_chunk);
        if (mask_chunk) {
            cb_pop_front(cb_mask, 1);
        }
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_block_contiguous(max_dst_tile_offset, sdpa_ms_cb, 1);
    cb_push_back(sdpa_ms_cb, 1);
    for (uint32_t i = 0; i < out_chunk_tiles; i += output_granularity) {
        PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
        pack_block_contiguous(mm2_dst_tile_offset + i, sdpa_output_cb, output_granularity);
        PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
    }
    cb_push_back(sdpa_output_cb, out_chunk_tiles);
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
    sdpa_custom_mm_block_uninit();
    MATH(t6_semaphore_wait_on_max<p_stall::STALL_SFPU>(semaphore::FPU_SFPU));
    cb_pop_front(cb_q_in, q_chunk_tiles);

    if (num_rounds == 0) {
        return;
    }
    reconfig_full_operand_srca(cb_ms_in);
    exp_tile_init<exp_approx_mode, scale_fp32>();
    for (uint32_t r = 0; r + 1 < num_rounds; ++r) {
        sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
            cb_ms_in, cb_interm_ms, cb_interm_ms, cb_out_in, cb_interm_out, cb_interm_out);
    }
    if (s_block == 0) {
        sdpa_tail<
            exp_approx_mode,
            true,
            block_size,
            num_blocks,
            scale_fp32,
            VectorMode::C,
            /*dense=*/false,
            /*untilize=*/true>(cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_rm);
    } else {
        sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
            cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_o);
    }
}
