// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bmm_large_block_zm_fused_bias_activation.cpp
 * @brief Blocked matmul with optional fused bias, SFPU activation, transpose, and untilize.
 *
 * This kernel handles the full-featured matmul path used by the matmul program factory
 * when any of: bias, activation, in0_transpose, untilize, DRAM sharding, or multi-block
 * h/w dims are enabled. For the simple case (none of the above), see bmm_large_block_zm.cpp.
 *
 * Pipeline phases (each optional, selected by compile-time defines):
 *   1. K-loop matmul: matmul_block helper with K-blocking, spill/reload, L1 accumulation.
 *      - in0_transpose: TransposePreKBlock functor transposes input before each K-block.
 *      - SFPU activation (no bias): PostComputeFn applies per sub-block on last K-block.
 *      - PACK_RELU (no bias): helper enables relu on last K-block output.
 *   2. Bias addition: add_bias_bcast_rows helper reads matmul partials, adds row-broadcast
 *      bias, optionally applies SFPU activation, packs to output.
 *   3. Untilize: reblock_and_untilize converts tiled output to row-major.
 *
 * NOTE: The perf microbenchmark copy at
 * tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/
 *   bmm_large_block_zm_fused_bias_activation_copy.cpp
 * is a standalone copy using direct LLK calls (cannot import ttnn kernel_lib).
 */

#include <cstdint>
#include <type_traits>

#include "api/compute/pack_untilize.h"
#include "api/compute/transpose_wh.h"
#include "internal/mod_div_lib.h"

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

#ifdef FUSE_BIAS
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#endif

#include "api/compute/eltwise_unary/sfpu_split_includes.h"

// ─── Local helper functions ─────────────────────────────────────────────────

/**
 * @brief Transpose a block of tiles from one CB to another (WH swap per tile).
 *
 * Processes in batches of block_size tiles for DST efficiency, with a tail
 * loop for remainders. Used by both the TransposePreKBlock functor (production)
 * and the SKIP_COMPUTE fallback path.
 */
template <uint32_t in0_block_num_tiles, uint32_t block_size = 4>
FORCE_INLINE void transpose_tile_block(uint32_t in0_transpose_cb_id, uint32_t in0_cb_id) {
    constexpr uint32_t num_blocks = in0_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in0_block_num_tiles % block_size;

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        cb_wait_front(in0_transpose_cb_id, block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_wh_tile(in0_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        cb_pop_front(in0_transpose_cb_id, block_size);
        cb_reserve_back(in0_cb_id, block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in0_cb_id);
        }
        tile_regs_release();
        cb_push_back(in0_cb_id, block_size);
    }

    if constexpr (last_block_size > 0) {
        cb_wait_front(in0_transpose_cb_id, last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_wh_tile(in0_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        cb_pop_front(in0_transpose_cb_id, last_block_size);
        cb_reserve_back(in0_cb_id, last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in0_cb_id);
        }
        tile_regs_release();
        cb_push_back(in0_cb_id, last_block_size);
    }
}

/**
 * @brief Reblock and untilize matmul output from sub-block layout to row-major.
 */
template <uint32_t out_subblock_w, uint32_t out_block_w>
inline void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t interm_cb_id,
    uint32_t out_cb_id) {
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    cb_wait_front(interm_cb_id, num_tiles_in_row_of_subblocks);

    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;
        cb_reserve_back(out_cb_id, out_block_w);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                copy_tile(interm_cb_id, block_offset + within_block_index + w, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<out_subblock_w, out_block_w>(out_cb_id, 1, n);
            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(out_cb_id, out_block_w);
        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}

// ─── Functors for matmul_block / add_bias_bcast_rows callbacks ──────────────

#ifdef SFPU_OP_INIT_ACTIVATION
// Applied per output sub-block after matmul (no-bias) or after bias addition.
struct SFPUPostCompute {
    ALWI void operator()(uint32_t num_tiles) const {
        for (uint32_t i = 0; i < num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
#endif

/**
 * PreKBlockFn: transposes input block before each K-block iteration.
 * Reconfigures data formats for the transpose, runs the transpose, then
 * reinits matmul. The matmul_block helper's L1_ACC management restores
 * accumulation state for the subsequent pack phase.
 */
template <
    uint32_t in0_block_num_tiles_v,
    uint32_t in0_transpose_cb_id_v,
    uint32_t in0_cb_id_v,
    uint32_t in1_cb_id_v,
    uint32_t in1_transpose_tile_v,
    uint32_t out_subblock_w_v,
    uint32_t out_subblock_h_v,
    uint32_t in0_block_w_v,
    uint32_t mm_partials_cb_id_v>
struct TransposePreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {
        reconfig_data_format_srca(in1_cb_id_v, in0_transpose_cb_id_v);
        transpose_wh_init_short(in0_transpose_cb_id_v);
        PACK((pack_reconfig_data_format(in0_cb_id_v)));
#ifdef PACKER_L1_ACC
        PACK((llk_pack_reconfig_l1_acc(0)));
#endif
        transpose_tile_block<in0_block_num_tiles_v>(in0_transpose_cb_id_v, in0_cb_id_v);
        mm_block_init_short_with_dt(
            in0_cb_id_v,
            in1_cb_id_v,
            in0_transpose_cb_id_v,
            in1_transpose_tile_v,
            out_subblock_w_v,
            out_subblock_h_v,
            in0_block_w_v);
        PACK((pack_reconfig_data_format(mm_partials_cb_id_v)));
    }
};

// ─── SKIP_COMPUTE fallback ──────────────────────────────────────────────────
//
// When SKIP_COMPUTE is defined, the matmul calls are omitted but all other
// pipeline operations (transpose, reload, spill, pack, L1_ACC management)
// still execute. This is used for perf microbenchmarking the non-compute
// portions of the kernel. The matmul_block helper cannot be used here because
// it encapsulates the matmul calls.

#ifdef SKIP_COMPUTE
template <
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t out_cb_id,
    uint32_t mm_out_cb_id,
    uint32_t mm_partials_cb_id,
    uint32_t in0_transpose_cb_id,
    bool in0_transpose_tile,
    bool in1_transpose_tile,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_block_num_tiles,
    uint32_t out_subblock_num_tiles,
    uint32_t out_block_num_tiles,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t num_blocks_inner_dim,
    bool spill>
void skip_compute_k_loop() {
    bool enable_reload = false;
    uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

    for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
        bool last_out = block == (num_blocks_inner_dim - 1);

        if constexpr (in0_transpose_tile) {
            reconfig_data_format_srca(in1_cb_id, in0_transpose_cb_id);
            transpose_wh_init_short(in0_transpose_cb_id);
            PACK((pack_reconfig_data_format(in0_cb_id)));
#ifdef PACKER_L1_ACC
            PACK((llk_pack_reconfig_l1_acc(0)));
#endif
            transpose_tile_block<in0_block_num_tiles>(in0_transpose_cb_id, in0_cb_id);
            mm_block_init_short_with_dt(
                in0_cb_id,
                in1_cb_id,
                in0_transpose_cb_id,
                in1_transpose_tile,
                out_subblock_w,
                out_subblock_h,
                in0_block_w);
            PACK((pack_reconfig_data_format(mm_partials_cb_id)));
        }

        cb_wait_front(in0_cb_id, in0_block_num_tiles);
        cb_wait_front(in1_cb_id, in1_block_num_tiles);

        int in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                tile_regs_acquire();
                if (enable_reload) {
                    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
                    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
                    copy_block_matmul_partials(mm_partials_cb_id, 0, 0, out_subblock_num_tiles);
                    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
                    mm_block_init_short_with_dt(
                        in0_cb_id,
                        in1_cb_id,
                        mm_partials_cb_id,
                        in1_transpose_tile,
                        out_subblock_w,
                        out_subblock_h,
                        in0_block_w);
                }

                // No matmul computation (SKIP_COMPUTE)

                if (last_out) {
                    tile_regs_commit();
                    cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                    tile_regs_wait();
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                    PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
#else
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif
                    pack_tile_block(0, mm_out_cb_id, out_subblock_num_tiles);
                    tile_regs_release();
                    cb_push_back(mm_out_cb_id, out_subblock_num_tiles);
                } else {
                    tile_regs_commit();
                    if (block == 0) {
                        cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                        out_num_tiles_to_wait += out_subblock_num_tiles;
                    }
                    cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                    tile_regs_wait();
#ifdef PACKER_L1_ACC
                    if (block == 0) {
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    } else if (block == 1 || in0_transpose_tile) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
#endif
                    pack_tile_block(0, mm_partials_cb_id, out_subblock_num_tiles);
                    tile_regs_release();
                    cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                }
                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
        if (block < num_blocks_inner_dim - 1) {
            cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
            cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
        }
        enable_reload = false;
#else
        if (block < num_blocks_inner_dim - 2) {
            cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
            cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
        }
        if (block == num_blocks_inner_dim - 2) {
            enable_reload = true;
        }
#endif
#else
        if constexpr (spill) {
            enable_reload = true;
        }
#endif
        cb_pop_front(in0_cb_id, in0_block_num_tiles);
        cb_pop_front(in1_cb_id, in1_block_num_tiles);
    }
}
#endif  // SKIP_COMPUTE

// ─── kernel_main ────────────────────────────────────────────────────────────

void kernel_main() {
    using namespace compute_kernel_lib;

#ifdef MATMUL_DRAM_SHARDED
    if (get_arg_val<uint32_t>(0) != 1) {
        return;
    }
#endif

    // ── Compile-time arguments ──────────────────────────────────────────
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t batch = get_compile_time_arg_val(13);
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);
    constexpr bool untilize_out = get_compile_time_arg_val(15);
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(16);
    constexpr bool in0_transpose_tile = (bool)get_compile_time_arg_val(17);

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    // ── Circular buffer IDs ─────────────────────────────────────────────
    constexpr uint32_t in0_cb_id = in0_transpose_tile ? get_named_compile_time_arg_val("cb_in0_transposed")
                                                      : get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t mm_partials_cb_id = get_named_compile_time_arg_val("cb_intermed0");
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;
    constexpr uint32_t in0_transpose_cb_id = get_named_compile_time_arg_val("cb_in0");

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t mm_out_cb_id = mm_partials_cb_id;
#else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
#endif

    // ── Feature flags ───────────────────────────────────────────────────
#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;

    constexpr bool l1_acc =
#ifdef PACKER_L1_ACC
        true;
#else
        false;
#endif

    constexpr bool do_relu =
#if defined(PACK_RELU) && !defined(FUSE_BIAS)
        true;
#else
        false;
#endif

    // ── Callback type aliases ───────────────────────────────────────────
    // PreKBlockFn: transpose before each K-block, or no-op.
    using XposeFn = TransposePreKBlock<
        in0_block_num_tiles,
        in0_transpose_cb_id,
        in0_cb_id,
        in1_cb_id,
        in1_transpose_tile,
        out_subblock_w,
        out_subblock_h,
        in0_block_w,
        mm_partials_cb_id>;
    using PreFn = std::conditional_t<in0_transpose_tile, XposeFn, matmul_block_config::NoPreKBlock>;

    // ── Init ────────────────────────────────────────────────────────────
#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

    // ── Main loop: batch × output blocks ────────────────────────────────
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
            bool is_batch_valid = false;
            UNPACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            MATH(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            PACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                // Reset packer state for this output block
#ifdef PACK_RELU
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
                }
#endif
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                // ── Phase 1: K-loop matmul ──────────────────────────────
#ifdef SKIP_COMPUTE
                skip_compute_k_loop<
                    in0_cb_id,
                    in1_cb_id,
                    out_cb_id,
                    mm_out_cb_id,
                    mm_partials_cb_id,
                    in0_transpose_cb_id,
                    in0_transpose_tile,
                    in1_transpose_tile,
                    in0_block_num_tiles,
                    in0_subblock_num_tiles,
                    in1_block_num_tiles,
                    out_subblock_num_tiles,
                    out_block_num_tiles,
                    in0_num_subblocks,
                    in1_num_subblocks,
                    in0_block_w,
                    out_subblock_h,
                    out_subblock_w,
                    num_blocks_inner_dim,
                    spill>();
#else
#ifdef FUSE_BIAS
                matmul_block<
                    in0_cb_id,
                    in1_cb_id,
                    out_cb_id,
                    mm_partials_cb_id,
                    in1_transpose_tile,
                    l1_acc,
                    /*pack_last_to_interm=*/true,
                    /*pack_relu=*/false,
                    matmul_block_config::NoPostCompute,
                    PreFn>(
                    in0_block_w,
                    in0_num_subblocks,
                    in1_num_subblocks,
                    num_blocks_inner_dim,
                    out_subblock_h,
                    out_subblock_w,
                    1,
                    {},
                    PreFn{});
#else
                {
                    // PostComputeFn: SFPU activation on last K-block (no-bias path only).
#ifdef SFPU_OP_INIT_ACTIVATION
                    using PostFn = SFPUPostCompute;
#else
                    using PostFn = matmul_block_config::NoPostCompute;
#endif
                    matmul_block<
                        in0_cb_id,
                        in1_cb_id,
                        mm_out_cb_id,
                        mm_partials_cb_id,
                        in1_transpose_tile,
                        l1_acc,
                        /*pack_last_to_interm=*/false,
                        do_relu,
                        PostFn,
                        PreFn>(
                        in0_block_w,
                        in0_num_subblocks,
                        in1_num_subblocks,
                        num_blocks_inner_dim,
                        out_subblock_h,
                        out_subblock_w,
                        1,
                        PostFn{},
                        PreFn{});
                }
#endif
#endif  // SKIP_COMPUTE

                // ── Phase 2: Bias addition ──────────────────────────────
#ifdef FUSE_BIAS
#ifdef PACK_RELU
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                PACK((pack_reconfig_data_format(untilize_mode_out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                PACK((llk_pack_reconfig_l1_acc(0)));
#endif

#ifdef SFPU_OP_INIT_ACTIVATION
                add_bias_bcast_rows<mm_partials_cb_id, bias_cb_id, untilize_mode_out_cb_id, SFPUPostCompute>(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    in1_block_w,
                    SFPUPostCompute{});
#else
                add_bias_bcast_rows<mm_partials_cb_id, bias_cb_id, untilize_mode_out_cb_id>(
                    in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in1_block_w);
#endif

                if constexpr (num_blocks_w_dim > 1) {
                    cb_pop_front(bias_cb_id, in1_block_w);
                }
#endif  // FUSE_BIAS

                // ── Phase 3: Untilize ───────────────────────────────────
                if constexpr (untilize_out) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif
#ifndef FUSE_BIAS
                    reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif  // !FUSE_BIAS
                    pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb_id);
                    copy_tile_to_dst_init_short(mm_partials_cb_id);
                    for (uint32_t i = 0; i < in0_num_subblocks; ++i) {
                        reblock_and_untilize<out_subblock_w, out_block_w>(
                            in1_num_subblocks, out_subblock_num_tiles, out_subblock_h, mm_partials_cb_id, out_cb_id);
                    }
                    pack_untilize_uninit(mm_partials_cb_id);
                }

                // ── Reconfigure for next output block ───────────────────
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
#ifdef FUSE_BIAS
                    reconfig_data_format(mm_partials_cb_id, in1_cb_id, bias_cb_id, in0_cb_id);
#else
                    reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
#endif
                    mm_block_init_short(
                        in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
}
