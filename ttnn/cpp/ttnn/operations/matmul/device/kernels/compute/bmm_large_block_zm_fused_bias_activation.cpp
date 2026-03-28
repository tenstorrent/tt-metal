// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "internal/mod_div_lib.h"

#ifdef FUSE_BIAS
#include "api/compute/bcast.h"
#endif

#include "api/compute/eltwise_unary/sfpu_split_includes.h"

// Helper library for non-PACKER_L1_ACC, non-in0_transpose paths
#ifndef PACKER_L1_ACC
#ifdef FUSE_BIAS
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_fused_bias_helpers.hpp"
#else
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#endif

// PostComputeFn for SFPU activation (gelu, silu, etc.)
#ifdef SFPU_OP_INIT_ACTIVATION
struct ApplyActivation {
    ALWI void operator()(uint32_t num_tiles) const {
        for (uint32_t i = 0; i < num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
#endif

// Select the right PostComputeFn type based on compile flags
#if defined(SFPU_OP_INIT_ACTIVATION)
using HelperPostComputeFn = ApplyActivation;
#elif defined(FUSE_BIAS)
using HelperPostComputeFn = compute_kernel_lib::matmul_block_fused_bias_config::NoPostCompute;
#else
using HelperPostComputeFn = compute_kernel_lib::matmul_block_config::NoPostCompute;
#endif
#endif  // PACKER_L1_ACC

// Please update
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// when making any changes to this file.
// Have to keep a copy because cannot import ttnn into tests/tt_metal.

/**
 * @brief Transposes a block of tiles from one circular buffer to another.
 *
 * This function reads a block of tiles from the input circular buffer (cb), performs a width-height
 * (WH) transpose on each tile, and writes the transposed tiles to the output circular buffer.
 * The operation is performed in blocks of `block_size` tiles for efficiency, with a separate loop
 * at the end to handle any leftover tiles when the total tile count is not divisible by
 * `block_size`. The default block size is 4, since there are guaranteed to be 4 tiles in the dst
 *               regs irrespective of dst sync mode or data format.
 *
 * @tparam in0_block_num_tiles The number of tiles in the block to be transposed.
 * @tparam block_size The number of tiles in each block to be transposed.
 * @param in0_transpose_cb_id Circular buffer ID to read the original tiles from.
 * @param in0_cb_id Circular buffer ID to which the transposed tiles are written.
 */
template <uint32_t in0_block_num_tiles, uint32_t block_size = 4>
FORCE_INLINE void transpose_tile_block(uint32_t in0_transpose_cb_id, uint32_t in0_cb_id) {
    constexpr uint32_t num_blocks = in0_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in0_block_num_tiles % block_size;
    // Lets do 2 passes: One loop until last and one last for the left overs
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

FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t mm_partials_cb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    // Reconfigure input
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block_matmul_partials(mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
    // Reconfigure srcA back
    mm_block_init_short_with_dt(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
}

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
                uint32_t tile_index = block_offset + within_block_index + w;
                copy_tile(interm_cb_id, tile_index, w);
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

void kernel_main() {
// RUNTIME ARGS
#ifdef MATMUL_DRAM_SHARDED
    const bool is_worker_core = get_arg_val<uint32_t>(0) == 1;
    // if not worker core, skip
    if (not is_worker_core) {
        return;
    }
#endif

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);  // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);     // outer inner dim (in inner dim blocks)
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);         // outer inner dim (in inner dim blocks)
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);         // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(13);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(15);                // untilize output
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(16);
    constexpr bool in0_transpose_tile = (bool)get_compile_time_arg_val(17);

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    constexpr uint32_t in0_cb_id = in0_transpose_tile ? get_named_compile_time_arg_val("cb_in0_transposed")
                                                      : get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t mm_partials_cb_id = get_named_compile_time_arg_val("cb_intermed0");
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;
    // When in0 needs to be transposed, the original data is read from cb_in0 (in0_transpose_cb_id),
    // transposed, and the result is written to cb_in0_transposed (in0_cb_id), which is then used
    // as input for the matmul call.
    constexpr uint32_t in0_transpose_cb_id = get_named_compile_time_arg_val("cb_in0");

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t mm_out_cb_id = mm_partials_cb_id;
#else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
#endif

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;

    // ═══════════════════════════════════════════════════════════════════════════
    // PATH SELECTION: PACKER_L1_ACC and in0_transpose require hand-written code.
    // All other configurations use the library helpers.
    // ═══════════════════════════════════════════════════════════════════════════

#if defined(PACKER_L1_ACC) || defined(SKIP_COMPUTE) || defined(FP32_DEST_ACC_EN)
    // ── Hand-written path: PACKER_L1_ACC / SKIP_COMPUTE / FP32_DEST_ACC_EN ──
    // L1 accumulation fundamentally changes spill/reload behavior.
    // FP32_DEST_ACC_EN needs per-pack pack_reconfig_data_format calls.
    // SKIP_COMPUTE needs per-block CB management without matmul.
    // Both require the original hand-written implementation.
    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
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
                bool enable_reload = false;
                uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

#ifdef PACK_RELU
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
                }
#endif

                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                    bool last_out = block == (num_blocks_inner_dim - 1);
#if not defined FUSE_BIAS and defined PACK_RELU
                    if (last_out) {
                        PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
                    }
#endif

                    if constexpr (in0_transpose_tile) {
                        transpose_wh_init_short(in0_transpose_cb_id);
                        PACK((pack_reconfig_data_format(in0_cb_id)));
#ifdef PACKER_L1_ACC
                        PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                        transpose_tile_block<in0_block_num_tiles>(in0_transpose_cb_id, in0_cb_id);
                        mm_block_init_short(
                            in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
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
                                reload_from_cb_to_dst(
                                    in0_cb_id,
                                    in1_cb_id,
                                    mm_partials_cb_id,
                                    in1_transpose_tile,
                                    out_subblock_num_tiles,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                            }

#ifndef SKIP_COMPUTE
                            uint32_t dst_index = 0;
                            uint32_t in0_index = in0_index_subblock_offset;
                            uint32_t in1_index = in1_index_subblock_offset;
                            for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                                matmul_block(
                                    in0_cb_id,
                                    in1_cb_id,
                                    in0_index,
                                    in1_index,
                                    dst_index,
                                    in1_transpose_tile,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                                in0_index++;
                                in1_index += in1_block_w;
                            }
#endif  // SKIP_COMPUTE

                            if (last_out) {
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                    SFPU_OP_FUNC_ACTIVATION
                                }
#endif
                                tile_regs_commit();
                                cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                                tile_regs_wait();

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                                if (block == 0) {
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#else
                                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif

                                uint32_t start_dst_index = 0;
                                pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);

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
                                } else if (block == 1) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                } else if (in0_transpose_tile) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#endif

                                uint32_t start_dst_index = 0;
                                pack_tile_block(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);

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

#ifdef FUSE_BIAS
#ifdef PACK_RELU
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                PACK((llk_pack_reconfig_l1_acc(0)));
#endif

                reconfig_data_format(in1_cb_id, mm_partials_cb_id, in0_cb_id, bias_cb_id);
                add_bcast_rows_init_short(mm_partials_cb_id, bias_cb_id);
                cb_wait_front(bias_cb_id, in1_block_w);
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
                        tile_regs_acquire();
                        for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                            uint32_t bcast_tile_idx = in1_index_subblock_offset;
                            for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                                add_tiles_bcast_rows(mm_partials_cb_id, bias_cb_id, i, bcast_tile_idx, i);
                                bcast_tile_idx++;
                            }
                        }
#ifndef SFPU_OP_INIT_ACTIVATION
                        tile_regs_commit();
#endif
                        cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);

#ifdef SFPU_OP_INIT_ACTIVATION
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            SFPU_OP_FUNC_ACTIVATION
                        }
                        tile_regs_commit();
#endif

                        cb_reserve_back(untilize_mode_out_cb_id, out_subblock_num_tiles);
                        tile_regs_wait();
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, untilize_mode_out_cb_id);
                        }
                        tile_regs_release();
                        cb_push_back(untilize_mode_out_cb_id, out_subblock_num_tiles);

                        in1_index_subblock_offset += out_subblock_w;
                    }
                }
                if constexpr (num_blocks_w_dim > 1) {
                    cb_pop_front(bias_cb_id, in1_block_w);
                }
#endif  // FUSE_BIAS
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
#endif  // FUSE_BIAS
                    pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb_id);
                    copy_tile_to_dst_init_short(mm_partials_cb_id);
                    for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                        reblock_and_untilize<out_subblock_w, out_block_w>(
                            in1_num_subblocks, out_subblock_num_tiles, out_subblock_h, mm_partials_cb_id, out_cb_id);
                    }
                    pack_untilize_uninit(mm_partials_cb_id);
                }
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

#else  // !PACKER_L1_ACC && !SKIP_COMPUTE && !FP32_DEST_ACC_EN
    // ── Helper-based path ───────────────────────────────────────────────
    // Uses matmul_block / matmul_block_fused_bias library helpers.
    // Handles: FUSE_BIAS, SFPU activation, PACK_RELU (disabled — helpers
    // manage intermediate packing), FP32_DEST_ACC_EN, untilize_out,
    // num_blocks_h/w_dim, get_batch_from_reader, IN1_TRANSPOSE_TILE.
    // Does NOT handle: in0_transpose_tile (falls through to hand-written).

    if constexpr (in0_transpose_tile) {
        // in0_transpose requires per-K-block tile transposition which cannot
        // be injected into the helper's K-block loop. Use hand-written path.
        // (Same code as the PACKER_L1_ACC path above, minus L1_ACC ifdefs.)
        mm_block_init(
            in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
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
                    bool enable_reload = false;
                    uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
#ifdef PACK_RELU
                    if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                        PACK((llk_pack_relu_config(ReluType::NO_RELU)));
                    }
#endif
                    if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                        PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                    }
                    for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                        bool last_out = block == (num_blocks_inner_dim - 1);
#if not defined FUSE_BIAS and defined PACK_RELU
                        if (last_out) {
                            PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
                        }
#endif
                        transpose_wh_init_short(in0_transpose_cb_id);
                        PACK((pack_reconfig_data_format(in0_cb_id)));
                        transpose_tile_block<in0_block_num_tiles>(in0_transpose_cb_id, in0_cb_id);
                        mm_block_init_short(
                            in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                        PACK((pack_reconfig_data_format(mm_partials_cb_id)));

                        cb_wait_front(in0_cb_id, in0_block_num_tiles);
                        cb_wait_front(in1_cb_id, in1_block_num_tiles);
                        int in0_index_subblock_offset = 0;
                        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                            int in1_index_subblock_offset = 0;
                            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                                tile_regs_acquire();
                                if (enable_reload) {
                                    reload_from_cb_to_dst(
                                        in0_cb_id,
                                        in1_cb_id,
                                        mm_partials_cb_id,
                                        in1_transpose_tile,
                                        out_subblock_num_tiles,
                                        out_subblock_w,
                                        out_subblock_h,
                                        in0_block_w);
                                }
                                uint32_t dst_index = 0;
                                uint32_t in0_index = in0_index_subblock_offset;
                                uint32_t in1_index = in1_index_subblock_offset;
                                for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                                    matmul_block(
                                        in0_cb_id,
                                        in1_cb_id,
                                        in0_index,
                                        in1_index,
                                        dst_index,
                                        in1_transpose_tile,
                                        out_subblock_w,
                                        out_subblock_h,
                                        in0_block_w);
                                    in0_index++;
                                    in1_index += in1_block_w;
                                }
                                if (last_out) {
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                        SFPU_OP_FUNC_ACTIVATION
                                    }
#endif
                                    tile_regs_commit();
                                    cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                                    tile_regs_wait();
#ifdef FP32_DEST_ACC_EN
                                    PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif
                                    uint32_t start_dst_index = 0;
                                    pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);
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
                                    uint32_t start_dst_index = 0;
                                    pack_tile_block(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);
                                    tile_regs_release();
                                    cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                                }
                                in1_index_subblock_offset += out_subblock_w;
                            }
                            in0_index_subblock_offset += in0_subblock_num_tiles;
                        }
                        if constexpr (spill) {
                            enable_reload = true;
                        }
                        cb_pop_front(in0_cb_id, in0_block_num_tiles);
                        cb_pop_front(in1_cb_id, in1_block_num_tiles);
                    }
#ifdef FUSE_BIAS
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
#ifdef FP32_DEST_ACC_EN
                    PACK((pack_reconfig_data_format(out_cb_id)));
#endif
                    reconfig_data_format(in1_cb_id, mm_partials_cb_id, in0_cb_id, bias_cb_id);
                    add_bcast_rows_init_short(mm_partials_cb_id, bias_cb_id);
                    cb_wait_front(bias_cb_id, in1_block_w);
                    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                        int in1_index_subblock_offset = 0;
                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                            cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
                            tile_regs_acquire();
                            for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                                uint32_t bcast_tile_idx = in1_index_subblock_offset;
                                for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                                    add_tiles_bcast_rows(mm_partials_cb_id, bias_cb_id, i, bcast_tile_idx, i);
                                    bcast_tile_idx++;
                                }
                            }
#ifndef SFPU_OP_INIT_ACTIVATION
                            tile_regs_commit();
#endif
                            cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
#ifdef SFPU_OP_INIT_ACTIVATION
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                SFPU_OP_FUNC_ACTIVATION
                            }
                            tile_regs_commit();
#endif
                            cb_reserve_back(untilize_mode_out_cb_id, out_subblock_num_tiles);
                            tile_regs_wait();
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                pack_tile(i, untilize_mode_out_cb_id);
                            }
                            tile_regs_release();
                            cb_push_back(untilize_mode_out_cb_id, out_subblock_num_tiles);
                            in1_index_subblock_offset += out_subblock_w;
                        }
                    }
                    if constexpr (num_blocks_w_dim > 1) {
                        cb_pop_front(bias_cb_id, in1_block_w);
                    }
#endif  // FUSE_BIAS
                    if constexpr (untilize_out) {
#ifdef PACK_RELU
                        PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif
#ifndef FUSE_BIAS
                        reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
#ifdef FP32_DEST_ACC_EN
                        PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#endif
                        pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb_id);
                        copy_tile_to_dst_init_short(mm_partials_cb_id);
                        for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                            reblock_and_untilize<out_subblock_w, out_block_w>(
                                in1_num_subblocks,
                                out_subblock_num_tiles,
                                out_subblock_h,
                                mm_partials_cb_id,
                                out_cb_id);
                        }
                        pack_untilize_uninit(mm_partials_cb_id);
                    }
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
    } else {
        // ── Helper path: no L1_ACC, no transpose, no SKIP_COMPUTE ───────
        // The matmul K-block loop + optional bias add is replaced by a single
        // helper call per (batch, bh, bw) iteration.
        mm_block_init(
            in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

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
                    // Pack reconfig for intermediate format (needed between bh/bw iterations)
                    if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                        PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                    }

#ifdef FUSE_BIAS
                    // Fused matmul + bias add + optional SFPU activation.
                    // The helper handles: K-block loop with spill/reload, bias row-broadcast add,
                    // and PostComputeFn (SFPU activation after bias).
                    compute_kernel_lib::matmul_block_fused_bias<
                        in0_cb_id,
                        in1_cb_id,
                        untilize_mode_out_cb_id,
                        mm_partials_cb_id,
                        bias_cb_id,
                        compute_kernel_lib::matmul_block_fused_bias_config::InitUninitMode::Neither,
                        compute_kernel_lib::matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode::
                            NoReconfigure,
                        in1_transpose_tile,
                        HelperPostComputeFn>(
                        {.block_w = in0_block_w,
                         .num_subblocks = in0_num_subblocks,
                         .block_num_tiles = in0_block_num_tiles,
                         .subblock_num_tiles = in0_subblock_num_tiles},
                        {.num_subblocks = in1_num_subblocks,
                         .block_num_tiles = in1_block_num_tiles,
                         .per_core_w = in1_block_w},
                        num_blocks_inner_dim,
                        {.h = out_subblock_h, .w = out_subblock_w, .num_tiles = out_subblock_num_tiles},
                        1,
                        HelperPostComputeFn{});

                    if constexpr (num_blocks_w_dim > 1) {
                        cb_pop_front(bias_cb_id, in1_block_w);
                    }
#else
                    // Matmul only (no bias). Optional SFPU activation via PostComputeFn.
                    compute_kernel_lib::matmul_block<
                        in0_cb_id,
                        in1_cb_id,
                        mm_out_cb_id,
                        mm_partials_cb_id,
                        compute_kernel_lib::matmul_block_config::InitUninitMode::Neither,
                        compute_kernel_lib::matmul_block_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
                        compute_kernel_lib::matmul_block_config::WaitPopMode::WaitAndPop,
                        in1_transpose_tile,
                        HelperPostComputeFn>(
                        {.block_w = in0_block_w,
                         .num_subblocks = in0_num_subblocks,
                         .block_num_tiles = in0_block_num_tiles,
                         .subblock_num_tiles = in0_subblock_num_tiles},
                        {.num_subblocks = in1_num_subblocks,
                         .block_num_tiles = in1_block_num_tiles,
                         .per_core_w = in1_block_w},
                        num_blocks_inner_dim,
                        {.h = out_subblock_h, .w = out_subblock_w, .num_tiles = out_subblock_num_tiles},
                        1,
                        HelperPostComputeFn{});
#endif  // FUSE_BIAS

                    // Untilize output if needed
                    if constexpr (untilize_out) {
#ifndef FUSE_BIAS
                        reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
#ifdef FP32_DEST_ACC_EN
                        PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#endif  // FUSE_BIAS
                        pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb_id);
                        copy_tile_to_dst_init_short(mm_partials_cb_id);
                        for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                            reblock_and_untilize<out_subblock_w, out_block_w>(
                                in1_num_subblocks,
                                out_subblock_num_tiles,
                                out_subblock_h,
                                mm_partials_cb_id,
                                out_cb_id);
                        }
                        pack_untilize_uninit(mm_partials_cb_id);
                    }

                    // Reconfig for next bh/bw/batch iteration
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
#endif  // PACKER_L1_ACC || SKIP_COMPUTE || FP32_DEST_ACC_EN
}
