// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>

#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "experimental/circular_buffer.h"
#include "internal/mod_div_lib.h"

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

#ifdef FUSE_BIAS
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#endif

#include "api/compute/eltwise_unary/sfpu_split_includes.h"

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
    experimental::CircularBuffer in0_transpose_cb(in0_transpose_cb_id);
    experimental::CircularBuffer in0_cb(in0_cb_id);
    constexpr uint32_t num_blocks = in0_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in0_block_num_tiles % block_size;
    // Lets do 2 passes: One loop until last and one last for the left overs
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        in0_transpose_cb.wait_front(block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_wh_tile(in0_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in0_transpose_cb.pop_front(block_size);

        in0_cb.reserve_back(block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in0_cb_id);
        }
        tile_regs_release();
        in0_cb.push_back(block_size);
    }

    if constexpr (last_block_size > 0) {
        in0_transpose_cb.wait_front(last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_wh_tile(in0_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in0_transpose_cb.pop_front(last_block_size);

        in0_cb.reserve_back(last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in0_cb_id);
        }
        tile_regs_release();
        in0_cb.push_back(last_block_size);
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
    experimental::CircularBuffer mm_partials_cb(mm_partials_cb_id);
    // Reconfigure input
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    mm_partials_cb.wait_front(out_subblock_num_tiles);

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block_matmul_partials(mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    mm_partials_cb.pop_front(out_subblock_num_tiles);
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
    experimental::CircularBuffer interm_cb(interm_cb_id);
    experimental::CircularBuffer out_cb(out_cb_id);
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    interm_cb.wait_front(num_tiles_in_row_of_subblocks);

    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;

        out_cb.reserve_back(out_block_w);
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
        out_cb.push_back(out_block_w);

        within_block_index += out_subblock_w;
    }
    interm_cb.pop_front(num_tiles_in_row_of_subblocks);
}

// SFPU post-compute functor for matmul helper (no-bias paths)
#ifdef SFPU_OP_INIT_ACTIVATION
struct PostMatmulSFPU {
    ALWI void operator()(uint32_t out_subblock_num_tiles) const {
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
// SFPU post-bias functor for bias helper (bias paths)
struct PostBiasSFPU {
    ALWI void operator()(uint32_t out_subblock_num_tiles) const {
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
#endif

// PreKBlockFn for in0_transpose: transposes input block before each K-block's matmul.
// Disables L1_ACC during the transpose pack phase, then reinits matmul and restores
// pack format for the matmul phase. The helper's L1_ACC management re-enables L1_ACC
// as needed during the subsequent pack phase.
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

    experimental::CircularBuffer in0_cb(in0_cb_id);
    experimental::CircularBuffer in1_cb(in1_cb_id);
    experimental::CircularBuffer out_cb(out_cb_id);
    experimental::CircularBuffer mm_partials_cb(mm_partials_cb_id);
    experimental::CircularBuffer untilize_mode_out_cb(untilize_mode_out_cb_id);

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t mm_out_cb_id = mm_partials_cb_id;
    experimental::CircularBuffer bias_cb(bias_cb_id);
#else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
#endif
    experimental::CircularBuffer mm_out_cb(mm_out_cb_id);

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;

    // Compile-time feature flags for helper template params
    constexpr bool l1_acc =
#ifdef PACKER_L1_ACC
        true;
#else
        false;
#endif

    // Matmul output CB and post-compute activation depend on the FUSE_BIAS path.
    // With bias: pack to interm for the bias phase, no activation on the matmul output.
    // Without bias: pack directly to output with optional RELU or SFPU activation.
#ifdef FUSE_BIAS
    constexpr uint32_t matmul_out_cb = out_cb_id;
    constexpr bool pack_to_interm = true;
    using MatmulPostFn = compute_kernel_lib::matmul_block_config::NoPostCompute;
#else
    constexpr uint32_t matmul_out_cb = untilize_mode_out_cb_id;
    constexpr bool pack_to_interm = false;
#if defined(SFPU_OP_INIT_ACTIVATION)
    using MatmulPostFn = PostMatmulSFPU;
#elif defined(PACK_RELU)
    using MatmulPostFn = compute_kernel_lib::matmul_block_config::HwRelu;
#else
    using MatmulPostFn = compute_kernel_lib::matmul_block_config::NoPostCompute;
#endif
#endif

    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
            // Check whether this batch is valid
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
#ifdef PACK_RELU
                // for each batch we start with relu disabled so that intermediate results are not relu'd
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
                }
#endif

                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                // ── K-loop: use helper for standard paths, inline for SKIP_COMPUTE ──
#ifdef SKIP_COMPUTE
                // SKIP_COMPUTE: inline K-loop (matmul calls skipped, reload/pack still runs)
                {
                    bool enable_reload = false;
                    uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

                    for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                        bool last_out = block == (num_blocks_inner_dim - 1);
// Configure packer once for pack out without Bias
#if not defined FUSE_BIAS and defined PACK_RELU
                        if (last_out) {
                            // if last block we pack the final result with relu enabled
                            PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
                        }
#endif

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

                        in0_cb.wait_front(in0_block_num_tiles);
                        in1_cb.wait_front(in1_block_num_tiles);

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

                                // SKIP_COMPUTE: no matmul computation

                                if (last_out) {
// If we fuse bias, we will pack out and run bias + optional sfpu in a separate loop
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                        SFPU_OP_FUNC_ACTIVATION
                                    }
#endif
                                    tile_regs_commit();
                                    // Pack out to output buffer
                                    mm_out_cb.reserve_back(out_subblock_num_tiles);
                                    tile_regs_wait();

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                    PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                                    if (block == 0) {  // no accumulation for first iteration
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
                                    mm_out_cb.push_back(out_subblock_num_tiles);

                                } else {
                                    tile_regs_commit();
                                    // Wait for tiles in output buffer to be written out since interm and output share
                                    // memory
                                    if (block == 0) {
                                        out_cb.reserve_back(out_num_tiles_to_wait);
                                        out_num_tiles_to_wait += out_subblock_num_tiles;
                                    }
                                    // Move partial result to interm buffer
                                    mm_partials_cb.reserve_back(out_subblock_num_tiles);
                                    tile_regs_wait();

#ifdef PACKER_L1_ACC
                                    if (block == 0) {  // no accumulation for first iteration
                                        PACK((llk_pack_reconfig_l1_acc(0)));
                                    } else if (block == 1) {
                                        PACK((llk_pack_reconfig_l1_acc(1)));
                                    } else if (in0_transpose_tile) {
                                        // For each block, l1_acc would have been enabled during the
                                        // transpose stage. So let us put it back here.
                                        PACK((llk_pack_reconfig_l1_acc(1)));
                                    }
#endif

                                    uint32_t start_dst_index = 0;
                                    pack_tile_block(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);

                                    tile_regs_release();
                                    mm_partials_cb.push_back(out_subblock_num_tiles);
                                }

                                in1_index_subblock_offset += out_subblock_w;
                            }
                            in0_index_subblock_offset += in0_subblock_num_tiles;
                        }

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                        if (block < num_blocks_inner_dim - 1) {
                            // Wait for l1 accumulation to populate interm buffer,
                            // then pop to update fifo rd pointer
                            mm_partials_cb.wait_front(out_block_num_tiles);
                            mm_partials_cb.pop_front(out_block_num_tiles);
                        }
                        // never reload when with bias, bias uses interm buffer
                        enable_reload = false;
#else
                        // Last iteration does spill and reload to output buffer
                        if (block < num_blocks_inner_dim - 2) {
                            mm_partials_cb.wait_front(out_block_num_tiles);
                            mm_partials_cb.pop_front(out_block_num_tiles);
                        }
                        if (block == num_blocks_inner_dim - 2) {
                            enable_reload = true;
                        }  // reload when last iteration
#endif
#else
                        if constexpr (spill) {
                            enable_reload = true;
                        }
#endif

                        in0_cb.pop_front(in0_block_num_tiles);
                        in1_cb.pop_front(in1_block_num_tiles);
                    }
                }
#else  // !SKIP_COMPUTE
                {
                    // ── Helper path: both transpose and non-transpose use matmul_block ──
                    // When in0_transpose_tile=true, a TransposePreKBlock functor transposes
                    // the input block before each K-block's matmul. This eliminates the
                    // ~170-line inline K-loop that was previously needed.
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
                    // When no transpose, use a no-op PreKBlockFn
                    using NoPreFn = compute_kernel_lib::matmul_block_config::NoPreKBlock;

                    // Select PreKBlockFn at compile time
                    using PreFn = std::conditional_t<in0_transpose_tile, XposeFn, NoPreFn>;

                    compute_kernel_lib::matmul_block<
                        in0_cb_id,
                        in1_cb_id,
                        matmul_out_cb,
                        mm_partials_cb_id,
                        in0_num_subblocks,
                        in1_num_subblocks,
                        out_subblock_h,
                        out_subblock_w,
                        in1_transpose_tile,
                        l1_acc,
                        pack_to_interm,
                        MatmulPostFn>(in0_block_w, num_blocks_inner_dim, 1, MatmulPostFn{}, PreFn{});
                }
#endif  // SKIP_COMPUTE

                // ── Bias phase (via helper) ──
#ifdef FUSE_BIAS
#ifdef PACK_RELU
                // if last block we pack the final result with relu enabled
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                PACK((pack_reconfig_data_format(untilize_mode_out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                PACK((llk_pack_reconfig_l1_acc(0)));
#endif

                compute_kernel_lib::add_bias_bcast_rows<
                    mm_partials_cb_id,
                    bias_cb_id,
                    untilize_mode_out_cb_id
#ifdef SFPU_OP_INIT_ACTIVATION
                    ,
                    PostBiasSFPU
#endif
                    >(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    in1_block_w
#ifdef SFPU_OP_INIT_ACTIVATION
                    ,
                    PostBiasSFPU{}
#endif
                );

                if constexpr (num_blocks_w_dim > 1) {
                    bias_cb.pop_front(in1_block_w);
                }
#endif  // FUSE_BIAS

                // ── Untilize phase ──
                if constexpr (untilize_out) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif  // PACK_RELU
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
                    // reconfigure unpacker df for src A and src B
                    reconfig_data_format(mm_partials_cb_id, in1_cb_id, bias_cb_id, in0_cb_id);
#else
                    // reconfigure unpacker df for src A
                    reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
#endif
                    // reconfigure init for matmul
                    mm_block_init_short(
                        in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
}
