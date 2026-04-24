// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bmm_large_block_zm_fused_bias_activation.cpp
 * @brief Blocked matmul with optional fused bias, SFPU activation, transpose, and untilize.
 *
 * Full-featured bmm path used when any of: bias, activation, in0_transpose, untilize,
 * DRAM sharding, or multi-block h/w dims are enabled. For the simple case see
 * bmm_large_block_zm.cpp.
 *
 * Pipeline phases (each optional, selected by compile-time defines):
 *   1. K-loop matmul (matmul_block helper): K-blocking, spill/reload, L1 accumulation.
 *      - in0_transpose: TransposePreKBlock functor transposes input before each K-block.
 *      - SFPU activation (no bias): PostComputeFn applies per sub-block on last K-block.
 *      - PACK_RELU (no bias): helper enables relu on last K-block output.
 *      - ROW_MAJOR_OUTPUT (factory-emitted define): helper does absolute-offset pack.
 *      - SKIP_COMPUTE (microbench define): helper elides inner matmul LLK call.
 *   2. Bias addition (add_bias_bcast_rows helper): reads partials, adds row-broadcast
 *      bias, optionally applies SFPU activation, packs to output. Caller owns bias
 *      CB wait/pop lifecycle since the reader may push bias only once across
 *      multiple bh/batch iterations.
 *   3. Untilize (reblock_and_untilize helper): gathers subblock-order output into
 *      row-major and untilizes.
 *
 * NOTE: The perf microbenchmark copy at
 * tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/
 *   bmm_large_block_zm_fused_bias_activation_copy.cpp
 * is a standalone copy using direct LLK calls (cannot import ttnn kernel_lib).
 */

#include <cstdint>
#include <type_traits>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/transpose_block_helpers.hpp"

#ifdef FUSE_BIAS
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "experimental/circular_buffer.h"
#endif

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

// Please update
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// when making any changes to this file.
// Have to keep a copy because cannot import ttnn into tests/tt_metal.
// With FUSE_BIAS: row_broadcast_bias (row-broadcast vs elementwise add_tiles) is compile-time arg 18 here;
// the perf copy uses index 14 (different compile-time arg layout).

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
    [[maybe_unused]] constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
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
    constexpr uint32_t bias_ntiles = get_named_compile_time_arg_val("bias_ntiles");
    constexpr uint32_t mm_out_cb_id = mm_partials_cb_id;
    // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles (bias has multiple M rows).
    constexpr bool row_broadcast_bias = (bool)get_compile_time_arg_val(18);
    experimental::CircularBuffer bias_cb(bias_cb_id);
#else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
#endif

    // ── Feature flags ───────────────────────────────────────────────────
#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool l1_acc =
#ifdef PACKER_L1_ACC
        true;
#else
        false;
#endif

    // PACK_RELU without bias: helper handles it via pack_relu template param.
    // PACK_RELU with bias: caller manages RELU config between matmul and bias phases.
    constexpr bool do_relu =
#if defined(PACK_RELU) && !defined(FUSE_BIAS)
        true;
#else
        false;
#endif

    // ROW_MAJOR_OUTPUT: factory opts in to absolute-offset packing; writers read row-major.
    constexpr OutputLayout output_layout =
#ifdef ROW_MAJOR_OUTPUT
        OutputLayout::RowMajor;
#else
        OutputLayout::SubblockMajor;
#endif

    // matmul_block packs its last K-block to interm when a downstream phase (bias, untilize)
    // consumes from interm.
    constexpr bool pack_last_to_interm =
#ifdef FUSE_BIAS
        true;
#else
        untilize_out;
#endif

    // ── Callback type aliases ───────────────────────────────────────────
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
    using PreFn = std::conditional_t<in0_transpose_tile, XposeFn, NoPreKBlock>;

#ifdef SFPU_OP_INIT_ACTIVATION
    using PostFn = std::conditional_t<pack_last_to_interm, NoPostCompute, SFPUPostCompute>;
#else
    using PostFn = NoPostCompute;
#endif

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
                constexpr uint32_t phase1_out_cb =
#ifdef FUSE_BIAS
                    out_cb_id;
#else
                    mm_out_cb_id;
#endif
                matmul_block<in1_transpose_tile, l1_acc, pack_last_to_interm, do_relu, output_layout, PostFn, PreFn>(
                    in0_cb_id,
                    in1_cb_id,
                    phase1_out_cb,
                    mm_partials_cb_id,
                    in0_block_w,
                    in0_num_subblocks,
                    in1_num_subblocks,
                    num_blocks_inner_dim,
                    out_subblock_h,
                    out_subblock_w,
                    1,
                    PostFn{},
                    PreFn{},
                    /*retain_in0=*/false,
                    // in1_block_w is the in1 CB read stride (= per_core_N_in1_sender on
                    // DRAM-sharded, possibly smaller than the output pack width). out_block_w
                    // = out_subblock_w * in1_num_subblocks is the row stride the compute
                    // actually packs into — on DRAM-sharded this is the padded
                    // per_core_N_compute, so row-major reserve/push and multi-row
                    // absolute-offset pack need it, not in1_block_w.
                    /*in1_per_core_w=*/in1_block_w,
                    /*out_row_width=*/out_block_w);

                // ── Phase 2: Bias addition ──────────────────────────────
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

                // Bias CB lifecycle: the reader pushes bias once when num_blocks_w_dim == 1
                // (tiles stay fronted across all bh/batch iterations) and pushes per-iter
                // when num_blocks_w_dim > 1. Wait on first fronting; pop only when consumed.
                if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                    cb_wait_front(bias_cb_id, bias_ntiles);
                }

                constexpr BiasBroadcast bias_broadcast =
                    row_broadcast_bias ? BiasBroadcast::RowBroadcast : BiasBroadcast::Elementwise;
#ifdef SFPU_OP_INIT_ACTIVATION
                add_bias_bcast_rows<
                    mm_partials_cb_id,
                    bias_cb_id,
                    untilize_mode_out_cb_id,
                    bias_broadcast,
                    output_layout,
                    SFPUPostCompute>(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    SFPUPostCompute{},
                    /*out_row_width=*/out_block_w);
#else
                add_bias_bcast_rows<
                    mm_partials_cb_id,
                    bias_cb_id,
                    untilize_mode_out_cb_id,
                    bias_broadcast,
                    output_layout>(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    compute_kernel_lib::bias_add_config::NoPostBias{},
                    /*out_row_width=*/out_block_w);
#endif

                if constexpr (num_blocks_w_dim > 1) {
                    cb_pop_front(bias_cb_id, bias_ntiles);
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
