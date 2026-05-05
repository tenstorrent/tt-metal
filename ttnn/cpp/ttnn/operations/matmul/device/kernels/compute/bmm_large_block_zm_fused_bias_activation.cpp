// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
 *      - TILE_PACK_ROW_MAJOR (factory-emitted define): helper does absolute-offset pack.
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

    experimental::CircularBuffer in0_buf(in0_cb_id);
    experimental::CircularBuffer in1_buf(in1_cb_id);
    experimental::CircularBuffer out_buf(out_cb_id);
    experimental::CircularBuffer mm_partials_buf(mm_partials_cb_id);
    experimental::CircularBuffer untilize_mode_out_buf(untilize_mode_out_cb_id);
    experimental::CircularBuffer in0_transpose_buf(in0_transpose_cb_id);

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t bias_ntiles = get_named_compile_time_arg_val("bias_ntiles");
    // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles (bias has multiple M rows).
    constexpr bool row_broadcast_bias = (bool)get_compile_time_arg_val(18);
    experimental::CircularBuffer bias_buf(bias_cb_id);
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

    // TILE_PACK_ROW_MAJOR: factory opts in to absolute-offset packing; writers read row-major.
    constexpr OutputLayout output_layout =
#ifdef TILE_PACK_ROW_MAJOR
        OutputLayout::RowMajor;
#else
        OutputLayout::SubblockMajor;
#endif

    // Last-block pack target: Interm when a downstream phase (bias add, untilize) consumes
    // from interm; OutWithRelu when PACK_RELU applies directly to the matmul output (no
    // downstream phase to host RELU); else plain Out. The enum makes the impossible
    // (Interm + Relu) combination unrepresentable.
#if defined(FUSE_BIAS)
    constexpr LastBlockTarget last_block_target = LastBlockTarget::Interm;
#elif defined(PACK_RELU)
    constexpr LastBlockTarget last_block_target = untilize_out ? LastBlockTarget::Interm : LastBlockTarget::OutWithRelu;
#else
            constexpr LastBlockTarget last_block_target = untilize_out ? LastBlockTarget::Interm : LastBlockTarget::Out;
#endif

    // Some downstream code still keys off pack_last_to_interm to decide post-bias type
    // selection; derive it from the enum for that local use only.
    constexpr bool pack_last_to_interm = (last_block_target == LastBlockTarget::Interm);

    // ── Callback type aliases ───────────────────────────────────────────
    using XposeFn =
        TransposePreKBlock<in0_block_num_tiles, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w>;

#ifdef SFPU_OP_INIT_ACTIVATION
    using PostFn = std::conditional_t<pack_last_to_interm, NoPostCompute, SFPUPostCompute>;
#else
    using PostFn = NoPostCompute;
#endif

    // ── Init ────────────────────────────────────────────────────────────
#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif
    // One-shot matmul init before the batch loop, mirroring the pre-refactor kernel:
    // mm_block_init configures unpack/math/pack registers for in0 × in1 → mm_partials.
    // Subsequent iterations re-arm via mm_block_init_short between iterations, and
    // matmul_block is invoked with InitMode::None so it doesn't re-do the full init
    // (which was found to corrupt state on heterogeneous-tile-shape DRAM-sharded
    // configs — see test_matmul_batched_dram_sharded[wkv_b2] tile_h=4 case).
    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    // reblock_and_untilize self-inits per call via the InitUninitMode template parameter
    // (we use Neither inside the in0_subblock loop so the standalone init/uninit wrappers
    // below handle it once).

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
#ifdef FUSE_BIAS
                auto& phase1_out_buf = out_buf;
#else
                auto& phase1_out_buf = untilize_mode_out_buf;
#endif
                const auto shape = MatmulBlockShape::of(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    in0_block_w,
                    num_blocks_inner_dim,
                    /*batch=*/1);
                // in1_block_w is the in1 CB read stride (= per_core_N_in1_sender on
                // DRAM-sharded, possibly smaller than the output pack width). out_block_w
                // = out_subblock_w * in1_num_subblocks is the row stride the compute
                // actually packs into — on DRAM-sharded this is the padded
                // per_core_N_compute, so row-major reserve/push and multi-row
                // absolute-offset pack need it, not in1_block_w.
                //
                // Two branches: XposeFn holds buffer references (no default ctor) so it
                // can't be passed through a single PreFn alias. kernel_main isn't a
                // template, so if-constexpr doesn't discard the unused branch's
                // construction — keep the type local to each branch.
                if constexpr (in0_transpose_tile) {
                    XposeFn xpose{in0_transpose_buf, in0_buf, in1_buf, mm_partials_buf};
                    matmul_block<
                        in1_transpose_tile,
                        l1_acc,
                        last_block_target,
                        output_layout,
                        matmul_config::InitMode::None,
                        /*retain_in0=*/false,
                        /*retain_in1=*/false,
                        PostFn,
                        XposeFn>(
                        in0_buf,
                        in1_buf,
                        phase1_out_buf,
                        mm_partials_buf,
                        shape,
                        PostFn{},
                        xpose,
                        /*in1_per_core_w=*/in1_block_w,
                        /*out_row_width=*/out_block_w);
                } else {
                    matmul_block<
                        in1_transpose_tile,
                        l1_acc,
                        last_block_target,
                        output_layout,
                        matmul_config::InitMode::None,
                        /*retain_in0=*/false,
                        /*retain_in1=*/false,
                        PostFn,
                        NoPreKBlock>(
                        in0_buf,
                        in1_buf,
                        phase1_out_buf,
                        mm_partials_buf,
                        shape,
                        PostFn{},
                        NoPreKBlock{},
                        /*in1_per_core_w=*/in1_block_w,
                        /*out_row_width=*/out_block_w);
                }

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
                    bias_buf.wait_front(bias_ntiles);
                }

                constexpr BiasBroadcast bias_broadcast =
                    row_broadcast_bias ? BiasBroadcast::RowBroadcast : BiasBroadcast::Elementwise;
                const auto bias_shape = BiasAddShape::of(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    /*out_row_width=*/out_block_w);
#ifdef SFPU_OP_INIT_ACTIVATION
                add_bias_bcast_rows<bias_broadcast, output_layout, SFPUPostCompute>(
                    mm_partials_buf, bias_buf, untilize_mode_out_buf, bias_shape, SFPUPostCompute{});
#else
                add_bias_bcast_rows<bias_broadcast, output_layout>(
                    mm_partials_buf, bias_buf, untilize_mode_out_buf, bias_shape);
#endif

                if constexpr (num_blocks_w_dim > 1) {
                    bias_buf.pop_front(bias_ntiles);
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
                    reblock_and_untilize_init<out_subblock_w, out_block_w>(mm_partials_buf, out_buf);
                    for (uint32_t i = 0; i < in0_num_subblocks; ++i) {
                        reblock_and_untilize<
                            out_subblock_w,
                            out_block_w,
                            reblock_untilize_config::InitUninitMode::Neither>(
                            in1_num_subblocks, out_subblock_num_tiles, out_subblock_h, mm_partials_buf, out_buf);
                    }
                    reblock_and_untilize_uninit(mm_partials_buf);
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
