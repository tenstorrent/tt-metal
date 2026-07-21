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
 *      - in0_transpose: lambda PreKBlockFn WH-transposes in0 before each K-block.
 *      - SFPU activation (no FUSE_BIAS): packer-thread activation runs in the helper's
 *        per-subblock pack stage on the last K-block (overlaps with the next math
 *        iteration; frees math-thread DST register pressure).
 *      - PACK_RELU (no FUSE_BIAS): helper enables relu on last K-block output.
 *      - TILE_PACK_ROW_MAJOR (factory-emitted define): helper does absolute-offset pack.
 *      - SKIP_COMPUTE (microbench define): helper elides inner matmul LLK call.
 *   2. Bias addition (add_bias_bcast_rows helper): reads partials, adds row-broadcast
 *      bias, optionally applies SFPU activation on the packer thread (FUSE_BIAS path),
 *      packs to output. Caller owns bias CB wait/pop lifecycle since the reader may
 *      push bias only once across multiple bh/batch iterations.
 *   3. Untilize (reblock_and_untilize helper): gathers subblock-order output into
 *      row-major and untilizes.
 *
 * Activation routing:
 *   FUSE_BIAS  → matmul produces partials only; bias helper applies activation after add.
 *   !FUSE_BIAS → matmul applies activation in its own pack stage; untilize phase (if
 *                active) reads the activated partials unchanged.
 * KernelActivation enum + 3 uint32_t params come from named compile-time args
 * (activation_type, activation_param0..2) emitted by the program factory.
 *
 * NOTE: The perf microbenchmark copy at
 * tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/
 *   bmm_large_block_zm_fused_bias_activation_copy.cpp
 * is a standalone copy using direct LLK calls (cannot import ttnn kernel_lib).
 */

#include <cstdint>
#include <type_traits>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers_advanced.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"

#ifdef FUSE_BIAS
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#endif

#include "api/compute/eltwise_binary.h"
#include "api/compute/transpose.h"
#include "api/compute/compute_kernel_hw_startup.h"

#ifdef SFPU_ACTIVATION
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#endif

namespace {

// WH-transpose `in_block_num_tiles` tiles from `in_transpose_buf` to `in_buf`,
// in chunks of `block_size` (defaults to 4, matches DST capacity across dst_sync
// modes and data formats) with a tail loop for the remainder. Inlined here
// because it has only ever had one caller — the in0_transpose PreKBlockFn lambda
// below.
template <uint32_t in_block_num_tiles, uint32_t block_size = 4, typename Buf>
FORCE_INLINE void transpose_tile_block(Buf& in_transpose_buf, Buf& in_buf) {
    using compute_kernel_lib::buf_id;
    constexpr uint32_t num_blocks = in_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in_block_num_tiles % block_size;

    const uint32_t in_transpose_cb_id = buf_id(in_transpose_buf);
    const uint32_t in_cb_id = buf_id(in_buf);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        in_transpose_buf.wait_front(block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_tile(in_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in_transpose_buf.pop_front(block_size);

        in_buf.reserve_back(block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in_cb_id);
        }
        tile_regs_release();
        in_buf.push_back(block_size);
    }

    if constexpr (last_block_size > 0) {
        in_transpose_buf.wait_front(last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_tile(in_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in_transpose_buf.pop_front(last_block_size);

        in_buf.reserve_back(last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in_cb_id);
        }
        tile_regs_release();
        in_buf.push_back(last_block_size);
    }
}

}  // namespace

// Please update
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// when making any changes to this file.
// Have to keep a copy because cannot import ttnn into tests/tt_metal.
// With FUSE_BIAS: row_broadcast_bias (row-broadcast vs elementwise add_tiles) is compile-time arg 18 here;
// the perf copy uses index 14 (different compile-time arg layout).

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
    [[maybe_unused]] constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
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
    [[maybe_unused]] constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);
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

    CircularBuffer in0_buf(in0_cb_id);
    CircularBuffer in1_buf(in1_cb_id);
    CircularBuffer out_buf(out_cb_id);
    CircularBuffer mm_partials_buf(mm_partials_cb_id);
    CircularBuffer untilize_mode_out_buf(untilize_mode_out_cb_id);
    CircularBuffer in0_transpose_buf(in0_transpose_cb_id);

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t bias_ntiles = get_named_compile_time_arg_val("bias_ntiles");
    // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles (bias has multiple M rows).
    constexpr bool row_broadcast_bias = (bool)get_compile_time_arg_val(18);
    CircularBuffer bias_buf(bias_cb_id);
#endif

    // ── SFPU activation params (compile-time, named CT args) ────────────
    // Always declared so the helper template arguments resolve to a value either way;
    // when SFPU_ACTIVATION is undefined the values default to NONE/0 and the helpers
    // statically discard their packer-side activation paths.
#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type =
        static_cast<KernelActivation>(get_named_compile_time_arg_val("activation_type"));
    constexpr uint32_t activation_param0 = get_named_compile_time_arg_val("activation_param0");
    constexpr uint32_t activation_param1 = get_named_compile_time_arg_val("activation_param1");
    constexpr uint32_t activation_param2 = get_named_compile_time_arg_val("activation_param2");
#else
    constexpr KernelActivation activation_type = KernelActivation::NONE;
    constexpr uint32_t activation_param0 = 0;
    constexpr uint32_t activation_param1 = 0;
    constexpr uint32_t activation_param2 = 0;
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
    constexpr OutputCBLayout output_layout =
#ifdef TILE_PACK_ROW_MAJOR
        OutputCBLayout::TileRowMajor;
#else
        OutputCBLayout::SubblockMajor;
#endif

    // Last-block pack target. Interm covers two distinct downstream phases (FUSE_BIAS
    // and untilize_out without FUSE_BIAS); see the activation routing block below for
    // how each case decides where activation runs.
#if defined(FUSE_BIAS)
    constexpr LastBlockTarget last_block_target = LastBlockTarget::Interm;
#elif defined(PACK_RELU)
    constexpr LastBlockTarget last_block_target = untilize_out ? LastBlockTarget::Interm : LastBlockTarget::OutWithRelu;
#else
            constexpr LastBlockTarget last_block_target = untilize_out ? LastBlockTarget::Interm : LastBlockTarget::Out;
#endif

    // Activation routing:
    //   FUSE_BIAS  → matmul produces partials only; bias helper applies activation.
    //   !FUSE_BIAS → matmul applies activation; downstream untilize (if any) consumes
    //                the activated partials unchanged.
#ifdef FUSE_BIAS
    constexpr KernelActivation matmul_activation = KernelActivation::NONE;
    constexpr KernelActivation bias_activation = activation_type;
#else
    constexpr KernelActivation matmul_activation = activation_type;
    constexpr KernelActivation bias_activation = KernelActivation::NONE;
#endif

    // ── Init ────────────────────────────────────────────────────────────
    // matmul_block is invoked with InitMode::None (avoids re-init corruption on
    // heterogeneous-tile-shape DRAM-sharded configs — see
    // test_matmul_batched_dram_sharded[wkv_b2] tile_h=4), so the kernel boots the matmul
    // init AND the activation init explicitly. mm_block_init is deprecated: boot with
    // compute_kernel_hw_startup (hw_configure) then matmul_block_init (unpack/math init).
    // ActivationInitHelper::init() is a compile-time no-op when activation_type ==
    // KernelActivation::NONE.
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, mm_partials_cb_id);
    matmul_block_init(in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    if constexpr (activation_type != KernelActivation::NONE) {
        ActivationInitHelper<activation_type, activation_param0, activation_param1>::init();
    }

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
                    PACK((llk_pack_relu_config(ReluConfig::none())));
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
                auto shape = MatmulBlockShape::of(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    in0_block_w,
                    num_blocks_inner_dim,
                    /*batch=*/1,
                    /*in1_per_core_w=*/in1_block_w,
                    /*out_row_width=*/out_block_w);
#ifdef MATMUL_DRAM_SHARDED
                // DRAM-sharded matmul pads per_core_N_compute beyond per_core_N_in1_sender
                // so out_subblock_w can be larger than the reader actually pushes for the
                // last in1 subblock. Narrow the matmul FMA's ct_dim on the last subblock
                // via the helper's last_in1_subblock_w_valid knob so the unpacker only
                // touches tiles that exist. Mirrors tt-metal #44872's kernel-side fix.
                shape.last_in1_subblock_w_valid = get_named_compile_time_arg_val("last_subblock_w_valid");
#endif
                // in1_block_w is the in1 CB read stride (= per_core_N_in1_sender on
                // DRAM-sharded, possibly smaller than the output pack width). out_block_w
                // = out_subblock_w * in1_num_subblocks is the row stride the compute
                // actually packs into — on DRAM-sharded this is the padded
                // per_core_N_compute, so row-major reserve/push and multi-row
                // absolute-offset pack need it, not in1_block_w.
                //
                // Two branches: the in0_transpose lambda captures the four CBs by
                // reference, so its closure type can't be hoisted out of the branch.
                // kernel_main isn't a template, so if-constexpr doesn't discard the
                // unused branch's construction — keep each lambda local to its branch.
                if constexpr (in0_transpose_tile) {
                    auto xpose = [&](uint32_t /*k_iter*/, uint32_t /*num_k_iters*/, bool /*is_last*/) {
                        const uint32_t in0_transpose_cb_id = buf_id(in0_transpose_buf);
                        const uint32_t in0_cb_id = buf_id(in0_buf);
                        const uint32_t in1_cb_id = buf_id(in1_buf);
                        const uint32_t mm_partials_cb_id = buf_id(mm_partials_buf);

                        reconfig_data_format_srca(in1_cb_id, in0_transpose_cb_id);
                        transpose_init(in0_transpose_cb_id);
                        PACK((pack_reconfig_data_format(in0_cb_id)));
#ifdef PACKER_L1_ACC
                        PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                        transpose_tile_block<in0_block_num_tiles>(in0_transpose_buf, in0_buf);
                        mm_block_init_short_with_dt(
                            in0_cb_id,
                            in1_cb_id,
                            in0_transpose_cb_id,
                            in1_transpose_tile,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                        PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                    };
                    matmul_block_gathered<
                        in1_transpose_tile,                                 // transpose
                        l1_acc,                                             // packer_l1_acc
                        last_block_target,                                  // last_block_target
                        output_layout,                                      // layout
                        matmul_config::InitMode::None,                      // init_mode
                        InputPolicy::WaitAndPopPerKBlock,                   // in0_policy
                        InputPolicy::WaitAndPopPerKBlock,                   // in1_policy
                        matmul_config::DataFormatReconfig::InputAndOutput,  // reconfig (was defaulted)
                        ActivationOp<
                            matmul_activation,
                            activation_param0,
                            activation_param1,
                            activation_param2>,  // Activation
                        NoPostCompute,           // PostComputeFn (math-thread; unused)
                        decltype(xpose),         // PreKBlockFn
                        NoPostKBlock,            // PostKBlockFn
                        NoKBlockInnerDimFn,      // KBlockInnerDimFn
                        NoIn0Source,             // In0SourceFn
                        NoIn1BaseOffset>(        // In1BaseOffsetFn
                        in0_buf,
                        in1_buf,
                        phase1_out_buf,
                        mm_partials_buf,
                        shape,
                        NoPostCompute{},
                        xpose);
                } else {
                    matmul_block_gathered<
                        in1_transpose_tile,
                        l1_acc,
                        last_block_target,
                        output_layout,
                        matmul_config::InitMode::None,
                        InputPolicy::WaitAndPopPerKBlock,
                        InputPolicy::WaitAndPopPerKBlock,
                        matmul_config::DataFormatReconfig::InputAndOutput,
                        ActivationOp<matmul_activation, activation_param0, activation_param1, activation_param2>,
                        NoPostCompute,
                        NoPreKBlock,
                        NoPostKBlock,
                        NoKBlockInnerDimFn,
                        NoIn0Source,
                        NoIn1BaseOffset>(
                        in0_buf, in1_buf, phase1_out_buf, mm_partials_buf, shape, NoPostCompute{}, NoPreKBlock{});
                }

                // ── Phase 2: Bias addition ──────────────────────────────
#ifdef FUSE_BIAS
#ifdef PACK_RELU
                // if last block we pack the final result with relu enabled
                PACK((llk_pack_relu_config(ReluConfig::zero())));
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
                add_bias_bcast_rows<
                    bias_broadcast,               // broadcast
                    output_layout,                // output_layout
                    bias_add_config::NoPostBias,  // PostBiasFn (math-thread; unused)
                    ActivationOp<bias_activation, activation_param0, activation_param1, activation_param2>>(
                    mm_partials_buf, bias_buf, untilize_mode_out_buf, bias_shape);

                if constexpr (num_blocks_w_dim > 1) {
                    bias_buf.pop_front(bias_ntiles);
                }
#endif  // FUSE_BIAS

                // ── Phase 3: Untilize ───────────────────────────────────
                if constexpr (untilize_out) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluConfig::none())));
#endif  // PACK_RELU
#ifndef FUSE_BIAS
                    reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif  // !FUSE_BIAS
        // This kernel manages the srcA / pack data-format reconfig externally above
        // (tangled with the FP32/PACKER_L1_ACC #ifdefs and the pack_reconfig_l1_acc
        // ordering), so reblock is invoked with NoReconfigure — the helper adds no
        // reconfig of its own. One call loops over all in0_num_subblocks internally.
                    reblock_and_untilize<
                        out_subblock_w,
                        out_block_w,
                        reblock_untilize_config::InitUninitMode::InitAndUninit,
                        reblock_untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                        in0_num_subblocks,
                        in1_num_subblocks,
                        out_subblock_num_tiles,
                        out_subblock_h,
                        mm_partials_buf,
                        out_buf);
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
#ifdef FUSE_BIAS
    // For num_blocks_w_dim == 1 the reader pushes bias once and the kernel holds it resident,
    // reusing it across all batch/bh/block iterations without popping. Pop it once here, after the
    // last use, so the CB is balanced. (For num_blocks_w_dim > 1 the per-block pop above already
    // balances each re-pushed bias block.)
    if constexpr (num_blocks_w_dim == 1) {
        bias_buf.pop_front(bias_ntiles);
    }
#endif
}
