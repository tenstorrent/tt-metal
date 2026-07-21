// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

namespace compute_kernel_lib {

/**
 * matmul_block_gathered — ADVANCED public wrapper over matmul_block_impl that exposes the six
 * per-K-block / per-subblock functor hooks for gather / fused-bias / cross-chip matmul kernels:
 *   PostComputeFn     per-subblock MATH-thread hook on the last K-block, before pack.
 *   PreKBlockFn       per-K-block hook before the input waits (see the restore contract).
 *   PostKBlockFn      per-K-block hook after the input pops and the L1_ACC drain.
 *   KBlockInnerDimFn  per-K-block FMA step count (padded / partial K-blocks).
 *   In0SourceFn       per-K-block in0 CB selector (alternates must share in0's dataformat).
 *   In1BaseOffsetFn   per-K-block in1 base-offset shift within the fronted region.
 * See matmul_block_impl's doc in matmul_block_helpers.hpp for the full semantics of every knob.
 *
 * The template-parameter order here puts reconfig and Activation BEFORE the functors (right after
 * in1_policy) so the perf-relevant / behavior-selecting knobs stay grouped with the other enums;
 * this wrapper remaps them back to matmul_block_impl's order (functors between in1_policy and
 * Activation) when forwarding. Callers that pass NO functor hooks should use the simple
 * matmul_block wrapper in matmul_block_helpers.hpp instead.
 */
template <
    bool transpose = false,
    bool packer_l1_acc = false,
    LastBlockTarget last_block_target = LastBlockTarget::Out,
    OutputCBLayout tile_order = OutputCBLayout::SubblockMajor,
    matmul_config::InitMode init_mode = matmul_config::InitMode::Short,
    InputPolicy in0_policy = InputPolicy::WaitAndPopPerKBlock,
    InputPolicy in1_policy = InputPolicy::WaitAndPopPerKBlock,
    matmul_config::DataFormatReconfig reconfig = matmul_config::DataFormatReconfig::InputAndOutput,
    typename Activation = NoneActivation,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock,
    typename PostKBlockFn = NoPostKBlock,
    typename KBlockInnerDimFn = NoKBlockInnerDimFn,
    typename In0SourceFn = NoIn0Source,
    typename In1BaseOffsetFn = NoIn1BaseOffset,
    // ── Compile-time block-shape opt-in (perf) — see matmul_block_impl's doc. ─────────
    uint32_t c_in0_num_subblocks = 0,
    uint32_t c_in1_num_subblocks = 0,
    uint32_t c_out_subblock_h = 0,
    uint32_t c_out_subblock_w = 0,
    uint32_t c_in0_block_k = 0,
    uint32_t c_num_k_blocks = 0,
    uint32_t c_batch = 1,
    uint32_t c_last_in1_subblock_w_valid = 0,
    uint32_t c_in1_per_core_w = 0,
    uint32_t c_out_row_width = 0,
    typename Buf = ::CircularBuffer>
ALWI void matmul_block_gathered(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    const MatmulBlockShape& shape,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
    PostKBlockFn post_k_block = {},
    KBlockInnerDimFn k_block_inner_dim = {},
    In0SourceFn in0_source_fn = {},
    In1BaseOffsetFn in1_base_offset_fn = {}) {
    // Remap to matmul_block_impl's template-arg order: functors sit between in1_policy and
    // Activation there; here they follow reconfig + Activation. Runtime functor args pass through.
    matmul_block_impl<
        transpose,
        packer_l1_acc,
        last_block_target,
        tile_order,
        init_mode,
        in0_policy,
        in1_policy,
        PostComputeFn,
        PreKBlockFn,
        PostKBlockFn,
        KBlockInnerDimFn,
        In0SourceFn,
        In1BaseOffsetFn,
        Activation,
        reconfig,
        c_in0_num_subblocks,
        c_in1_num_subblocks,
        c_out_subblock_h,
        c_out_subblock_w,
        c_in0_block_k,
        c_num_k_blocks,
        c_batch,
        c_last_in1_subblock_w_valid,
        c_in1_per_core_w,
        c_out_row_width,
        Buf>(
        in0_buf,
        in1_buf,
        out_buf,
        interm_buf,
        shape,
        post_compute,
        pre_k_block,
        post_k_block,
        k_block_inner_dim,
        in0_source_fn,
        in1_base_offset_fn);
}

}  // namespace compute_kernel_lib
