// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DataflowBuffer (DFB) port of kernels/compute/eltwise_utils.hpp. Both native compute kernels use it: in the DFB
// tree the FPU and the SFPU kernel restore the same physical-LHS SrcA format.
//
// Mechanically identical to the CircularBuffer helper, with the CB->DFB swap: the helper takes the kernel's
// DataflowBuffer objects, reads each operand id with get_id(), and syncs through them. See the CB original for
// the rationale on the pre -> activation -> post reconfigure dance.

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/common/operand_walk.hpp"
#if A_ROW || B_ROW || A_HELD || B_HELD
#include "api/compute/bcast.h"
#endif

// The physical-LHS SrcA format: that of the ring a's activation chain writes, else of a's input ring. A held a runs its
// chain while it expands, into the held ring, which has its input ring's format, so it has no post ring.
#if HAS_ACTIVATIONS(LHS) && !A_HELD
#define QSR_BINARY_SRCA_FORMAT_DFB static_cast<uint32_t>(dfb::post_lhs)
#else
#define QSR_BINARY_SRCA_FORMAT_DFB static_cast<uint32_t>(dfb::pre_lhs)
#endif

// Reads `per_core_block_size` tiles from dfb_pre, runs the per-operand activation chain on each tile
// in DST, writes the results into dfb_post. dfb_out's id is used only to briefly retarget the packer
// at dfb_post and then restore it to dfb_out's data format. It also reconfigures the unpacker SrcA
// temporarily, then restores the physical-LHS format for the next pass.
template <typename ActivationFn, typename DFBPre, typename DFBPost>
ALWI void preprocess_impl_dfb(
    DFBPre& dfb_pre,
    DFBPost& dfb_post,
    uint32_t dfb_out_id,
    uint32_t per_core_block_size,
    ActivationFn&& process_activations) {
    using namespace ckernel;

    // dfb_pre and dfb_post are the kernel's own objects. An object built here would drain when it returns and
    // wait for tiles that only this thread pops later.
    const uint32_t dfb_pre_id = dfb_pre.get_id();
    const uint32_t dfb_post_id = dfb_post.get_id();

    reconfig_data_format_srca(/*old*/ QSR_BINARY_SRCA_FORMAT_DFB, /*new*/ dfb_pre_id);
    pack_reconfig_data_format(/*old*/ dfb_out_id, /*new*/ dfb_post_id);
#ifdef ARCH_QUASAR
    // On Quasar pack_reconfig_data_format only reprograms the packer format gasket, not the packer
    // destination ring (reconfig_data_format.h); retarget the packer to dfb_post with pack_init, else
    // pack_tile keeps writing to dfb_out and dfb_post is never written.
    pack_init(dfb_post_id);
#endif

    dfb_pre.wait_front(per_core_block_size);
    dfb_post.reserve_back(per_core_block_size);

    tile_regs_acquire();
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_init(dfb_pre_id);
        copy_tile(dfb_pre_id, i, i);
        process_activations(i);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        pack_tile(i, dfb_post_id);
    }
    tile_regs_release();

    dfb_pre.pop_front(per_core_block_size);
    dfb_post.push_back(per_core_block_size);

    reconfig_data_format_srca(/*old*/ dfb_pre_id, /*new*/ QSR_BINARY_SRCA_FORMAT_DFB);
    pack_reconfig_data_format(/*old*/ dfb_post_id, /*new*/ dfb_out_id);
#ifdef ARCH_QUASAR
    pack_init(dfb_out_id);  // restore the packer destination ring to dfb_out (see above)
#endif
}

#if A_ROW || B_ROW || A_HELD || B_HELD
// Expands tiles 0 to n - 1 of dfb_src, which the caller has waited for, into n entries of dfb_dst with unary_bcast<bt>,
// through DST, and runs process_activations on each expanded tile there. dfb_dst is a self-loop ring of this thread,
// and both are the kernel's own objects. The packer is retargeted to dfb_dst and back to dfb_out_id, as
// preprocess_impl_dfb does. The caller pops dfb_src.
template <BroadcastType bt, bool kActivations, typename DFBSrc, typename DFBDst, typename ActivationFn>
ALWI void expand_dfb(
    DFBSrc& dfb_src, DFBDst& dfb_dst, uint32_t dfb_out_id, uint32_t n, ActivationFn&& process_activations) {
    const uint32_t dfb_src_id = dfb_src.get_id();
    const uint32_t dfb_dst_id = dfb_dst.get_id();

    dfb_dst.reserve_back(n);
    pack_reconfig_data_format(/*old*/ dfb_out_id, /*new*/ dfb_dst_id);
#ifdef ARCH_QUASAR
    pack_init(dfb_dst_id);
#endif
    reconfig_data_format(dfb_src_id, dfb_src_id);
    unary_bcast_init<bt>(dfb_src_id);

    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        if constexpr (kActivations) {
            // An activation's init can change the unpack and math setup, so the broadcast's init runs again, as
            // preprocess_impl_dfb runs copy_init before each tile.
            unary_bcast_init<bt>(dfb_src_id);
        }
        unary_bcast<bt>(dfb_src_id, i, i);
        process_activations(i);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(i, dfb_dst_id);
    }
    tile_regs_release();
    dfb_dst.push_back(n);

    pack_reconfig_data_format(/*old*/ dfb_dst_id, /*new*/ dfb_out_id);
#ifdef ARCH_QUASAR
    pack_init(dfb_out_id);
#endif
}

// The expansion of an operand with no activation chain on the way.
template <BroadcastType bt, typename DFBSrc, typename DFBDst>
ALWI void expand_dfb(DFBSrc& dfb_src, DFBDst& dfb_dst, uint32_t dfb_out_id, uint32_t n) {
    expand_dfb<bt, false>(dfb_src, dfb_dst, dfb_out_id, n, [](uint32_t) {});
}
#endif

#if A_HELD || B_HELD
// One compute batch of a held operand (COL, SCALAR). Its input ring holds n entries for the batch: the first holds the
// operand's new page where `changed`, and the others are credits only. On a change, the page expands into the held
// ring, with the operand's activation chain, and the old held entry is popped. All n input entries are popped, and
// the held entry is left at the front for the binary op. `holding` is the thread's own flag: false before its
// first expansion.
template <BroadcastType bt, bool kActivations, typename DFBIn, typename DFBHeld, typename ActivationFn>
ALWI void hold_dfb(
    DFBIn& dfb_in,
    DFBHeld& dfb_held,
    uint32_t dfb_out_id,
    uint32_t n,
    bool changed,
    bool& holding,
    ActivationFn&& process_activations) {
    dfb_in.wait_front(n);
    if (changed) {
        expand_dfb<bt, kActivations>(dfb_in, dfb_held, dfb_out_id, 1, process_activations);
        if (holding) {
            dfb_held.pop_front(1);
        }
        holding = true;
    }
    dfb_in.pop_front(n);
    dfb_held.wait_front(1);
}

// The walk of compute thread `thread_id` of `num_threads`: its tiles thread_id, thread_id + num_threads, ... of the
// cluster's walk, as the reader's tile counter holds them, with both operands' Maps. The factory sets these runtime
// args, the reader's own, wherever an operand is held.
struct ComputeWalk {
    binary_ng_qsr::WalkStep step;
    binary_ng_qsr::Cursor cursor;
};

ALWI ComputeWalk compute_walk(uint32_t thread_id, uint32_t num_threads) {
    const uint32_t row_tiles = get_arg(args::Wt);
    const binary_ng_qsr::OutputDims dims{
        {get_arg(args::Wt), get_arg(args::Ht), get_arg(args::C), get_arg(args::N), get_arg(args::D), get_arg(args::E)}};
    const binary_ng_qsr::OperandMap map_a{
        {get_arg(args::wt_stride),
         get_arg(args::ht_stride),
         get_arg(args::c_stride),
         get_arg(args::n_stride),
         get_arg(args::d_stride),
         get_arg(args::e_stride),
         get_arg(args::nD_stride)}};
    const binary_ng_qsr::OperandMap map_b{
        {get_arg(args::wt_stride_b),
         get_arg(args::ht_stride_b),
         get_arg(args::c_stride_b),
         get_arg(args::n_stride_b),
         get_arg(args::d_stride_b),
         get_arg(args::e_stride_b),
         get_arg(args::nD_stride_b)}};
#if SHARD_WALK_WIDTH
    const uint32_t col = thread_id % SHARD_WALK_WIDTH;
#else
    const uint32_t col = 0;
#endif
    const uint32_t first_page = binary_ng_qsr::walk_page(
        get_arg(args::start_tile_id), thread_id, SHARD_WALK_WIDTH, row_tiles, BANK_WALK_STRIDE);
    return ComputeWalk{
        binary_ng_qsr::make_walk_step(dims, num_threads, SHARD_WALK_WIDTH, row_tiles, BANK_WALK_STRIDE, map_a, map_b),
        binary_ng_qsr::make_cursor(first_page, col, dims, map_a, map_b)};
}
#endif

// Dispatcher: identical macro structure to the CB original. PREPROCESS(op, dfb_pre, dfb_post, dfb_out,
// n) compiles to either the activation pass or nothing, based on HAS_ACTIVATIONS(op). dfb_pre and
// dfb_post are the kernel's DataflowBuffer objects, and dfb_out is an id.
#define PREPROCESS(op, ...) P_CAT(PREPROCESS_, HAS_ACTIVATIONS(op))(op, __VA_ARGS__)

#define PREPROCESS_0(...)

#define PREPROCESS_1(op, dfb_pre, dfb_post, dfb_out, per_core_block_size) \
    preprocess_impl_dfb(                                                  \
        (dfb_pre), (dfb_post), (dfb_out), (per_core_block_size), [&](uint32_t i) { PROCESS_ACTIVATIONS(op, i); })
