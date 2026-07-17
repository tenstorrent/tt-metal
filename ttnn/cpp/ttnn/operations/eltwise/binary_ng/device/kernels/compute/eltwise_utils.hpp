// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Reads `per_core_block_size` tiles from cb_pre, runs the per-operand activation chain
// on each tile in DST, and writes the results into cb_post — i.e. produces the
// "activated" input that the downstream binary op consumes. cb_out is passed in only
// so we can briefly retarget the packer at cb_post and then restore it to cb_out's
// data format on the way out. FPU variant: also reconfigures the unpacker srca format
// for the pre/post switch, since the FPU binary op will read from a different CB next.
template <typename ActivationFn>
ALWI void preprocess_fpu_impl(
    CircularBuffer cb_pre,
    CircularBuffer cb_post,
    CircularBuffer cb_out,
    uint32_t per_core_block_size,
    ActivationFn&& process_activations) {
    using namespace ckernel;

    reconfig_data_format_srca(/*old*/ cb_post.get_cb_id(), /*new*/ cb_pre.get_cb_id());
    pack_reconfig_data_format(/*old*/ cb_out.get_cb_id(), /*new*/ cb_post.get_cb_id());

    cb_pre.wait_front(per_core_block_size);
    cb_post.reserve_back(per_core_block_size);

    tile_regs_acquire();
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile_to_dst_init_short(cb_pre.get_cb_id());
        copy_tile(cb_pre.get_cb_id(), i, i);
        process_activations(i);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        pack_tile(i, cb_post.get_cb_id());
    }
    tile_regs_release();

    cb_pre.pop_front(per_core_block_size);
    cb_post.push_back(per_core_block_size);

    reconfig_data_format_srca(/*old*/ cb_pre.get_cb_id(), /*new*/ cb_post.get_cb_id());
    pack_reconfig_data_format(/*old*/ cb_post.get_cb_id(), /*new*/ cb_out.get_cb_id());
}

// Dispatcher: per-operand activation lists are configured at compile time via host-side
// defines (e.g. PROCESS_LHS_ACTIVATIONS, PROCESS_RHS_ACTIVATIONS). HAS_ACTIVATIONS(op)
// expands to 1 or 0; this macro pastes that bit onto PREPROCESS_ to pick the matching
// arm below, so kernels can write a single `PREPROCESS(LHS, ...)` call and have it
// compile to either the activation pass or nothing at all.
#define PREPROCESS(op, ...) P_CAT(PREPROCESS_, HAS_ACTIVATIONS(op))(op, __VA_ARGS__)

// No-op arm: when an operand has no activations the kernel skips the entire preprocess
// step (no extra CB hop, no copy_tile pass) and reads the pre CB directly downstream.
#define PREPROCESS_0(...)

// Active arm: forwards to preprocess_fpu_impl, which performs the
// pre -> activation -> post copy/pack pass plus the FPU srca format reconfigures.
// The macro layer exists only to bind the `op` token (LHS / RHS / BCAST_OP / OTHER_OP)
// so PROCESS_##op##_ACTIVATIONS resolves via preprocessor concatenation — that part
// can't be a function template, since `op` is a token, not a value. The lambda then
// hands the resolved activation sequence to the (real, always-inline) helper.
#define PREPROCESS_1(op, cb_pre, cb_post, cb_out, per_core_block_size) \
    preprocess_fpu_impl(                                               \
        (cb_pre), (cb_post), (cb_out), (per_core_block_size), [&](uint32_t i) { PROCESS_ACTIVATIONS(op, i); })
