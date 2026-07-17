// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DataflowBuffer (DFB) port of kernels/compute/eltwise_utils.hpp (FPU preprocess).
//
// Mechanically identical to the CircularBuffer helper, with the CB->DFB swap: the LLK operand id
// comes from DFBAccessor's `operator uint32_t()` (a `dfb::<name>` token converts directly to the cb
// id every LLK call takes), and FIFO sync goes through a DataflowBuffer instance. See the CB original
// for the rationale on the pre -> activation -> post reconfigure dance.

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

// Reads `per_core_block_size` tiles from dfb_pre, runs the per-operand activation chain on each tile
// in DST, writes the results into dfb_post. dfb_out's id is used only to briefly retarget the packer
// at dfb_post and then restore it to dfb_out's data format. FPU variant: also reconfigures the
// unpacker srca format for the pre/post switch.
template <typename ActivationFn>
ALWI void preprocess_fpu_impl_dfb(
    uint32_t dfb_pre_id,
    uint32_t dfb_post_id,
    uint32_t dfb_out_id,
    uint32_t per_core_block_size,
    ActivationFn&& process_activations) {
    using namespace ckernel;

    DataflowBuffer dfb_pre(dfb_pre_id);
    DataflowBuffer dfb_post(dfb_post_id);

    reconfig_data_format_srca(/*old*/ dfb_post_id, /*new*/ dfb_pre_id);
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
        copy_tile_to_dst_init_short(dfb_pre_id);
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

    reconfig_data_format_srca(/*old*/ dfb_pre_id, /*new*/ dfb_post_id);
    pack_reconfig_data_format(/*old*/ dfb_post_id, /*new*/ dfb_out_id);
#ifdef ARCH_QUASAR
    pack_init(dfb_out_id);  // restore the packer destination ring to dfb_out (see above)
#endif
}

// Dispatcher: identical macro structure to the CB original. PREPROCESS(op, dfb_pre, dfb_post, dfb_out,
// n) compiles to either the activation pass or nothing, based on HAS_ACTIVATIONS(op). The dfb args
// are `dfb::<name>` accessor tokens (which convert to uint32_t ids).
#define PREPROCESS(op, ...) P_CAT(PREPROCESS_, HAS_ACTIVATIONS(op))(op, __VA_ARGS__)

#define PREPROCESS_0(...)

#define PREPROCESS_1(op, dfb_pre, dfb_post, dfb_out, per_core_block_size) \
    preprocess_fpu_impl_dfb(                                              \
        (dfb_pre), (dfb_post), (dfb_out), (per_core_block_size), [&](uint32_t i) { PROCESS_ACTIVATIONS(op, i); })
