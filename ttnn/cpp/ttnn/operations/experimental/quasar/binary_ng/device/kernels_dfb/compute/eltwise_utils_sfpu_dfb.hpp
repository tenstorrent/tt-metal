// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DataflowBuffer (DFB) port of kernels/compute/eltwise_utils_sfpu.hpp (SFPU preprocess).
//
// Mechanically identical to the CircularBuffer helper, with the CB->DFB swap. SFPU variant: no
// unpacker srca reconfigure here (the downstream binary SFPU op selects each operand before its
// copy_tile loop). LLK operand ids come from DFBAccessor's `operator uint32_t()`.

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

template <typename ActivationFn>
ALWI void preprocess_sfpu_impl_dfb(
    uint32_t dfb_pre_id,
    uint32_t dfb_post_id,
    uint32_t dfb_out_id,
    uint32_t per_core_block_size,
    ActivationFn&& process_activations) {
    using namespace ckernel;

    DataflowBuffer dfb_pre(dfb_pre_id);
    DataflowBuffer dfb_post(dfb_post_id);

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

    pack_reconfig_data_format(/*old*/ dfb_post_id, /*new*/ dfb_out_id);
#ifdef ARCH_QUASAR
    pack_init(dfb_out_id);  // restore the packer destination ring to dfb_out (see above)
#endif
}

#define PREPROCESS(op, ...) P_CAT(PREPROCESS_, HAS_ACTIVATIONS(op))(op, __VA_ARGS__)

#define PREPROCESS_0(...)

#define PREPROCESS_1(op, dfb_pre, dfb_post, dfb_out, per_core_block_size) \
    preprocess_sfpu_impl_dfb(                                             \
        (dfb_pre), (dfb_post), (dfb_out), (per_core_block_size), [&](uint32_t i) { PROCESS_ACTIVATIONS(op, i); })
