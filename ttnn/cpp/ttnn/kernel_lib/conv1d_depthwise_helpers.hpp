// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"

namespace compute_kernel_lib {

// Shared inner step of a 1D depthwise (FIR) convolution accumulation. Both the conv2d
// HEIGHT_SHARDED depthwise conv1d (conv2d/device/kernels/compute_depthwise_conv1d.cpp) and the
// streaming conv1d_depthwise op accumulate identically per output tile:
//
//   dst[0]  = act_cb[act_idx] * w_cb[w_idx]                              (FPU mul)
//   if (!first_tap): dst[0] += out_cb[0]   (FPU add via DST_TO_SRCB dest-reuse, consumes 1 out tile)
//   pack dst[0] -> out_cb                                               (single pack)
//
// Keeping the running partial in DST and pulling only the prior partial from L1 gives one pack per
// output tile and avoids the pack-format flips that corrupt block-float outputs in a round-tripped
// variant — while still using the FPU (not SFPU) for the add.
//
// srcB is repopulated from DST for the dest-reuse add, so it is forced back to w_cb's format on the
// next call's mul. srcA is set to out_cb for the add and restored to act_cb at the top of the next
// call — so callers must not rely on srcA/srcB format persisting across calls.
//
// The two ops differ only in how they source the operands, which the caller controls via the cb
// ids + indices and by owning the act_cb / w_cb lifetimes (wait_front / pop_front) and the
// per-block loop:
//   - conv2d depthwise: per-channel weight tiles streamed through w_cb (index 0, popped per tile);
//   - streaming op:      K resident scalar tap tiles in w_cb (index = tap, never popped).
inline void depthwise_fir_mac_tile(
    uint32_t act_cb, uint32_t act_idx, uint32_t w_cb, uint32_t w_idx, uint32_t out_cb, bool first_tap) {
    tile_regs_acquire();
    // mul: srcA = act, srcB = weight -> dst[0]
    reconfig_data_format_srca(act_cb);
    reconfig_data_format_srcb(w_cb);
    mul_tiles_init(act_cb, w_cb);
    mul_tiles(act_cb, w_cb, act_idx, w_idx, 0);

    if (!first_tap) {
        // dest-reuse add: dst[0] += out_cb[0]. srcA must match out_cb's format; srcB is filled
        // from dst[0] by the dest-reuse path.
        reconfig_data_format_srca(out_cb);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(out_cb);
        cb_wait_front(out_cb, 1);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(out_cb, 0, 0);
        cb_pop_front(out_cb, 1);
    }
    tile_regs_commit();

    cb_reserve_back(out_cb, 1);
    tile_regs_wait();
    pack_tile(0, out_cb);
    cb_push_back(out_cb, 1);
    tile_regs_release();
}

}  // namespace compute_kernel_lib
