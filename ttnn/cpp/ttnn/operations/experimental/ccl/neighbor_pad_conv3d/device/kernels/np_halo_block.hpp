// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

// A conv3d output block is a halo_last "boundary" block when any of its (h,w) edges touch the device
// output boundary — for same-conv pad-1 those are exactly the blocks whose receptive field needs the
// cross-device halo. The reader, writer, and compute walk blocks in the same (phase,t,h,w) order, so
// this predicate MUST be identical in the reader and writer to keep their two-phase block counts in
// lock-step (a mismatch desyncs the per-block reduction handshake -> hang).
inline bool np_is_boundary_block(
    uint32_t h_block, uint32_t h_block_end, uint32_t w_block, uint32_t w_block_end, uint32_t H_out, uint32_t W_out) {
    return (h_block == 0u) || (h_block_end >= H_out) || (w_block == 0u) || (w_block_end >= W_out);
}
