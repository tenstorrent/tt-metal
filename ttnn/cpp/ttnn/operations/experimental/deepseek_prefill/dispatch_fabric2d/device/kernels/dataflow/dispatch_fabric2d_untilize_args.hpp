// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The untilizer pool's three kernels, which turn a TILE input into the row-major staging buffer the
// stream cores read tokens from. The transport does not change: a token still arrives at the stream
// core's ring as one page of an interleaved DRAM buffer, and only which buffer that is moves.
//
// A stripe is one tile row of the input -- TILE_HEIGHT tokens, and TILE_HEIGHT pages of staging. The
// reader streams a stripe's tile columns into cb_in `block_ct_dim` at a time, the compute kernel packs
// each of those blocks into cb_out as rows, and the writer puts the rows into staging and tells every
// stream core that one more stripe has landed.
//
// The three roles share ONE compile-time argument layout, built once on the host, because the only
// thing that differs between the pool's cores is which stripes they take. That is `first_stripe`, a
// RUNTIME argument: held compile-time it would build a separate kernel binary per core for nothing.

namespace dspf2d {

struct UntilizeCtArgs {
    enum Idx : uint32_t {
        // Circular buffer indices, unlike the kCb* of the reader's control region, which are blocks
        // of one flat L1 carve.
        kTileCb,  // tiled stripe, reader -> compute
        kRowCb,   // untilized rows, compute -> writer
        kNumStripes,
        kPoolSize,      // stripes are dealt round robin over this many cores
        kTilesPerRow,   // emb_dim / TILE_WIDTH
        kBlockCtDim,    // tile columns per pack_untilize call; must divide kTilesPerRow
        kTileBytes,
        kTokenBytes,     // one untilized row, and one page of staging
        kRowsPerStripe,  // TILE_HEIGHT
        kStreamCount,
        kUntilizeSemAddr,
        // Where the kStreamCount pairs of (virtual x, virtual y) begin -- the stream cores whose
        // counter a landed stripe bumps. A BASE held in an argument, not the index of this argument:
        // the pairs are appended after the whole scalar block, and the TensorAccessorArgs after them.
        kStreamCoordsBase,
        kCount,
    };
};

struct UntilizeRtArg {
    enum Idx : uint32_t {
        // Index 0 for all three roles, which is the whole reason this enum is shared.
        kFirstStripe,
        // Pushed by the reader (the TILE input) and by the writer (the staging buffer) only. The
        // compute kernel addresses no memory and is handed `kFirstStripe` alone, so its runtime-arg
        // list is one word long and this slot is not there to read.
        kBufferAddr,
        kCount,
    };
};

}  // namespace dspf2d
