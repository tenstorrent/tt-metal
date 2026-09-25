// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Arguments of the untilizer pool's three kernels, which turn a TILE input into a row-major staging
// buffer in DRAM. The stream cores then read tokens from staging as pages of an interleaved buffer.
//
// A tile row is TILE_HEIGHT tokens of the input and TILE_HEIGHT pages of staging. The reader moves a
// tile row into cb_in `block_ct_dim` tiles at a time, the compute kernel untilizes each block into
// cb_out, and the writer copies the rows to staging and signals every stream core.
//
// The three kernels share one compile-time layout. Only the tile rows a core takes differ between the
// pool's cores, so `first_tile_row` is a runtime argument and all cores share one binary.

namespace dspf2d {

struct UntilizeCtArgs {
    enum Idx : uint32_t {
        // Circular buffer indices.
        kTileCb,  // tiled tile row, reader -> compute
        kRowCb,   // untilized rows, compute -> writer
        kNumTileRows,
        kPoolSize,     // tile rows are dealt round robin over this many cores
        kTilesPerRow,  // emb_dim / TILE_WIDTH
        kBlockCtDim,   // tile columns per pack_untilize call; must divide kTilesPerRow
        kTileBytes,
        kTokenBytes,      // one untilized row, and one page of staging
        kRowsPerTileRow,  // TILE_HEIGHT
        kStreamCount,
        kUntilizeSemAddr,
        // Holds the index where the kStreamCount (virtual x, virtual y) pairs of the stream cores begin.
        // The pairs follow the scalars, and the TensorAccessorArgs follow the pairs.
        kStreamCoordsBase,
        kCount,
    };
};

struct UntilizeRtArg {
    enum Idx : uint32_t {
        kFirstTileRow,  // all three kernels
        // Reader: the TILE input. Writer: the staging buffer. The compute kernel does not get this arg.
        kBufferAddr,
    };
};

}  // namespace dspf2d
