// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Arguments of the untilizer pool's three kernels, which turn a TILE input into a row-major staging
// buffer in DRAM. The stream cores then read tokens from staging as pages of an interleaved buffer.
//
// A tile row is TILE_HEIGHT tokens of the input and TILE_HEIGHT pages of staging. The reader moves a
// tile row into the tile CB `block_ct_dim` tiles at a time, the compute kernel untilizes each block into
// the row CB, and the writer copies the rows to staging and signals every stream core.
//
// Host and kernel index the same enum by name, so the argument order cannot drift between them. The host
// constructor derives every field from the plan and `to_ct_word_arr` serialises them; the kernel
// constructor reads the same fields back. Adding a field is an edit here plus one in the constructor,
// not a synchronised edit across the kernels.
//
// The three kernels share one compile-time layout. Only the tile rows a core takes differ between the
// pool's cores, so `first_tile_row` is a runtime argument and all cores share one binary.

#include <cstdint>

#include "dispatch_fabric2d_kernel_interface.hpp"

#ifndef KERNEL_BUILD
#include <tt-metalium/constants.hpp>

#include "../../dispatch_fabric2d_untilize.hpp"
#endif

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
        // Kept as a base index so a later field can be added without renumbering what the kernels read.
        // The pairs follow the scalars, and the TensorAccessorArgs follow the pairs.
        kStreamCoordsBase,
        kCount,
    };

    uint32_t tile_cb;
    uint32_t row_cb;
    uint32_t num_tile_rows;
    uint32_t pool_size;
    uint32_t tiles_per_row;
    uint32_t block_ct_dim;
    uint32_t tile_bytes;
    uint32_t token_bytes;
    uint32_t rows_per_tile_row;
    uint32_t stream_count;
    uint32_t untilize_sem_addr;
    uint32_t stream_coords_base = kCount;  // the stream coordinates follow the fixed args

#ifndef KERNEL_BUILD
    UntilizeCtArgs(
        const op::UntilizePlan& plan, uint32_t tile_cb_index, uint32_t row_cb_index, uint32_t pool, uint32_t streams) :
        tile_cb(tile_cb_index),
        row_cb(row_cb_index),
        num_tile_rows(plan.num_tile_rows),
        pool_size(pool),
        tiles_per_row(plan.tiles_per_row),
        block_ct_dim(plan.block_ct_dim),
        tile_bytes(plan.tile_bytes),
        token_bytes(plan.token_bytes),
        rows_per_tile_row(tt::constants::TILE_HEIGHT),
        stream_count(streams),
        untilize_sem_addr(plan.sem_addr) {}

    // Scalars, then the stream coordinates at the base index set above.
    std::vector<uint32_t> to_ct_word_arr(const std::vector<uint32_t>& stream_coords) const {
        constexpr uint32_t kUnset = 0xDEADBEEFu;
        std::vector<uint32_t> w(kCount, kUnset);
        w[kTileCb] = tile_cb;
        w[kRowCb] = row_cb;
        w[kNumTileRows] = num_tile_rows;
        w[kPoolSize] = pool_size;
        w[kTilesPerRow] = tiles_per_row;
        w[kBlockCtDim] = block_ct_dim;
        w[kTileBytes] = tile_bytes;
        w[kTokenBytes] = token_bytes;
        w[kRowsPerTileRow] = rows_per_tile_row;
        w[kStreamCount] = stream_count;
        w[kUntilizeSemAddr] = untilize_sem_addr;
        w[kStreamCoordsBase] = stream_coords_base;
        for (uint32_t i = 0; i < kCount; i++) {
            TT_FATAL(w[i] != kUnset, "dispatch_fabric2d: untilizer compile-time arg {} was never assigned", i);
        }
        // The kernels read the coordinates from this base and their accessor arguments after them, so the
        // kernels and this block must agree on its size.
        TT_FATAL(
            stream_coords.size() == 2u * stream_count,
            "dispatch_fabric2d: untilizer stream coordinates are {} words but the kernels index {}",
            stream_coords.size(),
            2u * stream_count);
        w.insert(w.end(), stream_coords.begin(), stream_coords.end());
        return w;
    }
#else
    constexpr UntilizeCtArgs() :
        tile_cb(get_compile_time_arg_val(kTileCb)),
        row_cb(get_compile_time_arg_val(kRowCb)),
        num_tile_rows(get_compile_time_arg_val(kNumTileRows)),
        pool_size(get_compile_time_arg_val(kPoolSize)),
        tiles_per_row(get_compile_time_arg_val(kTilesPerRow)),
        block_ct_dim(get_compile_time_arg_val(kBlockCtDim)),
        tile_bytes(get_compile_time_arg_val(kTileBytes)),
        token_bytes(get_compile_time_arg_val(kTokenBytes)),
        rows_per_tile_row(get_compile_time_arg_val(kRowsPerTileRow)),
        stream_count(get_compile_time_arg_val(kStreamCount)),
        untilize_sem_addr(get_compile_time_arg_val(kUntilizeSemAddr)),
        stream_coords_base(get_compile_time_arg_val(kStreamCoordsBase)) {}

    // The program factory appends the TensorAccessorArgs after the stream coordinates. Derived from the
    // block base, so adding a scalar keeps it right. The compute kernel has no accessor and does not
    // include the accessor header, so the two dataflow kernels instantiate TensorAccessorArgs themselves.
    static constexpr uint32_t accessor_base =
        get_compile_time_arg_val(kStreamCoordsBase) + 2u * get_compile_time_arg_val(kStreamCount);
#endif

    // Whole blocks only: pack_untilize puts block i at column offset i * block_ct_dim, so a smaller last
    // block would overlap the one before it.
    constexpr uint32_t num_blocks() const { return tiles_per_row / block_ct_dim; }
};

// Buffer bindings and the core's share of the work, rewritten by the framework per dispatch.
struct UntilizeRtArg {
    enum Idx : uint32_t {
        kFirstTileRow,  // all three kernels
        // Reader: the TILE input. Writer: the staging buffer. The compute kernel does not get this arg.
        kBufferAddr,
    };
};

}  // namespace dspf2d
