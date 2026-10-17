// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::prim::detail {

// Widest key chunk, in tiles, that any SDPA compute kernel may hand to the block row-max reduce
// (reduce_block_max_row_init_runtime via reduce_c_row_group). The reduce programs the chunk width
// straight into a MOP loop count and walks the chunk with the unpacker's tile counter, so the LLK
// caps it at 127 on every architecture (its compile-time path static_asserts the same bound; on
// Wormhole the loop count keeps only 7 bits and a wider chunk silently reduces (width & 127) tiles).
// The LLK's own assert is compiled out in production, so every op that reaches that kernel path
// enforces the limit here.
constexpr uint32_t kMaxKChunkTiles = 127;

// Rejects a key-chunk (or key-block) size the block row-max reduce cannot process. `name` is the
// attribute the caller exposes (k_chunk_size, block_size) so the message points at the user's knob.
inline void validate_chunk_fits_block_reduce(uint32_t chunk_size, const char* name) {
    TT_FATAL(
        chunk_size / tt::constants::TILE_WIDTH <= kMaxKChunkTiles,
        "{} must be at most {} ({} tiles): the compute kernel's block row-max reduce supports at most {} tiles on "
        "every architecture (on Wormhole a wider chunk silently reduces the wrong tiles). Got {}: {}",
        name,
        kMaxKChunkTiles * tt::constants::TILE_WIDTH,
        kMaxKChunkTiles,
        kMaxKChunkTiles,
        name,
        chunk_size);
}

}  // namespace ttnn::prim::detail
