// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::prim::detail {

// Widest key chunk, in tiles, that an SDPA compute kernel may hand to the block row-max reduce
// (reduce_block_max_row_init_runtime via reduce_c_row_group on the streaming compute path). The
// reduce programs the chunk width straight into a MOP loop count and walks the chunk with the
// unpacker's tile counter; the Wormhole and Blackhole LLKs cap it at 127 (compile-time static_assert,
// runtime LLK_ASSERT; on Wormhole the loop count keeps only 7 bits and a wider chunk silently reduces
// (width & 127) tiles). The LLK's assert is compiled out in production, so every in-tree ttnn SDPA op
// enforces the limit here on exactly the configurations whose compute kernel takes the block reduce:
// the legacy (non-streaming) kernel path loops reduce_tile over the chunk and has no such bound.
// The experimental Quasar SDPA fork takes the same streaming kernel when it runs on Wormhole or
// Blackhole (on Quasar itself its factory forces the standard kernel), so it enforces the limit on that
// path too. Not covered: the bge_m3 encoder_sdpa KernelDescriptor path onto compute/sdpa.cpp, which
// stays far below 127 tiles.
constexpr uint32_t kMaxKChunkTiles = 127;

// Names the configuration that selects the streaming compute kernel in the ops where the legacy
// kernel is still reachable (SDPA, RingJointSDPA: streaming iff fp32_dest_acc_en is false).
constexpr const char* kStreamingComputeCondition = " with fp32_dest_acc_en=false (the streaming compute kernel)";

// Rejects a key-chunk (or key-block) size the block row-max reduce cannot process. `name` is the
// attribute the caller exposes (k_chunk_size, block_size) so the message points at the user's knob;
// `condition` names the host-visible setting under which the op's compute kernel reaches the block
// reduce (empty when it always does), so the message states exactly which configuration is capped.
inline void validate_chunk_fits_block_reduce(uint32_t chunk_size, const char* name, const char* condition = "") {
    TT_FATAL(
        chunk_size / tt::constants::TILE_WIDTH <= kMaxKChunkTiles,
        "{} must be at most {} ({} tiles){}: the compute kernel's block row-max reduce supports at most {} tiles on "
        "every architecture (on Wormhole a wider chunk silently reduces the wrong tiles). Got {}: {}",
        name,
        kMaxKChunkTiles * tt::constants::TILE_WIDTH,
        kMaxKChunkTiles,
        condition,
        kMaxKChunkTiles,
        name,
        chunk_size);
}

}  // namespace ttnn::prim::detail
