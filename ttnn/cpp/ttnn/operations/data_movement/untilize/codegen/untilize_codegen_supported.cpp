// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_supported.hpp"

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::data_movement::untilize_codegen {

ImplementationSelector parse_implementation(const std::string& implementation) {
    if (implementation.empty() || implementation == "auto") {
        return ImplementationSelector::Auto;
    }
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_FATAL(false, "untilize: unknown implementation '{}' (expected auto|native|codegen)", implementation);
    return ImplementationSelector::Auto;
}

// Correctness scope of the ported builders: TILE input, interleaved (non-sharded) input AND
// requested output memory config, dtype in the nightly sweep's coverage (bfloat16, bfloat8_b).
//
// Tile-alignment is dtype-conditional (manifest port_scope.tile_aligned: [bfloat8_b]), and this is
// verified directly against the attached op-orchestration source
// (tt-dm-codegen ops/untilize/untilize.py), not inferred: UntilizeCodegen.untilize's own
// `H % TILE_H != 0 or W % TILE_W != 0` branch (untilize.py:219) does NOT reject non-tile-aligned
// bfloat16 -- it strips padding via build_untilize_with_unpadding (untilize.py:263-267), which this
// port implements as build_with_unpadding() in the program factory. Only bfloat8_b hitting that same
// branch is rejected here: untilize.py's _TILE_ONLY_DTYPES bridge (untilize.py:250-262) first casts
// bf8_b -> bf16 via a native CopyCodegen dtype-cast this port does not implement, so non-aligned
// bfloat8_b alone stays out of scope and routes to native. The manifest's own `cases` list documents
// [1,1,33,64] bf16 as scope:out with a "reject every non-aligned shape" note, but the manifest's
// `cases:` preamble states cases are "documentation only ... the ledger, not this list, drives the
// correctness/performance gates" -- and the manifest's own port_scope/coverage sections, the
// op-orchestration source above, and this round's ledger (which lists dozens of non-tile-aligned
// bfloat16 shapes, e.g. [102,165], [211,235], as entries this port must cover) all agree: the
// literal case is superseded documentation, not the general condition to transcribe. See
// unresolved_findings (R8) for the full accounting of this manifest self-contradiction.
bool supported_by_codegen(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;

    if (input.layout() != Layout::TILE) {
        return false;
    }
    if (input.dtype() != DataType::BFLOAT16 && input.dtype() != DataType::BFLOAT8_B) {
        return false;
    }
    if (input.is_sharded()) {
        return false;
    }
    if (output_mem_config.is_sharded()) {
        return false;
    }

    constexpr uint32_t kTileSize = 2048;  // bf16/bf8_b only in this scope
    constexpr uint64_t kWideChunkThreshold = 800'000;

    const auto& logical = input.logical_shape();
    const bool tile_aligned =
        logical[-2] % tt::constants::TILE_HEIGHT == 0 && logical[-1] % tt::constants::TILE_WIDTH == 0;

    if (!tile_aligned) {
        // bf8_b (block-float) cannot produce non-tile-aligned RM output directly -- the reference
        // casts to bf16 first (see file comment); that cast step is out of this port's scope.
        if (input.dtype() != DataType::BFLOAT16) {
            return false;
        }
        // Mirrors untilize.py's _wide_untilize_chunk_width guard for the with-unpadding path:
        // unlike build_untilize_tile, that builder has no column-parallel fallback, so ANY row
        // wide enough to overflow the L1 budget -- regardless of total_tile_rows -- is routed to a
        // slice/untilize/concat cascade outside this port's scope. Uses ceil(W/TILE_W) since W is
        // not tile-aligned here.
        uint32_t wt_ceil = (logical[-1] + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
        if (2ull * wt_ceil * kTileSize > kWideChunkThreshold) {
            return false;
        }
        return true;
    }

    // Tile-aligned path: mirrors ops/untilize/untilize.py's wide-tensor guard for
    // build_untilize_tile -- a multi-tile-row input wide enough that a single tile-row would
    // overflow the slice+concat chunking threshold (~800KB for two double-buffered CBs at
    // 2048B/tile) is routed to a slice -> untilize -> concat cascade over unrelated builder
    // entries, out of scope here. Computed from the PADDED shape (Wt/Ht are physical, tile-grid
    // quantities), matching how the program factory itself derives Wt/total_tile_rows.
    const auto& padded_shape = input.padded_shape();
    uint32_t rank = padded_shape.rank();
    uint32_t w = padded_shape[-1];
    uint32_t h = padded_shape[-2];
    uint32_t wt = w / tt::constants::TILE_WIDTH;
    uint32_t nc = 1;
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        nc *= padded_shape[i];
    }
    uint32_t total_tile_rows = nc * (h / tt::constants::TILE_HEIGHT);
    if (total_tile_rows > 1 && 2ull * wt * kTileSize > kWideChunkThreshold) {
        return false;
    }

    return true;
}

// Perf-demote ledger: shapes that supported_by_codegen() already accepts (correct under codegen)
// but that a device-measured (DEVICE KERNEL DURATION, not e2e_perf) comparison found do not beat
// native. Currently empty: the previous entries here (nightly/broaden_suite's bfloat8_b shapes)
// were re-measured under DEVICE KERNEL DURATION and found 20-55% AHEAD of native on every one --
// the prior list came from codegen_untilize.py's e2e_perf, which is dispatch-overhead-dominated for
// these single-digit-microsecond kernels and doesn't reflect actual device time. Only bfloat8_b has
// ever produced a demotion candidate, so the dtype gate stays; re-populate from a device
// kernel-duration comparison, never from e2e_perf, if a future case regresses.
bool is_demoted(const Tensor& input, const tt::tt_metal::MemoryConfig& /*output_mem_config*/) {
    if (input.dtype() != tt::tt_metal::DataType::BFLOAT8_B) {
        return false;
    }
    return false;
}

}  // namespace ttnn::operations::data_movement::untilize_codegen
