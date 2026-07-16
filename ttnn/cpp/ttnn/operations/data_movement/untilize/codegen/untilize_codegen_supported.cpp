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

// Correctness scope of the ported build_untilize_tile builder: TILE input, interleaved
// (non-sharded) input AND requested output memory config, dtype in the nightly sweep's coverage
// (bfloat16, bfloat8_b), and -- critically -- a tile-aligned logical shape.
//
// The codegen *reference* (ops/untilize/untilize.py) redirects ANY non-tile-aligned logical shape
// (H % TILE_H != 0 || W % TILE_W != 0) to build_untilize_with_unpadding -- a separate builder entry
// point this port does not implement -- whose output strips the tile padding down to the logical
// shape. build_untilize_tile instead emits the padded row-major output, so it cannot reproduce the
// reference for a non-tile-aligned shape; those must route to native. This transcribes the
// reference's own H%TILE_H / W%TILE_W redirect (untilize.py) as the general rejection condition; the
// manifest's [1,1,33,64] scope:out case is one representative of it, not the whole rule.
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

    // Non-tile-aligned logical shapes are out of scope (see file comment): the reference serves them
    // via build_untilize_with_unpadding, which this port does not implement. Gate on the general
    // condition, never a literal shape tuple.
    const auto& logical = input.logical_shape();
    if (logical[-2] % tt::constants::TILE_HEIGHT != 0 || logical[-1] % tt::constants::TILE_WIDTH != 0) {
        return false;
    }

    // Mirrors ops/untilize/untilize.py's wide-tensor guard: a multi-tile-row input wide enough that a
    // single tile-row would overflow the slice+concat chunking threshold (~800KB for two
    // double-buffered CBs at 2048B/tile, the only tile size in this bf16/bf8_b scope) is routed to a
    // slice -> untilize -> concat cascade over unrelated builder entries, out of scope here. Computed
    // from the PADDED shape (Wt/Ht are physical, tile-grid quantities), matching how the program
    // factory itself derives Wt/total_tile_rows.
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
    constexpr uint32_t kTileSize = 2048;  // bf16/bf8_b only in this scope
    constexpr uint64_t kWideChunkThreshold = 800'000;
    if (total_tile_rows > 1 && 2ull * wt * kTileSize > kWideChunkThreshold) {
        return false;
    }

    return true;
}

// Perf-demote ledger (nightly suite, device-measured): correct under codegen but does not beat
// native. Empty for untilize: every shape the nightly qualification flagged as a "demotion" was
// non-tile-aligned -- i.e. scope:out (served by the reference's build_untilize_with_unpadding, not
// build_untilize_tile), so it is already rejected by supported_by_codegen's tile-alignment gate and
// routes to native there. None was a genuine tile-aligned, codegen-correct-but-slower case (the only
// thing this perf gate is for; a demotion must be a subset of scope:in per review-checklist R12).
// Kept as the routing extension point should a future tile-aligned device regression surface.
bool is_demoted(const Tensor& /*input*/, const tt::tt_metal::MemoryConfig& /*output_mem_config*/) { return false; }

}  // namespace ttnn::operations::data_movement::untilize_codegen
