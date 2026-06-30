// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::prim {

enum class GroupNormMode : uint32_t { LEGACY = 0, WELFORD_NATIVE = 1, WELFORD_RECIPROCALS = 2 };

// True when the mask format is safe for the gamma-style partial-row NOC read
// (only face 0 / face 1 row 0 are fetched, rest of the tile is left
// uninitialized). Block-float formats (Bfp2/4/8) store a shared-exponent block
// per face — the data for "logical row 0" doesn't live at face byte offset 0
// — so partial reads are unsafe for them. Plain fp formats (Float16_b,
// Float16, Float32, ...) are safe.
inline bool mask_format_supports_partial_read(tt::DataFormat fmt) {
    switch (fmt) {
        case tt::DataFormat::Bfp2:
        case tt::DataFormat::Bfp2_b:
        case tt::DataFormat::Bfp4:
        case tt::DataFormat::Bfp4_b:
        case tt::DataFormat::Bfp8:
        case tt::DataFormat::Bfp8_b: return false;
        default: return true;
    }
}

// True when the writer kernel can synthesize the per-group 0/1 mask directly
// in L1 instead of fetching it from a host-built DRAM tensor. Requires
// non-block-float dtype (so the on-tile byte layout is the simple
// {face0_row0, face1_row0} = {[0, face_w_bytes), [face_bytes, ...)} map).
//
// Welford is supported: the Welford compute kernels were reordered to do
// `((x − μ) · rsqrt) · mask` instead of `((x − μ) · mask) · rsqrt`, putting
// the mask on the RHS row-source of a mul_tiles_bcast_rows. ROW broadcast
// reads only mask[0, c] — exactly the row-0 bytes the synthesizer writes.
inline bool mask_supports_synthesis(tt::DataFormat fmt, bool /*use_welford*/) {
    return mask_format_supports_partial_read(fmt);
}

int get_max_subblock(uint32_t n, uint32_t max_subblock_w);

bool is_rectangle_grid(const std::vector<CoreCoord>& core_coords);

void split_and_form_rectangle_grids(
    std::vector<CoreCoord>& group,
    std::vector<CoreCoord>& mcast_group_first,
    std::vector<CoreCoord>& mcast_group_mid,
    std::vector<CoreCoord>& mcast_group_last);

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width = 32);

}  // namespace ttnn::prim
