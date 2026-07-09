// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

enum class GroupNormMode : uint32_t { LEGACY = 0, WELFORD_NATIVE = 1, WELFORD_RECIPROCALS = 2 };

int get_max_subblock(uint32_t n, uint32_t max_subblock_w);

bool is_rectangle_grid(const std::vector<CoreCoord>& core_coords);

void split_and_form_rectangle_grids(
    std::vector<CoreCoord>& group,
    std::vector<CoreCoord>& mcast_group_first,
    std::vector<CoreCoord>& mcast_group_mid,
    std::vector<CoreCoord>& mcast_group_last);

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width = 32);

// Tiles the legacy ROW_MAJOR (TILIZE_IN) fast path keeps resident in c_17 (cb_in_tilized) for one per-core
// group: num_out_blocks_padded * out_block_h_normal * block_wt, mirroring the kernel/reader accounting.
uint32_t groupnorm_tilized_group_tiles(uint32_t block_ht, uint32_t num_out_blocks, uint32_t block_wt);

// Percent of usable L1 the tilize-in-L1 fast path may occupy before the factory falls back to the
// re-tilize path. Integer percent (integer fit check); the <100 margin covers the approximated small CBs.
inline constexpr uint64_t kGroupnormTilizedL1UsagePercent = 95;

// Flat allowance (in tiles) for the small fixed-size scalar/reduction CBs the fast-path L1 estimate does not
// sum individually. 32 tiles (~64 KB at bf16) comfortably covers them across the supported configurations.
inline constexpr uint32_t kGroupnormSmallCbAllowanceTiles = 32;

}  // namespace ttnn::prim
