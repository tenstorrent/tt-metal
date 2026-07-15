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

// Tiles the legacy ROW_MAJOR (TILIZE_IN) path keeps resident in c_17 for one per-core group:
// num_out_blocks_padded * out_block_h_normal * block_wt, mirroring the kernel/reader accounting.
uint32_t groupnorm_tilized_group_tiles(uint32_t block_ht, uint32_t num_out_blocks, uint32_t block_wt);

// Percent of usable L1 the resident group may occupy for the host to consider a legacy ROW_MAJOR input to
// "fit"; the <100 margin covers the approximated small CBs.
inline constexpr uint64_t kGroupnormTilizedL1UsagePercent = 95;

// Flat allowance (in tiles) for the small fixed-size scalar/reduction CBs the L1 fit estimate does not
// sum individually.
inline constexpr uint32_t kGroupnormSmallCbAllowanceTiles = 32;

// Host-side estimate of whether ROW_MAJOR interleaved I/O fits in L1 on-core (input tilized resident, output
// untilized on-core). When it does not fit, the host op runs a TILE-only composite instead -- tilizing the
// input and/or untilizing the output around the device op. Mirrors the CB-footprint accounting the program
// factory uses; deliberately conservative (over-estimates the mask CB as bf16) so a "fits" answer is always
// safe to allocate.
// The legacy path keeps one per-group block resident in c_17. The Welford path keeps the whole per-core
// batch resident in c_0/c_29 when tilize_in, and sizes c_16/c_30 to the full per-core width when
// untilize_out; those two costs are included only when the corresponding flag is set. The Welford
// estimate also accounts for the full mask tensor, ex_* reduction CBs, and xmm3 when it is the TILE
// output sink.
// `per_batch_hw` is padded_shape[1] * padded_shape[2] (matches the factory heuristic numerator
// shape[1]*shape[2]*shape[3] when multiplied by W). `available_l1` is usable per-core L1
// (l1_size_per_core minus the base-allocated region); `single_tile_size` is the bf16 tile size in bytes.
// `num_out_blocks_arg` uses the -1 == auto-heuristic convention.
bool groupnorm_rm_input_fits_l1(
    uint32_t Ht,
    uint32_t W,
    uint32_t per_batch_hw,
    uint32_t num_batches,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t num_groups,
    int num_out_blocks_arg,
    uint32_t tile_width,
    uint32_t single_tile_size,
    bool has_gamma,
    bool has_beta,
    bool has_mask,
    bool tilize_in,
    bool untilize_out,
    bool use_welford,
    uint64_t available_l1);

}  // namespace ttnn::prim
