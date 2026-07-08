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

// Number of tiles the legacy ROW_MAJOR (TILIZE_IN) compute kernel keeps in L1 for one per-core
// group. Mirrors the kernel/reader num_out_blocks_padded * out_block_h_normal * block_wt accounting
// exactly so the c_17 (cb_in_tilized) circular buffer is sized to hold the full group.
uint32_t groupnorm_tilized_group_tiles(uint32_t block_ht, uint32_t num_out_blocks, uint32_t block_wt);

// Fraction of usable L1 (percent) the legacy ROW_MAJOR tilize-in-L1 fast path is allowed to occupy
// before the program factory falls back to the re-tilize path. Integer percent so the fit check can
// use integer math instead of a floating-point comparison. The <100 margin leaves headroom for the
// small fixed-size CBs that are approximated rather than summed exactly (see kGroupnormSmallCbAllowanceTiles).
inline constexpr uint64_t kGroupnormTilizedL1UsagePercent = 95;

// Conservative allowance (in tiles) for the small fixed-size scalar/reduction CBs
// (in2/in3/in4/ex_partial/ex/ex_global/ex2pe/ex_external) that the L1 footprint estimate does not sum
// individually. 32 tiles (~64 KB at bf16) comfortably covers them across the supported configurations.
inline constexpr uint32_t kGroupnormSmallCbAllowanceTiles = 32;

// Byte sizes of the height-scaling circular buffers for one per-core group under the legacy ROW_MAJOR
// tilize-in-L1 configuration. Used to estimate the per-core L1 footprint of the fast path.
struct GroupNormGroupCbBytes {
    uint32_t in0 = 0;
    uint32_t out = 0;
    uint32_t in = 0;
    uint32_t in_tilized = 0;
    uint32_t x = 0;
    uint32_t xmm = 0;
    uint32_t xmm2 = 0;
    uint32_t xmm3 = 0;
    uint32_t rm_untilize = 0;  // c_30 untilize-output CB; paired with a c_20 reread CB of size `in`
};

// Total per-core L1 bytes for one group's tilize-in-L1 configuration: the height-scaling CBs in `cbs`
// (plus the c_30 output + c_20 reread pair when untilize_out) and the shared scalar/reduction/gamma/
// beta/mask CBs passed in `shared_cb_bytes`.
uint64_t groupnorm_group_l1_bytes(const GroupNormGroupCbBytes& cbs, bool untilize_out, uint64_t shared_cb_bytes);

}  // namespace ttnn::prim
