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

// Inputs for the welford ROW_MAJOR/untilize height-blocking decision (shared by the mcast and no_mcast
// factories). Byte fields (x/xmm/xmm2/xmm3, base_shared_cb_bytes) are per-core CB sizes; block_ht_g* are
// per-core group heights in tiles. block_ht_g2 is used only when has_group_2 (no-mcast uneven batches).
struct GroupNormWelfordBlockingParams {
    uint32_t num_out_blocks = 1;  // requested (user/heuristic) height chunking
    uint32_t block_ht_g1 = 0;
    uint32_t block_ht_g2 = 0;
    bool has_group_2 = false;
    uint32_t per_core_Nt = 0;
    uint32_t in_single_tile_size = 0;
    uint32_t out_single_tile_size = 0;
    uint32_t single_tile_size = 0;  // intermediate-format tile (used for the spilled welford-state CB)
    uint32_t x_cb_bytes = 0;
    uint32_t xmm_cb_bytes = 0;
    uint32_t xmm2_cb_bytes = 0;
    uint32_t xmm3_cb_bytes = 0;
    uint64_t base_shared_cb_bytes = 0;  // scalar/reduction/gamma/beta/mask CBs that don't scale with height
    bool untilize_out = false;
    uint64_t available_l1 = 0;
    bool tilize_in = false;
    bool reader_repack_output = false;
};

struct GroupNormWelfordBlocking {
    bool keep_whole_batch = true;  // true: whole batch resident in c_0/c_29 (fast); false: per-out-block fallback
    bool input_fits_l1 = false;    // welford fast path chosen for ROW_MAJOR input (drives the INPUT_FITS_L1 define)
    uint32_t num_out_blocks = 1;   // possibly grown (finer, uniform height blocking) so a fallback out-block fits
};

// Decide welford height-blocking for ROW_MAJOR input / ROW_MAJOR output. Keeps the whole per-core batch
// resident in c_0/c_29 (fast path) when it fits L1 (or is forced, e.g. repack). Otherwise selects the
// smallest num_out_blocks that divides every group's block_ht and makes one out-block fit L1 (the welford
// re-tilize fallback, which for ROW_MAJOR input also spills the 2-tile welford state to L1).
GroupNormWelfordBlocking groupnorm_welford_choose_blocking(const GroupNormWelfordBlockingParams& p);

}  // namespace ttnn::prim
