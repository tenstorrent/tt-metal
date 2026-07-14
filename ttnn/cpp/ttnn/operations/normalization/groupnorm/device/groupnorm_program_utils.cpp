// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_program_utils.hpp"

#include <limits>
#include <algorithm>

namespace ttnn::prim {

uint32_t groupnorm_tilized_group_tiles(uint32_t block_ht, uint32_t num_out_blocks, uint32_t block_wt) {
    // Matches the kernel's num_out_blocks_padded computation (reader_mcast_*_unary_gn.cpp /
    // compute/groupnorm.cpp): each out-block contributes out_block_h_normal * block_wt tiles, and a
    // trailing remainder becomes one (or more) extra padded out-blocks of the normal size.
    const uint32_t out_block_h_normal = block_ht / num_out_blocks;
    uint32_t num_out_blocks_padded = num_out_blocks;
    if (block_ht % num_out_blocks != 0) {
        const uint32_t residual = block_ht - num_out_blocks * out_block_h_normal;
        num_out_blocks_padded += residual / out_block_h_normal + 1;
    }
    return num_out_blocks_padded * out_block_h_normal * block_wt;
}

bool groupnorm_legacy_rm_input_fits_l1(
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
    bool untilize_out,
    uint64_t available_l1) {
    // Grid geometry, replicating the program factory (identical formulas in the mcast and no-mcast factories).
    uint32_t num_virtual_cols = std::min(grid_x, num_groups);
    while (num_virtual_cols > 0 && ((W / num_virtual_cols) % tile_width != 0 || (num_groups % num_virtual_cols) != 0)) {
        num_virtual_cols -= 1;
    }
    if (num_virtual_cols == 0) {
        // Invalid grid: treat as "does not fit" so the host takes the TILE composite path rather than
        // attempting an on-core resident tilize that the factory would reject.
        return false;
    }
    const uint32_t num_virtual_rows = (grid_x / num_virtual_cols) * grid_y;
    if (num_virtual_rows == 0 || Ht < num_virtual_rows) {
        return false;
    }
    const uint32_t per_core_Mt = Ht / num_virtual_rows;
    const uint32_t per_core_N = W / num_virtual_cols;
    const uint32_t per_core_Nt = (per_core_N + tile_width - 1) / tile_width;
    const uint32_t num_channels_per_group = W / num_groups;

    const auto [block_wt, unused_num_groups_per_reset] = find_max_tile_span(per_core_N, num_channels_per_group);
    (void)unused_num_groups_per_reset;
    // Per-core per-batch tile height. Matches factory: when num_batches > num_shards_r (== num_virtual_rows)
    // each core holds multiple batches (block_ht = Ht/num_batches); otherwise a batch is split across rows
    // (block_ht = per_core_Mt). Equality yields the same value either way.
    const uint32_t block_ht = (num_batches >= num_virtual_rows) ? (Ht / num_batches) : per_core_Mt;
    if (block_ht == 0) {
        return false;
    }

    // Resolve num_out_blocks: -1 == the factory's power-of-two heuristic, else the explicit value.
    uint32_t num_out_blocks;
    if (num_out_blocks_arg < 0) {
        const uint32_t HEURISTIC_BLOCK_SIZE_BASE = 256 * 256;
        const uint32_t MAX_HEURISTIC_NUM_OUT_BLOCKS = 256;
        // Factory: (shape[1] * shape[2] * shape[3]) / (BASE * num_virtual_cores)
        uint32_t heuristic = (per_batch_hw * W) / (HEURISTIC_BLOCK_SIZE_BASE * (num_virtual_cols * num_virtual_rows));
        heuristic = heuristic ? heuristic : 1;
        num_out_blocks = 1;
        while (num_out_blocks < heuristic && num_out_blocks < MAX_HEURISTIC_NUM_OUT_BLOCKS) {
            num_out_blocks <<= 1;
        }
    } else {
        num_out_blocks = num_out_blocks_arg == 0 ? 1 : static_cast<uint32_t>(num_out_blocks_arg);
    }
    if (num_out_blocks > block_ht) {
        // Factory TT_FATALs when the explicit value exceeds block_ht; for the auto heuristic we clamp so
        // the estimate still reflects the largest CB footprint the factory could actually allocate.
        num_out_blocks = block_ht;
    }

    // CB footprint, mirroring the factory's resident-path estimate (bf16 legacy). Seven per-out-block CBs
    // (in0/in/out/x/xmm/xmm2/xmm3), the resident tilized group (c_17), a flat small-CB allowance, plus
    // gamma/beta/mask and the c_30/c_20 scratch when the output is ROW_MAJOR.
    const uint64_t in0_block_tiles = static_cast<uint64_t>(block_ht / num_out_blocks) * block_wt;
    const uint64_t per_out_block_cb = in0_block_tiles * single_tile_size;
    const uint64_t resident_cb =
        static_cast<uint64_t>(groupnorm_tilized_group_tiles(block_ht, num_out_blocks, block_wt)) * single_tile_size;

    uint64_t est =
        resident_cb + 7 * per_out_block_cb + static_cast<uint64_t>(kGroupnormSmallCbAllowanceTiles) * single_tile_size;
    if (has_gamma) {
        est += static_cast<uint64_t>(per_core_Nt) * single_tile_size;
    }
    if (has_beta) {
        est += static_cast<uint64_t>(per_core_Nt) * single_tile_size;
    }
    if (has_mask) {
        // Legacy mask CB is block_wt * in_mask_tile * 2; over-estimate the mask tile as bf16 (conservative).
        est += static_cast<uint64_t>(block_wt) * single_tile_size * 2;
    }
    if (untilize_out) {
        est += 2 * per_out_block_cb;  // c_30 (untilize out) + c_20 (row-major reread), each one out-block
    }

    return est * 100 <= available_l1 * kGroupnormTilizedL1UsagePercent;
}

int get_max_subblock(uint32_t n, uint32_t max_subblock_w) {
    if (n <= max_subblock_w) {
        return n;
    }

    for (int quotient = max_subblock_w; quotient > 1; --quotient) {
        if (n % quotient == 0) {
            return quotient;
        }
    }
    return 1;
}

bool is_rectangle_grid(const std::vector<CoreCoord>& core_coords) {
    if (core_coords.empty()) {
        return true;
    }

    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();

    for (const auto& coord : core_coords) {
        min_x = std::min(min_x, static_cast<int>(coord.x));
        max_x = std::max(max_x, static_cast<int>(coord.x));
        min_y = std::min(min_y, static_cast<int>(coord.y));
        max_y = std::max(max_y, static_cast<int>(coord.y));
    }

    return ((max_x - min_x + 1) * (max_y - min_y + 1)) == static_cast<int>(core_coords.size());
}

void split_and_form_rectangle_grids(
    std::vector<CoreCoord>& group,
    std::vector<CoreCoord>& mcast_group_first,
    std::vector<CoreCoord>& mcast_group_mid,
    std::vector<CoreCoord>& mcast_group_last) {
    size_t remove_front = 0;
    size_t remove_back = 0;
    size_t min_total_removal = group.size();

    for (size_t front = 0; front <= group.size(); ++front) {
        for (size_t back = 0; front + back <= group.size(); ++back) {
            if (is_rectangle_grid(std::vector<CoreCoord>(group.begin() + front, group.end() - back))) {
                size_t total_removal = front + back;
                if (total_removal < min_total_removal) {
                    min_total_removal = total_removal;
                    remove_front = front;
                    remove_back = back;
                }
            }
        }
    }

    // Pop and push the front outliers
    for (size_t i = 0; i < remove_front; ++i) {
        mcast_group_first.push_back(mcast_group_mid.front());
        mcast_group_mid.erase(mcast_group_mid.begin());
    }

    // Pop and push the back outliers
    for (size_t i = 0; i < remove_back; ++i) {
        mcast_group_last.push_back(mcast_group_mid.back());
        mcast_group_mid.pop_back();
    }
}

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width) {
    uint32_t current_position = 0;
    uint32_t max_tile_span = 0;
    uint32_t num_groups_before_start_again_at_tile_beginning = static_cast<uint32_t>(-1);
    bool calc_num_groups_before_start_again_at_tile_beginning = true;

    while (current_position < W) {
        uint32_t group_end = current_position + group_size;
        uint32_t start_tile = current_position / tile_width;
        uint32_t end_tile = (group_end - 1) / tile_width;
        uint32_t current_tile_span = end_tile - start_tile + 1;

        max_tile_span = std::max(max_tile_span, current_tile_span);

        current_position = group_end;

        if (current_position % tile_width == 0 && calc_num_groups_before_start_again_at_tile_beginning) {
            num_groups_before_start_again_at_tile_beginning = current_position / group_size;
            calc_num_groups_before_start_again_at_tile_beginning = false;
        }
    }

    return {max_tile_span, num_groups_before_start_again_at_tile_beginning};
}

}  // namespace ttnn::prim
