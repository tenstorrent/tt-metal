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

uint64_t groupnorm_group_l1_bytes(const GroupNormGroupCbBytes& cbs, bool untilize_out, uint64_t shared_cb_bytes) {
    uint64_t bytes = static_cast<uint64_t>(cbs.in0) + cbs.out + cbs.in + cbs.in_tilized + cbs.x + cbs.xmm + cbs.xmm2 +
                     cbs.xmm3 + shared_cb_bytes;
    if (untilize_out) {
        bytes += cbs.rm_untilize + cbs.in;  // c_30 output + c_20 reread scratch
    }
    return bytes;
}

GroupNormWelfordBlocking groupnorm_welford_choose_blocking(const GroupNormWelfordBlockingParams& p) {
    // ROW_MAJOR input spills a 2-tile welford state to L1 in the fallback (no c_20 reread here — welford
    // folds the c_30 output into shared bytes below manually, so groupnorm_group_l1_bytes is called with
    // untilize_out=false).
    const uint32_t welford_state_bytes = p.tilize_in ? p.single_tile_size * 2 : 0;

    // Per-core L1 footprint (bytes) with `intake_tiles` resident on c_0/c_29 and per-out-block height `out_bh`.
    const auto footprint = [&](uint32_t out_bh, uint32_t intake_tiles, uint32_t state_bytes) {
        const uint64_t outblk_bytes = static_cast<uint64_t>(out_bh) * p.per_core_Nt * p.in_single_tile_size;
        const uint64_t shared = p.base_shared_cb_bytes + state_bytes + (p.untilize_out ? outblk_bytes : 0);
        return groupnorm_group_l1_bytes(
            {.in0 = intake_tiles * p.in_single_tile_size,
             .out = out_bh * p.per_core_Nt * p.out_single_tile_size,
             .in = intake_tiles * p.in_single_tile_size,
             .in_tilized = 0,
             .x = p.x_cb_bytes,
             .xmm = p.xmm_cb_bytes,
             .xmm2 = p.xmm2_cb_bytes,
             .xmm3 = p.xmm3_cb_bytes,
             .rm_untilize = 0},
            /*untilize_out=*/false,
            shared);
    };
    const auto fits = [&](uint64_t bytes) { return bytes * 100 <= p.available_l1 * kGroupnormTilizedL1UsagePercent; };
    // Both per-core groups live on different cores, so each must fit independently.
    const auto both_groups_fit = [&](uint32_t nob, uint32_t intake_g1, uint32_t intake_g2, uint32_t state_bytes) {
        bool ok = fits(footprint(p.block_ht_g1 / nob, intake_g1, state_bytes));
        if (p.has_group_2) {
            ok = ok && fits(footprint(p.block_ht_g2 / nob, intake_g2, state_bytes));
        }
        return ok;
    };

    const uint32_t batch_g1 = p.block_ht_g1 * p.per_core_Nt;
    const uint32_t batch_g2 = p.block_ht_g2 * p.per_core_Nt;
    const bool whole_batch_fits = both_groups_fit(p.num_out_blocks, batch_g1, batch_g2, 0);

    GroupNormWelfordBlocking r;
    r.input_fits_l1 = p.tilize_in && !p.reader_repack_output && whole_batch_fits;
    // A repack config always keeps the whole batch (no fallback wired); a tiled input keeps it resident
    // only to let the reader run ahead.
    r.keep_whole_batch = r.input_fits_l1 || p.reader_repack_output || (!p.tilize_in && whole_batch_fits);
    r.num_out_blocks = p.num_out_blocks;
    if (r.keep_whole_batch) {
        return r;
    }

    // Fallback: a welford out-block spans the full per-core channel width (per_core_Nt), so the requested
    // num_out_blocks may be too coarse; grow it (snapped to a divisor of every group's block_ht so all
    // out-blocks are uniform) until a single out-block fits every group.
    const uint32_t max_nob = p.has_group_2 ? std::min(p.block_ht_g1, p.block_ht_g2) : p.block_ht_g1;
    const auto divides_all = [&](uint32_t d) {
        return p.block_ht_g1 % d == 0 && (!p.has_group_2 || p.block_ht_g2 % d == 0);
    };
    const auto ceil_to_divisor = [&](uint32_t n) {
        uint32_t d = std::min(n, max_nob);
        while (d < max_nob && !divides_all(d)) {
            d++;
        }
        return d;
    };
    r.num_out_blocks = ceil_to_divisor(p.num_out_blocks);
    while (r.num_out_blocks < max_nob && !both_groups_fit(
                                             r.num_out_blocks,
                                             (p.block_ht_g1 / r.num_out_blocks) * p.per_core_Nt,
                                             (p.block_ht_g2 / r.num_out_blocks) * p.per_core_Nt,
                                             welford_state_bytes)) {
        r.num_out_blocks = ceil_to_divisor(r.num_out_blocks + 1);
    }
    return r;
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
