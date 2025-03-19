// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning work between multiple cores.
//

#pragma once

#include "ttnn/tensor/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {

namespace operations::core::work_split {

struct BlockSplit {
    uint32_t ncores;
    CoreRangeSet all_cores;
    CoreRangeSet core_range;
    CoreRangeSet core_range_cliff;
    uint32_t nblocks_per_core;
    uint32_t nblocks_per_core_cliff;
};

struct BlockSplitWH {
    uint32_t ncores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_range;
    CoreRangeSet cliff_row_core_range;
    CoreRangeSet cliff_col_core_range;
    CoreRangeSet cliff_col_row_core_range;
    uint32_t nblocks_per_core = 0;
    uint32_t single_block_size = 0;
    uint32_t single_block_size_cliff_row = 0;
    uint32_t single_block_size_cliff_col = 0;
    bool has_cliff_row = false;
    bool has_cliff_col = false;
    uint32_t full_cores_per_row = 0;
    uint32_t full_cores_per_col = 0;
};

inline std::pair<int, int> closest_square_larger_than_b(int b, int width, int height, int ref) {
    if (ref <= 0) {
        return {1, 1};
    }

    int sqrt_b = static_cast<int>(std::sqrt(b));

    // Check if b is a perfect square and if it meets the condition using integer arithmetic for ceiling.
    if (sqrt_b * sqrt_b == b) {
        int numX = (width + sqrt_b - 1) / sqrt_b;   // equivalent to ceil(width / sqrt_b)
        int numY = (height + sqrt_b - 1) / sqrt_b;  // equivalent to ceil(height / sqrt_b)
        if (numX * numY < ref) {
            return {b, sqrt_b};
        }
    }

    // Iterate over candidate values, starting from sqrt_b + 1,
    // and check the condition using integer arithmetic.
    for (int candidate = sqrt_b + 1; candidate <= width * height; ++candidate) {
        int square = candidate * candidate;
        int numX = (width + candidate - 1) / candidate;   // equivalent to ceil(width / candidate)
        int numY = (height + candidate - 1) / candidate;  // equivalent to ceil(height / candidate)
        if (numX * numY < ref) {
            return {square, candidate};
        }
    }

    return {1, 1};
}

inline BlockSplitWH split_blocks_for_tilize_wh(
    CoreCoord grid_size, uint32_t nblocks, uint32_t width_tiles, uint32_t height_tiles) {
    // Compute grid area and initial blocks-per-core using integer math.
    const uint32_t grid_area = grid_size.x * grid_size.y;
    uint32_t nblocks_per_core = (grid_area == 0) ? 1 : (nblocks + grid_area - 1) / grid_area;

    // Adjust nblocks_per_core and determine the optimal block size.
    auto [adjusted_nblocks_per_core, single_block_size] =
        closest_square_larger_than_b(nblocks_per_core, width_tiles, height_tiles, grid_area);
    nblocks_per_core = adjusted_nblocks_per_core;

    // Helper lambda for ceiling division.
    auto divCeil = [](uint32_t a, uint32_t b) -> uint32_t { return (a + b - 1) / b; };
    const uint32_t total_blocks_width = divCeil(width_tiles, single_block_size);
    const uint32_t total_blocks_height = divCeil(height_tiles, single_block_size);
    const uint32_t total_blocks = total_blocks_width * total_blocks_height;
    const uint32_t ncores = (nblocks_per_core == 0) ? nblocks : total_blocks;
    const uint32_t ncores_x = grid_size.x;
    const uint32_t ncores_y = (ncores_x == 0) ? 0 : divCeil(ncores, ncores_x);
    // Sets to hold different core ranges.
    std::set<CoreRange> core_range, cliff_col_core_range, cliff_row_core_range, cliff_col_row_core_range;
    std::set<CoreRange> all_cores;
    const uint32_t full_cores_per_row = width_tiles / single_block_size;
    const bool has_cliff_row = (full_cores_per_row < total_blocks_width);
    const uint32_t full_cores_per_col = height_tiles / single_block_size;
    const bool has_cliff_col = (full_cores_per_col < total_blocks_height);
    const uint32_t single_block_size_cliff_row = width_tiles - full_cores_per_row * single_block_size;
    const uint32_t single_block_size_cliff_col = height_tiles - full_cores_per_col * single_block_size;
    // Coordinates for assigning cores sequentially.
    uint32_t i_x = 0;
    uint32_t i_y = 0;
    auto addCore = [&](std::set<CoreRange>& targetSet) {
        CoreRange range{CoreCoord{i_x, i_y}, CoreCoord{i_x, i_y}};
        targetSet.insert(range);
        all_cores.insert(range);
        // Update core coordinates in a cyclic row-wise manner.
        if (i_x == grid_size.x - 1) {
            i_x = 0;
            i_y++;
        } else {
            i_x++;
        }
    };
    // Distribute cores over full rows (each row may have an extra "cliff" block at the end).
    for (uint32_t row = 0; row < full_cores_per_col; row++) {
        for (uint32_t col = 0; col < full_cores_per_row; col++) {
            addCore(core_range);
        }
        if (has_cliff_row) {
            addCore(cliff_row_core_range);
        }
    }
    // Add the cliff column if present.
    if (has_cliff_col) {
        for (uint32_t col = 0; col < full_cores_per_row; col++) {
            addCore(cliff_col_core_range);
        }
        if (has_cliff_row) {
            addCore(cliff_col_row_core_range);
        }
    }
    return BlockSplitWH{
        ncores,
        all_cores,
        core_range,
        cliff_row_core_range,
        cliff_col_core_range,
        cliff_col_row_core_range,
        nblocks_per_core,
        single_block_size,
        single_block_size_cliff_row,
        single_block_size_cliff_col,
        has_cliff_row,
        has_cliff_col,
        full_cores_per_row,
        full_cores_per_col};
}

inline BlockSplit split_blocks_for_tilize(CoreCoord grid_size, uint32_t nblocks) {
    size_t grid_area = grid_size.x * grid_size.y;
    const uint32_t nblocks_per_core = grid_area == 0 ? 1 : std::ceil(static_cast<float>(nblocks) / grid_area);
    const uint32_t ncores = nblocks_per_core == 0 ? nblocks : std::ceil(static_cast<float>(nblocks) / nblocks_per_core);
    const uint32_t nblocks_per_core_cliff = nblocks_per_core == 0 ? 0 : nblocks % nblocks_per_core;
    const uint32_t ncores_x = grid_size.x;
    const uint32_t ncores_y = ncores_x == 0 ? 0 : std::ceil(static_cast<float>(ncores) / ncores_x);
    const uint32_t ncores_x_cliff = ncores - (ncores_y - 1) * ncores_x;

    std::set<CoreRange> core_range, cliff_core_range;
    std::optional<CoreCoord> cliff_core;

    // Top non-cliff range (full rows)
    const uint32_t top_range_end_y = ncores_y - (ncores_x_cliff < ncores_x || nblocks_per_core_cliff > 0);

    if (top_range_end_y > 0) {
        auto range = CoreRange{CoreCoord{0, 0}, CoreCoord{ncores_x - 1, top_range_end_y - 1}};
        core_range.insert(range);
    }

    if (ncores_x_cliff < ncores_x && nblocks_per_core_cliff == 0) {
        // Last partial row (non-cliff)
        auto range = CoreRange{CoreCoord{0, ncores_y - 1}, CoreCoord{ncores_x_cliff - 1, ncores_y - 1}};
        core_range.insert(range);
    } else if (nblocks_per_core_cliff > 0) {
        // Last partial row (excluding last core) and single cliff core
        if (ncores_x_cliff > 1) {  // Add range only if there are cores before the cliff core
            auto range = CoreRange{CoreCoord{0, ncores_y - 1}, CoreCoord{ncores_x_cliff - 2, ncores_y - 1}};
            core_range.insert(range);
        }
        cliff_core = CoreCoord{ncores_x_cliff - 1, ncores_y - 1};
    }

    std::set<CoreRange> all_cores = core_range;

    if (cliff_core.has_value()) {
        cliff_core_range.insert(CoreRange{*cliff_core, *cliff_core});
        if (all_cores.size() == 1) {
            // Cliff core is in a new row, insert it into all_cores
            all_cores.insert(cliff_core_range.begin(), cliff_core_range.end());
        } else {
            // Cliff core is in the same row as the last core range, increment its end
            auto last_range = *all_cores.rbegin();
            auto node = all_cores.extract(last_range);
            node.value().end_coord = *cliff_core;
            all_cores.insert(std::move(node));
        }
    }

    return BlockSplit{ncores, all_cores, core_range, cliff_core_range, nblocks_per_core, nblocks_per_core_cliff};
}

// BlockRep represents a repeated sequence of data blocks, mixed blocks, and padding blocks.
// It is convient to pass to the device kernels because a single data structure made of 4 ints
// can represent pure data rows, pure padding rows or a mixture thereof.
struct BlockRep {
    // number of data blocks
    uint32_t n_data;
    // number of mixed data rows in a mixed block, 0 means no mixed block
    uint32_t n_mixed;
    // number of padding blocks
    uint32_t n_pads;
    // total repeat times
    uint32_t times;

    BlockRep(uint32_t n_data, uint32_t n_mixed, uint32_t n_pads, uint32_t times) :
        n_data(n_data), n_mixed(n_mixed), n_pads(n_pads), times(times) {
        if (n_data == 0 && n_mixed == 0) {
            n_pads *= times;
            times = 1;
        } else if (n_pads == 0 && n_mixed == 0) {
            n_data *= times;
            times = 1;
        }
    }

    bool has_mixed_block() const { return n_mixed > 0; }

    uint32_t single_rep() const { return n_data + has_mixed_block() + n_pads; }

    uint32_t block_count() const { return single_rep() * times; }

    uint32_t data_row_count() const { return (n_data * 32 + n_mixed) * times; }

    std::pair<std::vector<BlockRep>, std::vector<BlockRep>> split_at(uint32_t idx) const {
        // TT_ASSERT(idx <= block_count());

        std::vector<BlockRep> first;
        std::vector<BlockRep> second;

        int rep_idx = idx / single_rep();
        if (rep_idx > 0) {
            first.emplace_back(n_data, n_mixed, n_pads, rep_idx);
        }

        int within_rep_idx = idx % single_rep();
        bool is_within_rep = within_rep_idx > 0;
        if (is_within_rep) {
            if (within_rep_idx <= n_data) {
                first.emplace_back(within_rep_idx, 0, 0, 1);
                second.emplace_back(n_data - within_rep_idx, n_mixed, n_pads, 1);
            } else if (within_rep_idx == n_data + 1 && has_mixed_block()) {
                first.emplace_back(n_data, n_mixed, 0, 1);
                second.emplace_back(0, 0, n_pads, 1);
            } else {
                within_rep_idx -= n_data + has_mixed_block();
                first.emplace_back(n_data, n_mixed, within_rep_idx, 1);
                second.emplace_back(0, 0, n_pads - within_rep_idx, 1);
            }
        }

        int remaining_times = times - rep_idx - is_within_rep;
        if (remaining_times > 0) {
            second.emplace_back(n_data, n_mixed, n_pads, remaining_times);
        }

        return {first, second};
    }
};

// FullRep is a repeated sequence of data rows followed by pure padding. It represents the row
// pattern seen from the outer-most dimension of a 4D tensor when padding is added to the second
// or the thrird dimension.
struct FullRep {
    BlockRep rep;
    BlockRep pad;
    uint32_t times_total;

    FullRep(
        uint32_t n_rows,
        uint32_t n_pads,
        uint32_t times,
        uint32_t pads_mul,
        uint32_t times_total,
        uint32_t tile_height) :
        rep{n_rows / tile_height, n_rows % tile_height, n_pads / tile_height, times},
        pad{0, 0, (n_rows + n_pads) * pads_mul / tile_height, 1},
        times_total(times_total) {
        TT_FATAL((n_rows + n_pads) % tile_height == 0 && "total rows must be divisible by {}", "Error", tile_height);
    }

    std::vector<BlockRep> to_block_reps() const {
        if (pad.n_pads == 0) {
            return std::vector<BlockRep>(times_total, rep);
        }

        std::vector<BlockRep> block_reps;
        block_reps.reserve(2 * times_total);

        for (int i = 0; i < times_total; ++i) {
            block_reps.push_back(rep);
            block_reps.push_back(pad);
        }

        return block_reps;
    }
};

inline bool compare_assignments(const BlockRep& el0, const BlockRep& el1) {
    return (
        el0.n_data == el1.n_data && el0.n_mixed == el1.n_mixed && el0.n_pads == el1.n_pads && el0.times == el1.times);
}

inline std::vector<std::vector<BlockRep>> distribute_work(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    uint32_t num_cores,
    uint32_t blocks_per_core,
    bool has_cliff,
    uint32_t nblocks_per_core_cliff,
    uint32_t tile_height) {
    auto input_w = logical_shape[-4];
    auto input_z = logical_shape[-3];
    auto input_y = logical_shape[-2];

    auto padding_w = padded_shape[-4] - input_w;
    auto padding_z = padded_shape[-3] - input_z;
    auto padding_y = padded_shape[-2] - input_y;

    // total work is a full rep followed by a padding.
    auto full_rep_blocks = FullRep(input_y, padding_y, input_z, padding_z, input_w, tile_height).to_block_reps();
    std::deque<BlockRep> total_work(full_rep_blocks.begin(), full_rep_blocks.end());
    total_work.emplace_back(0, 0, (input_y + padding_y) * (input_z + padding_z) * padding_w, 1);

    std::vector<std::vector<BlockRep>> core_assignments;
    core_assignments.reserve(num_cores);

    for (int i = 0; i < num_cores; i++) {
        int blocks_to_process = blocks_per_core;
        if (i == num_cores - 1 && has_cliff) {
            blocks_to_process = nblocks_per_core_cliff;
        }

        // Assign blocks to cores
        std::vector<BlockRep> core_blocks;
        int core_blocks_count = 0;
        while (core_blocks_count < blocks_to_process) {
            if (total_work.empty()) {
                break;
            }

            int remaining_core_blocks = blocks_to_process - core_blocks_count;
            auto& first = total_work.front();
            if (first.block_count() <= remaining_core_blocks) {
                core_blocks.push_back(first);
                core_blocks_count += first.block_count();
                total_work.pop_front();
            } else {
                auto [head, tail] = first.split_at(remaining_core_blocks);
                for (auto& el : head) {
                    core_blocks.push_back(el);
                    core_blocks_count += el.block_count();
                }
                total_work.pop_front();
                total_work.insert(total_work.begin(), tail.begin(), tail.end());
            }
        }

        core_assignments.push_back(core_blocks);
    }

    return core_assignments;
}

}  // namespace operations::core::work_split

using namespace operations::core::work_split;

}  // namespace ttnn
