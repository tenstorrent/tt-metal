// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning work between multiple cores.
//

#pragma once

#include "ttnn/tensor/types.hpp"
#include "tt_metal/common/core_coord.h"

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

inline BlockSplit split_blocks_for_tilize(CoreCoord grid_size, uint32_t nblocks) {
    const uint32_t nblocks_per_core = std::ceil(static_cast<float>(nblocks) / (grid_size.x * grid_size.y));
    const uint32_t ncores = std::ceil(static_cast<float>(nblocks) / nblocks_per_core);
    const uint32_t nblocks_per_core_cliff = nblocks % nblocks_per_core;
    const uint32_t ncores_x = grid_size.x;
    const uint32_t ncores_y = std::ceil(static_cast<float>(ncores) / ncores_x);
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

    FullRep(uint32_t n_rows, uint32_t n_pads, uint32_t times, uint32_t pads_mul, uint32_t times_total) :
        rep{n_rows / 32, n_rows % 32, n_pads / 32, times},
        pad{0, 0, (n_rows + n_pads) * pads_mul, 1},
        times_total(times_total) {
        TT_FATAL((n_rows + n_pads) % 32 == 0 && "total rows must be divisible by 32", "Error");
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

inline std::vector<std::vector<BlockRep>> distribute_work(
    const tt::tt_metal::LegacyShape& unpadded, const Padding& padding, uint32_t num_cores, uint32_t blocks_per_core, bool has_cliff, uint32_t nblocks_per_core_cliff) {
    auto input_w = unpadded.rank() >= 4 ? unpadded[-4] : 1;
    auto input_z = unpadded.rank() >= 3 ? unpadded[-3] : 1;
    auto input_y = unpadded.rank() >= 2 ? unpadded[-2] : 1;

    auto padding_w = unpadded.rank() >= 4 ? padding[padding.get_normalized_index(-4)].back : 0;
    auto padding_z = unpadded.rank() >= 3 ? padding[padding.get_normalized_index(-3)].back : 0;
    auto padding_y = unpadded.rank() >= 2 ? padding[padding.get_normalized_index(-2)].back : 0;

    // total work is a full rep followed by a padding.
    auto full_rep_blocks = FullRep(input_y, padding_y, input_z, padding_z, input_w).to_block_reps();
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

} // namespace operations::core::work_split

using namespace operations::core::work_split;

}  // namespace ttnn
