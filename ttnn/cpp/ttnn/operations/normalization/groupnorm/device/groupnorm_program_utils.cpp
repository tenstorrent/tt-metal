// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_program_utils.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <algorithm>

namespace ttnn::prim {

uint32_t GroupNormPadCorrection::scaler_bits(uint32_t reduce_factor_w) const {
    const float sc = 1.0f / std::sqrt(
                                static_cast<float>(reduce_factor_w) * static_cast<float>(logical_hw) /
                                static_cast<float>(padded_hw));
    return std::bit_cast<uint32_t>(sc);
}

GroupNormPadCorrection make_group_norm_pad_correction(uint32_t logical_hw, uint32_t padded_hw, bool use_welford) {
    // Welford cannot express this: its kernels transpose H*W into the tile columns and track the
    // sample count in tile units, so the padding rows cannot be excluded. ttnn::group_norm routes
    // non-tile-aligned Welford requests to the two-pass path instead.
    GroupNormPadCorrection pad;
    pad.active = !use_welford && (logical_hw != padded_hw);
    pad.logical_hw = logical_hw;
    pad.padded_hw = padded_hw;
    pad.kernel_logical_hw = pad.active ? logical_hw : padded_hw;
    pad.k_bits = std::bit_cast<uint32_t>(static_cast<float>(padded_hw) / static_cast<float>(logical_hw) - 1.0f);
    return pad;
}

void append_group_norm_pad_correction_cbs(
    tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs,
    const GroupNormPadCorrection& pad,
    std::array<uint32_t, 3> cb_indices,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    tt::DataFormat data_format,
    uint32_t single_tile_size) {
    if (!pad.active) {
        return;
    }
    for (uint32_t cb_index : cb_indices) {
        cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = single_tile_size,
            .core_ranges = core_ranges,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index),
                .data_format = data_format,
                .page_size = single_tile_size,
            }}},
        });
    }
}

bool groupnorm_needs_fp32_reconfig(std::initializer_list<tt::DataFormat> reconfig_formats) {
    return std::any_of(reconfig_formats.begin(), reconfig_formats.end(), [](tt::DataFormat format) {
        return format != tt::DataFormat::Float16_b;
    });
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

bool is_rectangle_grid(const std::vector<tt::tt_metal::CoreCoord>& core_coords) {
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
    std::vector<tt::tt_metal::CoreCoord>& group,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_first,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_mid,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_last) {
    size_t remove_front = 0;
    size_t remove_back = 0;
    size_t min_total_removal = group.size();

    for (size_t front = 0; front <= group.size(); ++front) {
        for (size_t back = 0; front + back <= group.size(); ++back) {
            if (is_rectangle_grid(std::vector<tt::tt_metal::CoreCoord>(group.begin() + front, group.end() - back))) {
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
