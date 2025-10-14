// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::data_movement::reshape::detail {

struct SegmentMapData {
    using value_type = uint32_t;

    value_type input_page_index;
    value_type input_page_offset;
    value_type output_page_offset;
    value_type num_elements;
    bool operator==(const SegmentMapData& other) const {
        return input_page_index == other.input_page_index && input_page_offset == other.input_page_offset &&
               output_page_offset == other.output_page_offset && num_elements == other.num_elements;
    }

    static constexpr uint32_t size = 4;
};
inline std::pair<uint16_t, uint16_t> unpack_rt_short(uint32_t packed) { return {packed >> 16, packed & 0xFFFF}; }

}  // namespace ttnn::operations::data_movement::reshape::detail
