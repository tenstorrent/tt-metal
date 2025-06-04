// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

    static constexpr uint32_t size = 4;
};

}  // namespace ttnn::operations::data_movement::reshape::detail
