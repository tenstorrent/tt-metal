// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

// One translation of a single-format circular buffer, shared by the padded_slice and slice_write factories.
inline tt::tt_metal::CBDescriptor make_slice_cb_descriptor(
    uint32_t cb_index,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t page_size,
    uint32_t num_pages,
    tt::DataFormat data_format,
    tt::tt_metal::Buffer* buffer = nullptr) {
    return tt::tt_metal::CBDescriptor{
        .total_size = num_pages * page_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = data_format,
            .page_size = page_size,
        }}},
        .buffer = buffer,
    };
}

}  // namespace ttnn::experimental::prim
