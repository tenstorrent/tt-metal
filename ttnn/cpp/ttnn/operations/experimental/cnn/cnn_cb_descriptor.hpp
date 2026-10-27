// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

// One single-format circular buffer, shared by convert_to_chw and convert_to_hwc.
inline tt::tt_metal::CBDescriptor make_cnn_circular_buffer(
    const tt::tt_metal::CoreRangeSet& core_grid,
    uint32_t index,
    uint32_t total_size,
    uint32_t page_size,
    const tt::DataFormat& format,
    tt::tt_metal::Buffer* buffer,
    bool log_cb = false) {
    if (log_cb) {
        log_debug(
            tt::LogType::LogOp,
            "Creating CB at index {} with total size {} B and page size {} B",
            index,
            total_size,
            page_size);
    }
    return tt::tt_metal::CBDescriptor{
        .total_size = total_size,
        .core_ranges = core_grid,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = page_size,
        }}},
        .buffer = buffer,
    };
}

}  // namespace ttnn::experimental::prim
