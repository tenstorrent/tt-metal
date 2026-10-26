// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/dynamic_length.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::dynamic_length {

using tt::tt_metal::DataType;

SplitLength split(const ValidLength& length, std::size_t capacity) {
    if (const auto* scalar = std::get_if<std::size_t>(&length)) {
        return {*scalar, std::nullopt};
    }
    return {capacity, std::get<Tensor>(length)};
}

void validate_axes(std::span<const DynamicAxis> axes, std::string_view operation_name) {
    for (const auto& axis : axes) {
        TT_FATAL(axis.alignment > 0, "{}: {} alignment must be positive", operation_name, axis.name);
        if (axis.tensor != nullptr) {
            const auto& t = *axis.tensor;
            TT_FATAL(
                t.dtype() == DataType::UINT32 || t.dtype() == DataType::INT32,
                "{}: {} tensor must be UINT32 or INT32 (the kernels read element 0 as a raw 32-bit word). Got {}",
                operation_name,
                axis.name,
                t.dtype());
            TT_FATAL(
                t.storage_type() == StorageType::DEVICE, "{}: {} tensor must be on device", operation_name, axis.name);
            TT_FATAL(t.is_allocated(), "{}: {} tensor must be allocated on device", operation_name, axis.name);
            TT_FATAL(
                t.logical_volume() == 1,
                "{}: {} tensor must hold exactly one value (the kernels read page 0, element 0). Got volume {}",
                operation_name,
                axis.name,
                t.logical_volume());
            continue;
        }
        TT_FATAL(
            axis.value <= axis.capacity,
            "{}: {} ({}) exceeds the capacity ({}) the program is sized for. Allocate the inputs at a larger "
            "capacity",
            operation_name,
            axis.name,
            axis.value,
            axis.capacity);
        TT_FATAL(
            axis.value % axis.alignment == 0,
            "{}: {} ({}) must be a multiple of {}",
            operation_name,
            axis.name,
            axis.value,
            axis.alignment);
    }
}

}  // namespace ttnn::dynamic_length
