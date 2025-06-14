// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
#include <cstdint>

inline size_t get_flat_index_from_shape(
    const ttnn::ccl::Shape4D<uint32_t>& shape, const ttnn::ccl::Shape4D<uint32_t>& index) {
    std::size_t offset = index.x;
    std::size_t inner_volume = shape.x;
    offset += index.y * inner_volume;
    inner_volume *= shape.y;
    offset += index.z * inner_volume;
    inner_volume *= shape.z;
    offset += index.w * inner_volume;
    return offset;
}
