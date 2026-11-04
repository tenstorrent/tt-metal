// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_new.hpp"

#include "ttnn/operations/index_fill_new/device/index_fill_new_device_operation.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"

namespace ttnn {

Tensor index_fill_new(
    const Tensor& input,
    const uint32_t dim,
    const Tensor& index,
    const std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config) {
    // The kernels read the index as one row of values, so a TILE index (padded to a full tile) is untilized first.
    const Tensor index_rm = index.layout() == tt::tt_metal::Layout::ROW_MAJOR
                                 ? index
                                 : ttnn::to_layout(index, tt::tt_metal::Layout::ROW_MAJOR);
    return ttnn::prim::index_fill_new(input, dim, index_rm, value, memory_config);
}

}  // namespace ttnn
