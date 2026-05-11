// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill.hpp"

#include "ttnn/operations/index_fill/device/index_fill_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor index_fill(
    const Tensor& input,
    const uint32_t dim,
    const Tensor& index,
    const std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config) {
    TT_OP_SCOPE("ttnn::index_fill");
    return ttnn::prim::index_fill(input, dim, index, value, memory_config);
}

}  // namespace ttnn
