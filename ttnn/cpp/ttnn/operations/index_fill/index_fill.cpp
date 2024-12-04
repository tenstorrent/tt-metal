// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/operations/index_fill/device/index_fill_device_operation.hpp"

namespace ttnn::operations::index_fill {

Tensor IndexFill::invoke(
    const Tensor& input,
    const uint32_t dim,
    const Tensor& index,
    const std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::index_fill(input, dim, index, value, memory_config);
}

}  // namespace ttnn::operations::index_fill
