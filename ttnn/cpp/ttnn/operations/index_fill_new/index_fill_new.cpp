// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_new.hpp"

#include "ttnn/operations/index_fill_new/device/index_fill_new_device_operation.hpp"

namespace ttnn {

Tensor index_fill_new(
    const Tensor& input,
    const uint32_t dim,
    const Tensor& index,
    const std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::index_fill_new(input, dim, index, value, memory_config);
}

}  // namespace ttnn
