// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full.hpp"

#include "device/full_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::full {
Tensor Full::invoke(
    const ttnn::Shape& shape,
    const float fill_value,
    const ttnn::Tensor any,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::full(shape, fill_value, any, dtype, layout, memory_config);
}

Tensor Full::invoke(
    const ttnn::Shape& shape,
    const int fill_value,
    const ttnn::Tensor any,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::full(shape, fill_value, any, dtype, layout, memory_config);
}
}  // namespace ttnn::operations::full
