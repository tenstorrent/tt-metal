// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full.hpp"

#include "device/full_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::full {
Tensor Full::invoke(
    const std::vector<uint32_t>& shape,
    const std::variant<float, int> fill_value,
    const ttnn::Tensor& any,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    std::vector<uint32_t> processed_shape = shape;
    if (processed_shape.size() == 1) {
        processed_shape.insert(processed_shape.begin(), 1);
    }
    SimpleShape simpleShape(std::move(const_cast<std::vector<uint32_t>&>(processed_shape)));
    return ttnn::prim::full(simpleShape, fill_value, any, dtype, layout, memory_config);
}
}  // namespace ttnn::operations::full
