// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like.hpp"

#include "ttnn/operations/full/device/full_device_operation.hpp"

namespace ttnn::operations::full_like {

Tensor FullLike::invoke(
    const Tensor& input,
    const std::variant<float, int> fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Full Like: Input must be on device");
    const auto& shape = input.logical_shape();
    return ttnn::prim::full(
        ttnn::SmallVector<uint32_t>(shape.cbegin(), shape.cend()),
        fill_value,
        input.device(),
        dtype.value_or(input.dtype()),
        layout.value_or(input.layout()),
        memory_config.value_or(input.memory_config()));
}

}  // namespace ttnn::operations::full_like
