// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like.hpp"

#include "ttnn/operations/full_like/device/full_like_device_operation.hpp"

namespace ttnn::operations::full_like {

Tensor FullLike::invoke(
    const Tensor &input,
    const std::variant<float, int> fill_value,
    const std::optional<DataType> &dtype,
    const std::optional<Layout> &layout,
    const std::optional<MemoryConfig> &memory_config) {
    return ttnn::prim::moreh_full_like(input, fill_value, dtype, layout, memory_config);
}

}  // namespace ttnn::operations::full_like
