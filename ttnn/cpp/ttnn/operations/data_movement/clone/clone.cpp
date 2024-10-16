// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone.hpp"

#include "device/clone_device_operation.hpp"

namespace ttnn::operations::data_movement::clone {
Tensor Clone::invoke(const Tensor& input,
                     const std::optional<DataType>& dtype,
                     const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::clone(input, dtype, memory_config);
}
}  // namespace ttnn::operations::data_movement::clone
