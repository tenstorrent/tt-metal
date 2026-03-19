// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_device_operation.hpp"

namespace ttnn {

Tensor moreh_getitem(
    const std::optional<const Tensor>& input,
    const std::vector<Tensor>& index_tensors,
    const ttnn::SmallVector<uint32_t>& index_dims,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
