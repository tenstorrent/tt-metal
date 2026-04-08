// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor embedding_bw(
    const Tensor& input_tensor_arg,
    const Tensor& weight_tensor_arg,
    const Tensor& output_gradient_tensor_arg,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
