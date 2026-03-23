// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "bcast_types.hpp"

namespace ttnn {
Tensor bcast(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttnn::BcastOpMath bcast_op,
    ttnn::BcastOpDim bcast_dim,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
}  // namespace ttnn
