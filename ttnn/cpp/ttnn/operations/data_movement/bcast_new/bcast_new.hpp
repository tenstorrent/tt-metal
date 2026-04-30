// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
Tensor bcast_new(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttnn::BcastOpMath bcast_op,
    ttnn::BcastOpDim bcast_dim,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
}  // namespace ttnn
