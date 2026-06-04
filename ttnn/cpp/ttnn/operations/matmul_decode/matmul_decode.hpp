// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// matmul_decode: template op skeleton that performs a matrix multiply between two tensors.
//
// This is currently implemented as a composite operation that delegates to ttnn::matmul.
// It exists as a starting point: replace the body in matmul_decode.cpp with a dedicated
// device operation (operation_attributes_t / tensor_args_t / program factories) when a
// custom decode-optimized implementation is required.
Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt);

}  // namespace ttnn
