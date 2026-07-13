// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "device/my_matmul_device_operation.hpp"

namespace ttnn {

// Public matmul: C = A @ B, with A [.., M, K], B [.., K, N] -> C [.., M, N].
Tensor my_matmul(const Tensor& input_tensor_a, const Tensor& input_tensor_b);

}  // namespace ttnn
