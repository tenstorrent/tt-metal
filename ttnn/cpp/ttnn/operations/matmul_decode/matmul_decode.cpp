// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode.hpp"

#include "device/matmul_decode_device_operation.hpp"

namespace ttnn {

Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype) {
    return ttnn::prim::matmul_decode(input_tensor_a, input_tensor_b, partial_width_sharded, dtype);
}

}  // namespace ttnn
