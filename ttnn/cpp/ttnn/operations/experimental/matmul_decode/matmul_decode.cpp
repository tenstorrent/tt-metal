// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode.hpp"

#include "device/matmul_decode_device_operation.hpp"

namespace ttnn::experimental {

Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::prim::matmul_decode(input_tensor_a, input_tensor_b, partial_width_sharded, dtype, output_mem_config);
}

}  // namespace ttnn::experimental
