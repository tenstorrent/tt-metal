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
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    bool fused_gelu,
    bool interleaved_output,
    bool fused_gelu_approx) {
    return ttnn::prim::matmul_decode(
        input_tensor_a,
        input_tensor_b,
        partial_width_sharded,
        dtype,
        compute_kernel_config,
        fused_gelu,
        interleaved_output,
        fused_gelu_approx);
}

}  // namespace ttnn
