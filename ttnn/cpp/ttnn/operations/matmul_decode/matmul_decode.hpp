// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

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
    bool partial_width_sharded = false,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool fused_gelu = false,
    bool interleaved_output = false,
    bool fused_gelu_approx = false,
    bool reshard_input = false,
    uint32_t reshard_cores = 2);

// gate_up_matmul_decode: fused GeGLU gate+up projection. ONE gather of the activation A, TWO
// partial-width-sharded resident-L1 weights (gate_b, up_b on the SAME core grid), TWO outputs:
//   gate = gelu(A @ gate_w)  (tanh-approx when fused_gelu_approx),  up = A @ up_w.
// Replaces two separate matmul_decode(partial_width_sharded, reshard_input) calls (gate w/
// fused_gelu, up w/o) -- sharing the x-gather + halving the reduce/dispatch. Returns {gate, up},
// both width-sharded with the same shape/layout the two single-output calls produced.
std::array<Tensor, 2> gate_up_matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& gate_b,
    const Tensor& up_b,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool fused_gelu_approx = false,
    bool reshard_input = false,
    uint32_t reshard_cores = 2);

}  // namespace ttnn
