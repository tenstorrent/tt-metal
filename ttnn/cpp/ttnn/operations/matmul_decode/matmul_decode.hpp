// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    // deep-plan_14 Lever 0: settable fat-fill (out_subblock_h/w, in0_block_w) + temporal
    // (k_stream / k_slice_tiles) knobs. Defaults preserve deep-plan_13 behavior exactly.
    std::optional<uint32_t> out_subblock_h = std::nullopt,
    std::optional<uint32_t> out_subblock_w = std::nullopt,
    uint32_t in0_block_w = 1,
    bool k_stream = false,
    uint32_t k_slice_tiles = 0);

}  // namespace ttnn
