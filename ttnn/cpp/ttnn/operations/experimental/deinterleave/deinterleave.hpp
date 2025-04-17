// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deinterleave {
struct DeinterleaveToBatch {
    static Tensor invoke(
        const Tensor& input,
        const uint32_t input_height,
        const uint32_t input_width,
        const std::array<uint32_t, 2> stride_hw,
        const uint32_t barrier_threshold,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

struct DeinterleaveLocal {
    static OptionalTensors invoke(
        const Tensor& input,
        const uint32_t input_height,
        const uint32_t input_width,
        const std::array<uint32_t, 2> stride_hw,
        const uint32_t barrier_threshold,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);

    static OptionalTensors create_async_optional_output_tensors(
        // static std::vector<Tensor> create_async_output_tensors(
        const Tensor& input,
        const uint32_t input_height,
        const uint32_t input_width,
        const std::array<uint32_t, 2> stride_hw,
        const uint32_t barrier_threshold,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::experimental::deinterleave

namespace ttnn {
namespace experimental {
constexpr auto deinterleave_to_batch = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::deinterleave_to_batch",
    ttnn::operations::experimental::deinterleave::DeinterleaveToBatch>();

constexpr auto deinterleave_local = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::deinterleave_local",
    ttnn::operations::experimental::deinterleave::DeinterleaveLocal>();

}  // namespace experimental
}  // namespace ttnn
