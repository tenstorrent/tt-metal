// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::conv1d_depthwise {

// Depthwise 1D FIR filter with taps shared across all channels:
//   y[b, t, c] = sum_{j<K} taps[j] * x[b, t*stride + j, c]
// Input/output are (B, T_pad, C) ROW_MAJOR; output T_out = (T_pad - K) / stride + 1.
struct Conv1dDepthwiseOperation {
    struct operation_attributes_t {
        std::vector<float> taps;
        uint32_t stride;
        DataType dtype;
        DeviceComputeKernelConfig compute_kernel_config;
        MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::conv1d_depthwise

namespace ttnn::prim {
ttnn::operations::experimental::conv1d_depthwise::Conv1dDepthwiseOperation::tensor_return_value_t conv1d_depthwise(
    const Tensor& input,
    const std::vector<float>& taps,
    uint32_t stride,
    const DataType& dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const MemoryConfig& memory_config);
}  // namespace ttnn::prim
