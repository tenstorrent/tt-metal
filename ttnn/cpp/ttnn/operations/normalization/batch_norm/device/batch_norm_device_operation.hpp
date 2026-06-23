// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::normalization {
struct BatchNormOperation {
    struct operation_attributes_t {
        const float eps;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;

        DataType input_dtype;
        std::optional<DataType> dtype;
        ttsl::hash::hash_t to_hash() const;
        DataType get_dtype() const;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& batch_mean;
        const Tensor& batch_var;
        std::optional<Tensor> weight;
        std::optional<Tensor> bias;
        std::optional<Tensor> output;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct BatchNormFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<BatchNormFactory>;

    static void validate_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::normalization

namespace ttnn::prim {
ttnn::operations::normalization::BatchNormOperation::tensor_return_value_t batch_norm(
    const Tensor& input,
    const Tensor& batch_mean,
    const Tensor& batch_var,
    float eps,
    std::optional<Tensor> weight,
    std::optional<Tensor> bias,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
}  // namespace ttnn::prim
