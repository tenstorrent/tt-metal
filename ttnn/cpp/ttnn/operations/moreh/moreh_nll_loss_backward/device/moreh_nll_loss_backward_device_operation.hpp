// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

struct MorehNllLossBackwardDeviceOperation {
    // Define the operation attributes. This is it to store all variables needed by operations that aren't tensors
    struct operation_attributes_t {
        const bool reduction_mean;
        const int32_t ignore_index;
        const MemoryConfig memory_config;
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config;
    };

    // Define the tensor arguments. This is it to store all tensors passed in and/or out of the operation
    // Tensor arguments don't need to be just input tensors, they can be output tensors, input/output tensors, optional
    // tensors, etc.
    struct tensor_args_t {
        const Tensor& target_tensor;
        const Tensor& output_grad_tensor;
        const std::optional<const Tensor> weight_tensor;
        const std::optional<const Tensor> divisor_tensor;
        const std::optional<const Tensor> input_grad_tensor;
    };

    // Define the return types for the shape(s) of the operation
    // Can be a single ttnn::Shape, std::optional<ttnn::Shape>, std::vector<ttnn::Shape>, std::tuple<ttnn::Shape> etc.
    using shape_return_value_t = ttnn::Shape;

    // Define the return types for the tensor(s) of the operation
    // Can be a single Tensor, std::optional<Tensor, ...>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = ttnn::Tensor;

    struct Factory {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<Factory>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& target_tensor,
        const Tensor& output_grad_tensor,
        const bool reduction_mean,
        const std::optional<const Tensor> weight_tensor,
        const std::optional<const Tensor> input_grad_tensor,
        const std::optional<const Tensor> divisor_tensor,
        const std::optional<int32_t> ignore_index,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward

namespace ttnn::prim {
constexpr auto moreh_nll_loss_backward = ttnn::register_operation<
    "ttnn::prim::moreh_nll_loss_backward",
    ttnn::operations::moreh::moreh_nll_loss_backward::MorehNllLossBackwardDeviceOperation>();
}  // namespace ttnn::prim
