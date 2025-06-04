// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad {
struct MorehLayerNormBackwardInputGradOperation {
    struct operation_attributes_t {
        uint32_t normalized_dims;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& output_grad;
        const Tensor& input;
        const Tensor& mean;
        const Tensor& rstd;
        const std::optional<const Tensor>& input_grad;
        const std::optional<const Tensor>& gamma;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& input_grad);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& input_grad);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& mean,
        const Tensor& rstd,
        uint32_t normalized_dims,
        const std::optional<const Tensor>& input_grad,
        const std::optional<const Tensor>& gamma,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad

namespace ttnn::prim {
constexpr auto moreh_layer_norm_backward_input_grad = ttnn::register_operation<
    "ttnn::prim::moreh_layer_norm_backward_input_grad",
    operations::moreh::moreh_layer_norm_backward_input_grad::MorehLayerNormBackwardInputGradOperation>();
}  // namespace ttnn::prim
