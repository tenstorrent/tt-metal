// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
struct MorehGroupNormBackwardGammaBetaGradOperation {
    struct operation_attributes_t {
        const uint32_t num_groups;
        const std::vector<bool> are_required_outputs;
        const MemoryConfig gamma_grad_memory_config;
        const MemoryConfig beta_grad_memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };
    struct tensor_args_t {
        const Tensor& output_grad;
        const Tensor& input;
        const Tensor& mean;
        const Tensor& rstd;
        const std::optional<const Tensor> gamma_grad;
        const std::optional<const Tensor> beta_grad;
    };

    using spec_return_value_t = std::vector<std::optional<TensorSpec>>;
    using tensor_return_value_t = std::vector<std::optional<Tensor>>;

    struct MorehGroupNormBackwardGammaBetaGradFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernels_id;
            tt::tt_metal::KernelHandle writer_kernels_id;
            uint32_t num_cores_to_be_used;
            std::size_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& outputs);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& outputs);
    };

    using program_factory_t = std::variant<MorehGroupNormBackwardGammaBetaGradFactory>;

    static void validate_tensors(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_group_norm_backward::MorehGroupNormBackwardGammaBetaGradOperation::tensor_return_value_t
moreh_group_norm_backward_gamma_beta_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t num_groups,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor>& gamma_grad = std::nullopt,
    const std::optional<const Tensor>& beta_grad = std::nullopt,
    const std::optional<MemoryConfig>& gamma_grad_memory_config = std::nullopt,
    const std::optional<MemoryConfig>& beta_grad_memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
}  // namespace ttnn::prim
