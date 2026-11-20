// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::moreh::moreh_sgd {
struct MorehSgdOperation {
    struct operation_attributes_t {
        float lr;
        float momentum;
        float dampening;
        float weight_decay;
        bool nesterov;
        bool momentum_initialized;
        const MemoryConfig param_out_memory_config;
        const MemoryConfig momentum_buffer_out_memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "momentum_initialized",
            "param_out_memory_config",
            "momentum_buffer_out_memory_config",
            "compute_kernel_config");
        auto attribute_values() const {
            return std::forward_as_tuple(
                momentum,
                dampening,
                weight_decay,
                nesterov,
                momentum_initialized,
                param_out_memory_config,
                momentum_buffer_out_memory_config,
                compute_kernel_config);
        }
    };

    struct tensor_args_t {
        const Tensor& param_in;
        const Tensor& grad;
        const std::optional<Tensor>& momentum_buffer_in;
        const std::optional<Tensor>& param_out;
        const std::optional<Tensor>& momentum_buffer_out;
    };

    using spec_return_value_t = std::vector<std::optional<tt::tt_metal::TensorSpec>>;
    using tensor_return_value_t = std::vector<std::optional<Tensor>>;

    struct MorehSgdProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);

        // Writes the current lr and the buffer addresses into the cached program on every cache hit.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<MorehSgdProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::moreh::moreh_sgd

namespace ttnn::prim {
ttnn::operations::moreh::moreh_sgd::MorehSgdOperation::tensor_return_value_t moreh_sgd(
    const Tensor& param_in,
    const Tensor& grad,
    const std::optional<Tensor>& momentum_buffer_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const std::optional<MemoryConfig>& param_out_memory_config,
    const std::optional<MemoryConfig>& momentum_buffer_out_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
}
