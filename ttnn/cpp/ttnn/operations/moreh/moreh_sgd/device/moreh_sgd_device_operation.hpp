// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"

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

    // Metal 2.0 program factory (MetalV2FactoryConcept). Single-variant: work is split across the
    // core grid (reader/writer on all cores, compute per core-group), but every node runs the same
    // program shape. `create_program_artifacts` is only detected by the framework as a
    // `program_factory_t` variant alternative, so the legacy `HasDirectDescriptor` shape is replaced
    // by this nested factory struct + variant + `select_program_factory`.
    struct MorehSgdProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MorehSgdProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

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
