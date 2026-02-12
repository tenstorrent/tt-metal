// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::bernoulli_new {

// ---------------------------------------------------------------
// BernoulliNewDeviceOperation -- ProgramDescriptor variant.
//
// Functionally identical to BernoulliDeviceOperation but the
// ProgramFactory implements only `create_descriptor` instead of
// the traditional create / override_runtime_arguments pair.
// The framework adapts the descriptor into the full lifecycle.
// ---------------------------------------------------------------
struct BernoulliNewDeviceOperation {
    struct operation_attributes_t {
        uint32_t seed;
        const DataType dtype;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::optional<Tensor>& output;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // ---------------------------------------------------------------
    // The factory exposes a SINGLE method: create_descriptor.
    // No shared_variables_t.  No create.  No override_runtime_arguments.
    // The framework handles everything through DescriptorMeshWorkloadFactoryAdapter.
    // ---------------------------------------------------------------
    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Same hash as original Bernoulli: exclude seed so cache hits work across different seeds.
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bernoulli_new

namespace ttnn::prim {
ttnn::operations::bernoulli_new::BernoulliNewDeviceOperation::tensor_return_value_t bernoulli_new(
    const Tensor& input,
    uint32_t seed,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
}  // namespace ttnn::prim
