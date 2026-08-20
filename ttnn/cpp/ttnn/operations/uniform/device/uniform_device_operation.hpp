// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::uniform {

struct UniformDeviceOperation {
    struct operation_attributes_t {
        const float lower_bound;
        const float upper_bound;
        uint32_t seed;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;

        // Bounds/seed are re-applied via override_runtime_arguments, so they're excluded from the
        // hash. Shape/dtype/device come from the input tensor (tensor_args).
        static constexpr auto attribute_names = std::forward_as_tuple("memory_config", "compute_kernel_config");
        auto attribute_values() const { return std::forward_as_tuple(memory_config, compute_kernel_config); }
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct UniformProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        // Writes every per-dispatch arg (seed/from/to, hash-excluded) and the output address in place
        // on each cache hit. Supersedes get_dynamic_runtime_args and resolve_bindings; no rebuild.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<UniformProgramFactory>;

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::uniform

namespace ttnn::prim {
// lower_bound and upper_bound are inclusive, dtype-representable output bounds
// selected by the caller from the public half-open interval.
ttnn::Tensor uniform(
    const Tensor& input,
    float lower_bound,
    float upper_bound,
    uint32_t seed,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
}  // namespace ttnn::prim
