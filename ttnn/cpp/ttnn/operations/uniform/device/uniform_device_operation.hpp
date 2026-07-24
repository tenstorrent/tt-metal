// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::operations::uniform {

struct UniformDeviceOperation {
    struct operation_attributes_t {
        const float from;
        const float to;
        uint32_t seed;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;

        // from/to/seed are re-applied via get_dynamic_runtime_args, so they're excluded from the
        // hash. Shape/dtype/device come from the input tensor (tensor_args).
        static constexpr auto attribute_names = std::forward_as_tuple("memory_config", "compute_kernel_config");
        auto attribute_values() const { return std::forward_as_tuple(memory_config, compute_kernel_config); }
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // seed/from/to are excluded from the program hash (so calls differing only in those values
    // cache-hit instead of recompiling); they are DYNAMIC and re-applied to the cached program on
    // every dispatch. Must mirror the compute-kernel runtime args built in create_descriptor().
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::uniform

namespace ttnn::prim {
ttnn::Tensor uniform(
    const Tensor& input,
    float from,
    float to,
    uint32_t seed,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
}  // namespace ttnn::prim
