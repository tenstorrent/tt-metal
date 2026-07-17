// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::bernoulli {

struct BernoulliDeviceOperation {
    struct operation_attributes_t {
        uint32_t seed;
        const DataType dtype;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;

        // seed is re-applied via override_runtime_arguments, so it's excluded from the hash.
        // Shape/device come from the input tensor (tensor_args).
        static constexpr auto attribute_names = std::forward_as_tuple("dtype", "memory_config", "compute_kernel_config");
        auto attribute_values() const { return std::forward_as_tuple(dtype, memory_config, compute_kernel_config); }
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::optional<Tensor>& output;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Re-derives ALL per-dispatch state (incl. the hash-excluded seed) via create_descriptor and
    // re-applies it to the cached program on every hit. Supersedes get_dynamic_runtime_args.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::bernoulli

namespace ttnn::prim {
ttnn::operations::bernoulli::BernoulliDeviceOperation::tensor_return_value_t bernoulli(
    const Tensor& input,
    uint32_t seed,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
}  // namespace ttnn::prim
