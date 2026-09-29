// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::randn {

struct RandnDeviceOperation {
    struct operation_attributes_t {
        const ttnn::Shape shape;
        DataType dtype;
        Layout layout;
        const MemoryConfig memory_config;
        MeshDevice* device;
        const DeviceComputeKernelConfig compute_kernel_config;
        std::optional<uint32_t> seed;
    };

    struct tensor_args_t {};

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        // Redraws the per-core seeds (hash-excluded) and re-applies the output address in place on
        // every cache hit; supersedes the address binding. No rebuild.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    // Hashes every attribute except `seed`; the seed is re-applied per dispatch by
    // ProgramFactory::override_runtime_arguments.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::randn

namespace ttnn::prim {
ttnn::operations::randn::RandnDeviceOperation::tensor_return_value_t randn(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<uint32_t> seed);
}  // namespace ttnn::prim
