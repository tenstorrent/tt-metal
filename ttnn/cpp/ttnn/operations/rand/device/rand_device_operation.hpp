// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::operations::rand {

struct RandDeviceOperation {
    struct operation_attributes_t {
        ttnn::Shape shape;
        DataType dtype;
        Layout layout;
        MemoryConfig memory_config;
        MeshDevice* device;
        float from;
        float to;
        uint32_t seed;
        ttsl::SmallVector<bool> mesh_dim_is_sharded;

        // The cache key. from/to/seed/mesh_dim_is_sharded are deliberately excluded — they vary per
        // dispatch and are re-applied on every cache hit by RandProgramFactory::override_runtime_arguments,
        // so calls differing only in those values reuse the cached program.
        // `device` must be listed FIRST: rand has no input tensor, so the framework discovers the mesh
        // device by reflecting over attribute_values() (get_first_object_of_type), whose tuple path only
        // inspects the first element.
        static constexpr auto attribute_names =
            std::forward_as_tuple("device", "shape", "dtype", "layout", "memory_config");
        auto attribute_values() const { return std::forward_as_tuple(device, shape, dtype, layout, memory_config); }
    };

    struct tensor_args_t {};

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // Descriptor program factory (Metal 2.0 named bindings). Owns both the cache-miss builder
    // (create_descriptor) and the cache-hit re-application (override_runtime_arguments); the
    // framework resolves both on the factory through program_factory_t.
    struct RandProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);

        // Re-applies every per-dispatch arg on each cache hit (per-core seed/from/to and the output
        // address), from the same builder create_descriptor uses. Replaces get_dynamic_runtime_args
        // and resolve_bindings for this op.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };
    using program_factory_t = std::variant<RandProgramFactory>;

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::rand

namespace ttnn::prim {
ttnn::operations::rand::RandDeviceOperation::tensor_return_value_t uniform(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    float from,
    float to,
    uint32_t seed,
    ttsl::SmallVector<bool> mesh_dim_is_sharded = {});
}  // namespace ttnn::prim
