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
        const ttnn::Shape shape;
        DataType dtype;
        Layout layout;
        const MemoryConfig memory_config;
        MeshDevice* device;
        const float from;
        const float to;
        uint32_t seed;
        ttsl::SmallVector<bool> mesh_dim_is_sharded;

        // Cache key. seed/from/to are omitted (re-applied per dispatch via override_runtime_arguments,
        // so calls differing only in those values reuse the cached program). `device` must be FIRST:
        // rand has no input tensor, so the framework discovers the mesh device via
        // get_first_object_of_type over attribute_values(), whose tuple path inspects only element 0.
        static constexpr auto attribute_names =
            std::forward_as_tuple("device", "shape", "dtype", "layout", "memory_config");
        auto attribute_values() const { return std::forward_as_tuple(device, shape, dtype, layout, memory_config); }
    };

    struct tensor_args_t {};

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct RandProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);

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
