// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_op.hpp"

namespace ttnn::operations::ccl {

struct MultiDeviceScatterDeviceOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const uint32_t cluster_axis;
        const MemoryConfig output_mem_config;
    };
    struct tensor_args_t {
        const Tensor input_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = Tensor;

    struct MultiDeviceScatter {
        using OverrideRuntimeArgsCallback = std::function<void(
            const void*,
            tt::tt_metal::Program&,  // ‼  no const, exact type
            const std::vector<tt::tt_metal::Tensor>&,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>&,
            const std::vector<tt::tt_metal::Tensor>&)>;

        // -- shared variables --------------------------------------------
        struct shared_variables_t {
            OverrideRuntimeArgsCallback override_runtime_arguments_callback;
            ttnn::operations::data_movement::SliceDeviceOperation slice_op;
        };
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiDeviceScatter>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const uint32_t cluster_axis,
        const ttnn::MemoryConfig& memory_config);
};
}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::example
constexpr auto multidevice_scatter = ttnn::
    register_operation<"ttnn::prim::multidevice_scatter", ttnn::operations::ccl::MultiDeviceScatterDeviceOperation>();
}  // namespace ttnn::prim
