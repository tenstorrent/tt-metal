// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>
#include <cstdint>
#include <optional>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/core.hpp>
#include <ttnn/types.hpp>
#include <ttnn/decorators.hpp>
#include <ttnn/device_operation.hpp>
#include <ttnn/distributed/types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::debug {

struct ApplyDeviceDelayDeviceOperation {
    struct operation_attributes_t {
        const std::vector<std::vector<uint32_t>> delays;
        const CoreRangeSet worker_core_range_set;
        const ttnn::MeshDevice* mesh_device;
    };

    // We need a dummy tensor args that can provide mesh device info
    struct tensor_args_t {
        ttnn::Tensor input_tensor;
    };

    // Return a minimal dummy tensor since the infrastructure doesn't support void
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;

    struct ApplyDeviceDelayMeshWorkload {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle kernel_id;
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

    using program_factory_t = std::variant<ApplyDeviceDelayMeshWorkload>;

    // Mandatory methods
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        ttnn::MeshDevice& mesh_device,
        const std::vector<std::vector<uint32_t>>& delays,
        const CoreRangeSet& subdevice_core_range_set);
};

}  // namespace ttnn::operations::debug

namespace ttnn::prim {
// Register the operation
constexpr auto apply_device_delay = ttnn::
    register_operation<"ttnn::prim::apply_device_delay", ttnn::operations::debug::ApplyDeviceDelayDeviceOperation>();
}  // namespace ttnn::prim
