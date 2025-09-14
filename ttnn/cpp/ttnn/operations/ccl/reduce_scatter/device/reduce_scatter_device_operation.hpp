// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"

namespace ttnn::operations::ccl {

struct ReduceScatterDeviceOperation {
    struct operation_attributes_t {
        const MemoryConfig memory_config;
        uint32_t dim;
        const std::optional<uint32_t> cluster_axis;
        const std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
        const tt::tt_fabric::Topology topology;
        const uint32_t num_links;
    };

    struct tensor_args_t {
        const Tensor input_tensor;
        std::optional<Tensor> optional_output_tensor;
    };

    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    struct ReduceScatterProgram {
        struct shared_variables_t {
            std::vector<tt::tt_metal::GlobalSemaphore> multidevice_semaphores;
            tt::tt_metal::GlobalSemaphore barrier_semaphore;
            ttnn::ReduceScatterProgramArtifacts program_artifacts;
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
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const std::vector<tt::tt_metal::GlobalSemaphore>& multidevice_semaphores,
            const tt::tt_metal::GlobalSemaphore& barrier_semaphore);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ReduceScatterProgram>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        uint32_t dim,
        std::optional<uint32_t> cluster_axis,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
        const ttnn::MemoryConfig& memory_config,
        const std::optional<ttnn::Tensor>& optional_output_tensor,
        uint32_t num_links,
        tt::tt_fabric::Topology topology);
};
}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
constexpr auto reduce_scatter =
    ttnn::register_operation<"ttnn::prim::reduce_scatter", ttnn::operations::ccl::ReduceScatterDeviceOperation>();
}  // namespace ttnn::prim
