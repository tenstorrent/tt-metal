// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>
#include <vector>
#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl {

struct AllGatherConcatDeviceOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const uint32_t num_links;
        const uint32_t ring_size;
        const MemoryConfig output_mem_config;
        const ttnn::ccl::Topology topology;
        const std::optional<GlobalSemaphore> semaphore;
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
        const uint32_t num_heads;
        bool use_noc1_only;
        const uint32_t cluster_axis;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        const Tensor& buffer_tensor;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct AllGatherConcatProgram {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle worker_sender_reader_kernel_id;
            tt::tt_metal::KernelHandle worker_sender_writer_kernel_id;
            tt::tt_metal::KernelHandle concat_reader_kernel_id;
            tt::tt_metal::CBHandle cb_q_output;
            std::vector<CoreCoord> sender_worker_cores;
            std::vector<CoreCoord> cores;
            uint32_t num_concat_worker_cores;
        };

        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coord,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<AllGatherConcatProgram>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& buffer_tensor,
        uint32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& semaphore,
        uint32_t num_heads,
        bool use_noc1_only,
        const MemoryConfig& memory_config,
        std::optional<uint32_t> num_links,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id);
};

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {
constexpr auto all_gather_concat = ttnn::register_operation<
    "ttnn::prim::all_gather_concat",
    ttnn::operations::experimental::ccl::AllGatherConcatDeviceOperation>();
}  // namespace ttnn::prim
