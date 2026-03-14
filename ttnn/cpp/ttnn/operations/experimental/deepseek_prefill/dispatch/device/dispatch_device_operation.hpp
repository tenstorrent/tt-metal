// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

namespace detail {

std::pair<std::array<uint32_t, 2>, std::array<uint32_t, 2>> get_cb_sizes(
    const Tensor& input_tensor,
    const Tensor& weights_tensor,
    const Tensor& indices_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis);

}  // namespace detail

struct DispatchDeviceOperation {
    struct operation_attributes_t {
        const uint32_t dispatch_group_size;
        const uint32_t experts_per_chip;
        const uint32_t num_routed_experts;
        const uint32_t num_experts_per_tok;
        const uint32_t metadata_len;
        const uint32_t max_dispatched_tokens_per_expert;
        const std::optional<uint32_t> axis;
        const uint32_t num_links;
        const tt::tt_fabric::Topology topology;
        const MemoryConfig output_mem_config;
        const CoreRangeSet worker_core_range_set;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "dispatch_group_size",
            "experts_per_chip",
            "num_routed_experts",
            "num_experts_per_tok",
            "metadata_len",
            "max_dispatched_tokens_per_expert",
            "axis",
            "num_links",
            "topology",
            "output_mem_config",
            "worker_core_range_set");

        auto attribute_values() const {
            return std::forward_as_tuple(
                dispatch_group_size,
                experts_per_chip,
                num_routed_experts,
                num_experts_per_tok,
                metadata_len,
                max_dispatched_tokens_per_expert,
                axis,
                num_links,
                topology,
                output_mem_config,
                worker_core_range_set);
        };
    };

    struct tensor_args_t {
        const Tensor input_tensor;
        const Tensor weights_tensor;
        const Tensor indices_tensor;
        const Tensor expert_offsets_tensor;
        const Tensor expert_dispatch_table_tensor;
    };

    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;
    using topology_return_value_t = std::array<tt::tt_metal::TensorTopology, 2>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    struct DispatchProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            std::vector<CoreCoord> cores;
            const GlobalSemaphore init_semaphore;
            const GlobalSemaphore cross_device_semaphore;
        };

        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
        static ttnn::device_operation::CachedProgram<
            DispatchDeviceOperation::DispatchProgramFactory::shared_variables_t>
        create_at(
            const operation_attributes_t& operation_attributes,
            const MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const MeshCoordinateRangeSet& tensor_coords,
            const GlobalSemaphore& init_semaphore,
            const GlobalSemaphore& cross_device_semaphore);
        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<DispatchProgramFactory>;

    // Mandatory methods
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch

namespace ttnn::prim {
ttnn::operations::experimental::deepseek_prefill::dispatch::DispatchDeviceOperation::tensor_return_value_t
prefill_dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weights_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatched_tokens_per_expert,
    std::optional<uint32_t> axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set);
}  // namespace ttnn::prim
