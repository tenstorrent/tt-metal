// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_async_device_operation_types.hpp"
#include "all_gather_async_default_program_factory.hpp"
#include "all_gather_async_llama_sharded_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::experimental::prim {

struct AllGatherAsyncDeviceOperation {
    using operation_attributes_t = AllGatherAsyncParams;
    using tensor_args_t = AllGatherAsyncInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DefaultMeshWorkloadFactory, LlamaShardedMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const std::optional<ttnn::Tensor>& persistent_output_buffer,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links,
        const std::optional<MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        const std::optional<uint32_t>& cluster_axis,
        bool use_optimal_ccl_for_llama,
        bool use_all_gather_async_llama_sharded,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        const std::optional<uint32_t>& chunks_per_sync,
        const std::optional<uint32_t>& num_workers_per_link,
        const std::optional<uint32_t>& num_buffers_per_channel,
        bool reverse_order,
        const std::optional<CoreRangeSet>& sub_core_grid,
        const MeshDevice* optional_mesh_device);
};

enum class AllGatherAsyncVersion {
    LLAMA_MINIMAL_SHARDED = 0,
    MINIMAL_DEFAULT = 1,
};

AllGatherAsyncVersion select_version(const AllGatherAsyncParams& operation_attributes);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
constexpr auto all_gather_async =
    ttnn::register_operation<"ttnn::prim::all_gather_async", ttnn::experimental::prim::AllGatherAsyncDeviceOperation>();
}  // namespace ttnn::prim
