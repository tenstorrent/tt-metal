// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_attention_all_gather_async_device_operation_types.hpp"
#include "ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async {

using ttnn::ccl::EriscDatamoverBuilder;

struct RingAttentionAllGatherAsyncDeviceOperation {
    using operation_attributes_t = ring_attention_all_gather_async::operation_attributes_t;

    using tensor_args_t = ring_attention_all_gather_async::tensor_args_t;

    using spec_return_value_t = ring_attention_all_gather_async::spec_return_value_t;

    using tensor_return_value_t = ring_attention_all_gather_async::tensor_return_value_t;

    using program_factory_t = std::variant<RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& persistent_output_buffer,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::ccl::Topology topology,
        uint32_t num_links,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id);
};

}  // namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async

// TODO: Remove the following once ring_join_sdpa is migrated to new infra
namespace ttnn {
tt::tt_metal::operation::ProgramWithCallbacks
ring_attention_all_gather_async_multi_core_with_workers_program_with_callbacks(
    tt::tt_metal::Program& program,
    const std::vector<Tensor>& input_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    std::vector<Tensor>& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    CoreCoord core_grid_offset = CoreCoord(0, 0));
}

namespace ttnn::prim {
constexpr auto ring_attention_all_gather_async = ttnn::register_operation<
    "ttnn::prim::ring_attention_all_gather_async",
    ttnn::operations::experimental::ccl::ring_attention_all_gather_async::RingAttentionAllGatherAsyncDeviceOperation>();
}  // namespace ttnn::prim
