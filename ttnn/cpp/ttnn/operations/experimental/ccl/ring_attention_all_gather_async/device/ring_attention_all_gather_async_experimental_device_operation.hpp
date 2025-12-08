// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/distributed/types.hpp"
#include <optional>
#include <tuple>
#include <vector>

namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async {

struct RingAttentionAllGatherAsyncExperimentalDeviceOperation {
    using operation_attributes_t = RingAttentionAllGatherAsyncDeviceOperation::operation_attributes_t;

    using tensor_args_t = RingAttentionAllGatherAsyncDeviceOperation::tensor_args_t;

    using spec_return_value_t = RingAttentionAllGatherAsyncDeviceOperation::spec_return_value_t;

    using tensor_return_value_t = RingAttentionAllGatherAsyncDeviceOperation::tensor_return_value_t;

    using program_factory_t = RingAttentionAllGatherAsyncDeviceOperation::program_factory_t;

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
        uint32_t num_links,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const std::optional<MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id);
};

}  // namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async

namespace ttnn::prim {
constexpr auto ring_attention_all_gather_async_experimental = ttnn::register_operation<
    "ttnn::prim::experimental::ring_attention_all_gather_async",
    ttnn::operations::experimental::ccl::RingAttentionAllGatherAsyncExperimentalDeviceOperation>();
}  // namespace ttnn::prim
