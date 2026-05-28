// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory {
    using operation_attributes_t = RingAttentionAllGatherAsyncParams;

    using tensor_args_t = RingAttentionAllGatherAsyncInputs;

    using tensor_return_value_t = std::vector<Tensor>;

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};
}  // namespace ttnn::experimental::prim

namespace ttnn {

// Append all kernels, CBs, semaphores, and runtime args required by the
// ring-attention all-gather worker pipeline to `desc`.
//
// `desc` may already contain entries from a parent op (e.g., ring_joint_sdpa).
// Semaphore IDs assigned by this helper start at `desc.semaphores.size()` at
// entry and are sequential. The descriptor framework auto-patches buffer
// addresses on cache hits so callers do not need to retain kernel handles or
// implement an override_runtime_arguments path.
void ring_attention_all_gather_async_multi_core_with_workers_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const std::vector<Tensor>& input_tensor,
    const MeshCoordinate& target_device_coord,
    std::optional<MeshCoordinate> forward_device_coord,
    std::optional<MeshCoordinate> backward_device_coord,
    std::vector<Tensor>& output_tensor,
    int32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    CoreCoord core_grid_offset = CoreCoord(0, 0),
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR,
    // When set, gather only this single batch slot of `input_tensor` (dim-0 index) and write it to
    // batch slot 0 of `output_tensor` (which must then be batch-1). Lets a fused consumer keep a full
    // KV cache as input while the gathered scratch buffer is only one slot wide. std::nullopt => gather
    // the full batch (default, unchanged behavior).
    std::optional<uint32_t> input_batch_slice_idx = std::nullopt);

}  // namespace ttnn
