// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <cstdint>
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

namespace ring_attention_all_gather_async_detail {

// All-gather reader runtime-arg layout: [0]=dim, [1]=ring_size, [2]=out_ready_sem,
// followed by one tensor-descriptor block per gathered input.
constexpr uint32_t kReaderRuntimeArgHeaderCount = 3;
// All-gather writer runtime-arg layout: [0]=dim, [1]=sem_noc0_x, [2]=sem_noc0_y, [3]=ring_size,
// [4]=out_ready_sem, followed by one tensor-descriptor block per gathered input.
constexpr uint32_t kWriterRuntimeArgHeaderCount = 5;
constexpr uint32_t kTensorDescriptorFieldCount = 9;
constexpr uint32_t kInputBatchBaseFieldOffset = 7;
// Per-(batch,head) page count each worker is allowed to gather. Defaults to the full input
// (input_Ht * input_Wt); the fused ring_joint_sdpa path patches it down to the logical_n-valid
// slab prefix so the gather moves only kv_actual-sized data, not the whole oversized cache.
constexpr uint32_t kValidPagesFieldOffset = 8;

inline uint32_t input_batch_base_pages(
    uint32_t batch_idx, uint32_t num_heads, uint32_t tensor_height_tiles, uint32_t tensor_width_tiles) {
    return batch_idx * num_heads * tensor_height_tiles * tensor_width_tiles;
}

}  // namespace ring_attention_all_gather_async_detail

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
    // When set, gather only this batch slot (dim-0 index) of `input_tensor` into slot 0 of
    // `output_tensor` — lets a consumer keep a full KV cache as input with a batch-1 gathered buffer
    // (a full-batch output also works; only slot 0 is written). std::nullopt => full batch (default).
    std::optional<uint32_t> input_batch_slice_idx = std::nullopt,
    // When set, gather only the first `gather_valid_Ht` tile-rows per (batch,head) instead of the
    // full input height — lets a consumer keep an oversized (growing) KV cache as input while moving
    // only the valid (e.g. logical_n-sized) prefix. Capped to the input height per gathered tensor.
    // std::nullopt => gather the full input (default). The fused ring_joint_sdpa path also re-patches
    // this per dispatch on cache hits (see apply_ring_joint_scalar_runtime_args).
    std::optional<uint32_t> gather_valid_Ht = std::nullopt);

}  // namespace ttnn
