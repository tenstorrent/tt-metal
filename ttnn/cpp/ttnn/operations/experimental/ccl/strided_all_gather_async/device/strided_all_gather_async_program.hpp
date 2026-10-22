// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "strided_all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::prim {

struct StridedAllGatherAsyncProgramFactory {
    // Per-coord program build: ring index and fabric neighbours depend on the mesh coordinate.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const StridedAllGatherAsyncParams& operation_attributes,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Re-applies the input/output addresses and the caller-supplied semaphore addresses, which
    // StridedAllGatherAsync::compute_program_hash leaves out of the key.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const StridedAllGatherAsyncParams& operation_attributes,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Appends the strided all-gather kernels, CBs and semaphores to `desc`, which may already hold a parent
// op's entries (the fused matmul). The first appended kernel lands at desc.kernels.size() on entry.
void strided_all_gather_async_minimal_default_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler>& fused_op_signaler,
    bool read_local_slice_from_input,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> mm_cores_y,
    std::optional<uint32_t> mm_block_ht,
    std::optional<uint32_t> mm_block_wt,
    CoreCoord core_grid_offset = CoreCoord(0, 0),
    MMSignalAggregatorMode mm_signal_aggregator_mode = MMSignalAggregatorMode::Auto);

// Cache-hit patch for a program built by strided_all_gather_async_minimal_default_helper starting at
// `first_kernel_index`: rewrites the input/output addresses and every caller-supplied semaphore address
// (out-ready semaphores, and on a fused program the per-worker aggregator semaphores).
void strided_all_gather_async_patch_runtime_args(
    tt::tt_metal::Program& program,
    uint32_t first_kernel_index,
    const StridedAllGatherAsyncParams& attributes,
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    bool fused);

}  // namespace ttnn::experimental::prim
