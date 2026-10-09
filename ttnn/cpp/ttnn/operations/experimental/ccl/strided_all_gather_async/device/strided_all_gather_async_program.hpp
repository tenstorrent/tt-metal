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

// Kernel push order and runtime-arg layout of the pipeline appended by
// strided_all_gather_async_minimal_default_helper. strided_all_gather_async_patch_runtime_args writes at
// these positions, so they must track the helper in lockstep. Starting at the helper's first kernel index:
//   [reader, writer] per worker pair, pair = (link * kNumDirectionsPerLink + dir) * num_workers + worker,
//   then one matmul-signal aggregator per direction (fused op, when the aggregators are in use),
//   then the fabric mux kernels, whose count differs per device.
namespace strided_all_gather_async_layout {
inline constexpr uint32_t kNumDirectionsPerLink = 2;
inline constexpr uint32_t kNumMuxCoresPerDirectionPerLink = 1;

// Reader runtime args: [0] input address, [1] output address, ..., [9] out-ready semaphore address.
inline constexpr uint32_t kReaderInputAddrArg = 0;
inline constexpr uint32_t kReaderOutputAddrArg = 1;
inline constexpr uint32_t kReaderSemaphoreArg = 9;
// Writer runtime args: [0] output address, ..., [11] out-ready semaphore address.
inline constexpr uint32_t kWriterOutputAddrArg = 0;
inline constexpr uint32_t kWriterSemaphoreArg = 11;
// Fused-op writers end with [writer_signals_mm, aggregator noc x, aggregator noc y, aggregator semaphore].
inline constexpr uint32_t kWriterAggregatorTailSize = 4;
inline constexpr uint32_t kWriterAggregatorTailSignalsOffset = 0;
inline constexpr uint32_t kWriterAggregatorTailSemaphoreOffset = 3;
// Aggregator runtime args: 6 header words, ring_size k-block counts, then one semaphore address per AG worker.
inline constexpr uint32_t kAggregatorHeaderArgs = 6;
// semaphore[dir] is the out-ready semaphore of direction dir; the per-worker aggregator semaphores follow,
// direction-major: semaphore[kAggregatorSemaphoreBase + dir * num_ag_workers + global_worker_id].
inline constexpr uint32_t kAggregatorSemaphoreBase = kNumDirectionsPerLink;

inline uint32_t worker_pair_index(uint32_t link, uint32_t dir, uint32_t worker, uint32_t num_workers_per_direction) {
    return (((link * kNumDirectionsPerLink) + dir) * num_workers_per_direction) + worker;
}
inline uint32_t reader_kernel_index(uint32_t first_kernel_index, uint32_t pair) {
    return first_kernel_index + (2 * pair);
}
inline uint32_t writer_kernel_index(uint32_t first_kernel_index, uint32_t pair) {
    return first_kernel_index + (2 * pair) + 1;
}
inline uint32_t aggregator_kernel_index(
    uint32_t first_kernel_index, uint32_t num_links, uint32_t num_workers_per_direction, uint32_t dir) {
    return first_kernel_index + (2 * num_links * kNumDirectionsPerLink * num_workers_per_direction) + dir;
}
}  // namespace strided_all_gather_async_layout

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

// Workers per direction per link. Shared by the descriptor build and the cache-hit patch so both see the
// same kernel layout.
uint32_t strided_all_gather_async_num_workers_per_direction(
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t output_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    std::optional<uint32_t> num_workers_per_direction_opt);

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
