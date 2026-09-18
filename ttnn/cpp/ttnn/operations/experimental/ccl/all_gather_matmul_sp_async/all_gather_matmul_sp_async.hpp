// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Default number of bottom core-grid rows reserved for the all-gather workers (the matmul gets the rows above).
// Callers that pass `ccl_core_rows` positionally should use this constant rather than a literal.
// Measured on the 1x4 Blackhole galaxy (Llama-8B TP4 shapes, 2 links): 2 rows let the all-gather run its default
// 4 workers/link (185 us, vs 235 us with the 2 workers that fit one row) while the sub-batched matmul still uses
// 8 rows (per_core_M is unchanged); 3 rows slow the matmul down (per_core_M=3).
inline constexpr uint32_t kDefaultAllGatherMatmulSpCclCoreRows = 2;

// Fused sequence-parallel "gather-then-multiply" (dim 2):
//   input [B,1,S/T,K] (this rank's sequence shard), weight [1,1,K,N] (or [1,1,N,K] with transpose_b)
//   -> {gathered [B,1,S,K], mm = gathered @ weight (+ bias) [B,1,S,N]}, T = mesh extent along cluster_axis.
// The matmul processes one (batch, sequence slice) sub-batch per iteration: the local slice first (read from the
// input), then each remote slice as soon as the all-gather has delivered it. Both outputs are allocated by the op
// (the gathered tensor is returned because the weight gradient needs it). `multi_device_global_semaphore`: 2
// semaphores, as all_gather_async. Ring is demoted to Linear when the axis is not wrap-wired.
std::vector<Tensor> all_gather_matmul_sp_async(
    const Tensor& input,
    const Tensor& weight,
    uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    bool transpose_b = false,
    const std::optional<const Tensor>& bias = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    uint32_t ccl_core_rows = kDefaultAllGatherMatmulSpCclCoreRows,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt);

// The matmul schedule rank `ring_index` of `T` uses (see sp_matmul_fusion_common::sp_ag_schedule), one row per
// matmul iteration: [in0_idx, out_idx, wait_dir, wait_count, is_local]. Exposed for tests.
std::vector<std::vector<uint32_t>> all_gather_matmul_sp_ag_schedule(
    ttnn::ccl::Topology topology, uint32_t ring_size, uint32_t ring_index, uint32_t batch);

}  // namespace ttnn::experimental
