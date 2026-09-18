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

// Fused sequence-parallel "multiply-then-scatter" (dim 2):
//   input [B,1,S,K] (this rank's K shard), weight [1,1,K,N] (or [1,1,N,K] with transpose_b)
//   -> reduce_scatter(input @ weight, dim=2, cluster_axis) : [B,1,S/T,N], T = mesh extent along cluster_axis.
// The full [B,1,S,N] partial and the reduce-scatter intermediates are allocated by the op and freed on return;
// nothing is caller-owned. `multi_device_global_semaphore` takes 3 semaphores (as reduce_scatter_minimal_async).
// No bias: add it after the reduction.
Tensor matmul_reduce_scatter_sp_async(
    const Tensor& input,
    const Tensor& weight,
    uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    bool transpose_b = false,
    std::optional<uint32_t> num_links = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    uint32_t ccl_core_rows = 2,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    // Measurement knob (perf decomposition only): the reduce-scatter waits for the whole matmul instead of
    // starting on the first finished slice. Same result, no overlap.
    bool debug_serialize_reduce_scatter = false);

// Order in which rank `ring_index`'s reduce-scatter readers first touch their local input slices (see
// rs_first_touch_order in the program factory). Exposed for tests; used by the SP schedule.
std::vector<uint32_t> matmul_reduce_scatter_sp_rs_first_touch_order(
    ttnn::ccl::Topology topology, uint32_t ring_size, uint32_t ring_index);

}  // namespace ttnn::experimental
