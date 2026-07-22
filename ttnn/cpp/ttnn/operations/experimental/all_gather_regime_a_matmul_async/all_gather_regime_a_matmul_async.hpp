// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::experimental {

// Fused all-gather(in0, dim=-1) @ in1 using regime_a_matmul as the compute engine.
// (REGIME_A_AGMM_EXECUTION_PLAN.md). Working name; may change in API review.
//
// Task 2 scope: D=1 delegates to (is behaviorally identical to) production regime_a_matmul. The D>1
// fabric-streaming program is implemented in Task 3; for D>1 this validates the host plan and then reports
// that the streaming path is not yet implemented. bf16 only, no transpose/batching, tile-aligned K
// sharding, no epilogues (all validated).
ttnn::Tensor all_gather_regime_a_matmul_async(
    const ttnn::Tensor& input_tensor,   // in0 : [.., M, K_local] (K-sharded across the cluster axis)
    const ttnn::Tensor& weight_tensor,  // in1 : [.., K_global, N] DRAM width-sharded (8 banks)
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    uint32_t num_links = 1,
    uint32_t num_workers_per_link = 1,
    uint32_t num_buffers_per_channel = 2,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore = {},
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
