// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
// #include <tt-metalium/operation.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "device/all_gather_minimal_matmul_async_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include <tt-metalium/global_semaphore.hpp>

namespace ttnn {

std::vector<ttnn::Tensor> all_gather_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<float> scalar,
    const std::optional<ttnn::Tensor>& addcmul_input_tensor1,
    const std::optional<ttnn::Tensor>& addcmul_input_tensor2,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    uint32_t num_links = 1,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    bool force_transpose = true,
    uint32_t num_workers_per_link = 1,
    uint32_t num_buffers_per_channel = 1,
    int32_t chunks = 1,
    int32_t dim = -1,
    std::optional<uint32_t> fsdp_cluster_axis = std::nullopt,
    const std::vector<GlobalSemaphore>& fsdp_multi_device_global_semaphore = {},
    const std::optional<ttnn::Tensor>& persistent_weight_buffer = std::nullopt,
    std::optional<ttnn::ccl::Topology> fsdp_topology = std::nullopt,
    bool fuse_swiglu = false,
    const std::vector<uint32_t>& chunk_sizes = {},
    std::optional<bool> transpose_core_grid = std::nullopt,
    // Tuned Blackhole Fabric2D path: with a 2D multicast matmul schedule the TP all-gather of the input runs inside
    // that matmul's program, on the worker column right of its grid, and the matmul starts on this device's own K
    // slice while the others arrive. persistent_output_buffer is the gathered [M, K] buffer the other devices write
    // into, barrier_semaphore the gather's ready semaphore, multi_device_global_semaphore the ranks - 1 slice
    // counters, and gather_table each device's rows from all_gather_minimal_matmul_async_gather_table.
    const std::optional<const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>& matmul_program_config =
        std::nullopt,
    const std::optional<ttnn::Tensor>& gather_table = std::nullopt);

// Each device's gather_table rows for the tuned path (uint32 rows of 64 words, devices in row-major mesh order): its
// rank, and its gather senders' destinations and fabric connections. matmul_grid is the schedule's compute grid.
std::vector<std::vector<uint32_t>> all_gather_minimal_matmul_async_gather_table(
    const MeshDevice& mesh_device,
    int32_t cluster_axis,
    uint32_t num_links,
    const CoreCoord& matmul_grid,
    const std::vector<GlobalSemaphore>& slice_counters);

}  // namespace ttnn
