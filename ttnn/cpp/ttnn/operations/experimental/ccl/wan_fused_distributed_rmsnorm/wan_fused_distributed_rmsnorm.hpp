// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Composite fused op: pre-allgather RMSNorm stats + all-gather of stats + post-allgather
// RMSNorm (with optional head split / RoPE / dtype cast). Wan2.2 attention path.
//
// First-draft implementation simply chains the three existing ops. The intent is to replace
// this with a single device op in a follow-up.
ttnn::Tensor wan_fused_distributed_rmsnorm(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    float epsilon = 1e-5,
    uint32_t num_heads_per_device = 1,
    bool per_head_norm = false,
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt,
    const std::optional<const ttnn::Tensor>& transformation_mat = std::nullopt,
    const std::optional<const ttnn::Tensor>& rope_cos = std::nullopt,
    const std::optional<const ttnn::Tensor>& rope_sin = std::nullopt,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool use_device_op = false);

// Allocate the persistent stats DRAM scratch buffer required by the all-gather
// path (TP>1, whole-row norm). Returns std::nullopt for shapes that don't
// all-gather (TP=1 or per_head_norm), which reduce locally and need no scratch.
// The caller is expected to hold this tensor across launches — it's a regular
// device tensor allocated as a mesh-coherent MeshBuffer, which is what the
// fabric mcast needs.
std::optional<ttnn::Tensor> wan_fused_distributed_rmsnorm_create_stats_buffer(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    uint32_t num_heads_per_device = 1,
    bool per_head_norm = false,
    // MUST equal the num_links the op is invoked with: the buffer's chunk/window
    // geometry depends on num_workers, which the program rounds to a multiple of
    // num_links. A mismatch silently corrupts multi-row-chunk AG (see .cpp).
    uint32_t num_links = 1,
    // RoPE/weight passed through so the buffer's chunk/window sizing matches the
    // program's per-head-RoPE / streaming chunk clamp (needed for per-head RoPE
    // and wide-feature streaming shapes).
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,
    const std::optional<const ttnn::Tensor>& transformation_mat = std::nullopt,
    const std::optional<const ttnn::Tensor>& rope_cos = std::nullopt,
    const std::optional<const ttnn::Tensor>& rope_sin = std::nullopt);

}  // namespace ttnn::experimental
