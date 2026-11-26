// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// Fused all-to-all (head-sharded -> sequence-sharded) + per-head matmul over a ring of G devices along
// cluster_axis: device q ends with rows [q*S/G, (q+1)*S/G) of every source's chunk for all H = G*Hl heads, each
// head multiplied by its own (K, N) matrix of a replicated weight.
//
//   input_tensor           (B, Hl, S, K), bfloat16, TILE, DRAM interleaved
//   weight                 (1, H, K, N), bfloat16 or bfloat8_b, TILE, interleaved, replicated
//   memory_config          output placement (interleaved, DRAM or L1; default DRAM)
//   compute_kernel_config  default HiFi4, fp32 DEST, no approximation
//   scratch_tensor         landing scratch from all_to_all_matmul_scratch_tensor; allocated per call when absent
//
// Returns (B, H, S/G, N) per device, bfloat16, TILE.
Tensor all_to_all_matmul(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring,
    int32_t num_links = 2,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<Tensor>& scratch_tensor = std::nullopt);

// The landing scratch a call with these arguments uses; pass it back as scratch_tensor to skip the per-call
// allocation.
Tensor all_to_all_matmul_scratch_tensor(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring,
    int32_t num_links = 2,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
