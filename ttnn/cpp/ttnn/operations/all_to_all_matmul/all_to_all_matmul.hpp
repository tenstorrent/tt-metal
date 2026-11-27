// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/workload/legacy/program_descriptors.hpp>

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

// q-stem mode (MLA early-Q stem in one call): the input is the unsplit q_b_proj output, per device
//   input_tensor           (1, 1, S, Hl * (K + qk_rope_head_dim)), bfloat16, TILE, DRAM or L1 interleaved; head j's
//                          columns are [j*(K+R), (j+1)*(K+R)): nope (K = weight.shape[2]) then rope
// Returns (prefix, rope_seq) per device:
//   prefix    (1, H, S/G, N): all_to_all_matmul of the nope columns (as if split into (1, Hl, S, K) first)
//   rope_seq  (1, H, S/G, qk_rope_head_dim): the rope columns moved by the same exchange (no matmul), placed by
//             rope_memory_config (interleaved, default DRAM)
// Ring only, tile-aligned S/G; qk_rope_head_dim a positive multiple of 32.
std::tuple<Tensor, Tensor> all_to_all_matmul_qstem(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t qk_rope_head_dim,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring,
    int32_t num_links = 2,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<Tensor>& scratch_tensor = std::nullopt,
    const std::optional<MemoryConfig>& rope_memory_config = std::nullopt);

// The landing scratch a call with these arguments uses; pass it back as scratch_tensor to skip the per-call
// allocation. qk_rope_head_dim: the q-stem call's (nullopt: the plain call's).
Tensor all_to_all_matmul_scratch_tensor(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring,
    int32_t num_links = 2,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<int32_t>& qk_rope_head_dim = std::nullopt);

}  // namespace ttnn
