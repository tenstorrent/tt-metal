// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Fused multi-device matmul + reduce-scatter (SUM) over one mesh axis: every device holds a K-slice A[g] (..., M, K)
// and W[g] (K, N); the output is the sum over the group, scattered along scatter_dim.
//
//   input_tensor           A, rank 2-4 with leading dims 1, bfloat16 / bfloat8_b, TILE, DRAM or L1 interleaved
//   weight                 W (K, N), bfloat16 / bfloat8_b / bfloat4_b, TILE, DRAM or L1 interleaved
//   cluster_axis           mesh axis of the group (0 or 1)
//   scatter_dim            -1 / -2 (or the matching non-negative dim)
//   topology               Linear or Ring (a 2-device ring is the line)
//   num_links              links per hop (1 or 2); when absent, every usable link up to 2
//   compute_kernel_config  math fidelity and approx mode (fp32 DEST and full sync are forced on); default HiFi2
//   memory_config          interleaved output placement; default DRAM
//   output_tensor          preallocated output; must match the spec the op allocates, never an input
//   l1_headroom            bytes of each core's L1 the plan leaves free for tensors the op does not own (the
//                          planner does not see them); default 64 KiB
//   staging_tensor         caller-held staging (matmul_reduce_scatter_prefill_staging): the op then allocates none per
//                          call. Any size at least the call's need serves, so one sized for the largest shape serves
//                          all; a larger one counts at its full size, so a smaller shape's plan leaves room for it (it
//                          may pick smaller K blocks or more chunks). Other stagings the caller holds are L1 the op
//                          does not see (l1_headroom). The caller keeps it alive while calls that use it run (a trace
//                          replays into it).
Tensor matmul_reduce_scatter_prefill(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim = -2,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    const std::optional<int32_t>& num_links = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<uint32_t>& l1_headroom = std::nullopt,
    const std::optional<Tensor>& staging_tensor = std::nullopt);

// A staging tensor for calls with these arguments (L1, height-sharded bf16 tiles over the full grid, replicated on the
// mesh), to pass as staging_tensor; the same arguments resolve the same plan as the op.
Tensor matmul_reduce_scatter_prefill_staging(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim = -2,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    const std::optional<int32_t>& num_links = std::nullopt,
    const std::optional<uint32_t>& l1_headroom = std::nullopt);

}  // namespace ttnn::experimental
