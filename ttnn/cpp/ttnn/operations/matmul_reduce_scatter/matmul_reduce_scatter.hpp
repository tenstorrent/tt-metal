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

// Fused multi-device matmul + reduce-scatter (SUM) over one mesh axis: every device of a group holds a K-slice
// A[g] (..., M, K) and W[g] (K, N); the output is the group sum of A[g] @ W[g], scattered along scatter_dim.
//
//   input_tensor, weight   DRAM interleaved, TILE; A rank 2-4 with leading dims 1, W (K, N)
//   cluster_axis           mesh axis of the group (0 or 1)
//   scatter_dim            -1 or -2 (non-negative values count from the front)
//   topology               Linear or Ring (a 2-device ring is the line)
//   num_links              1 or 2; default every usable link
//   compute_kernel_config  math fidelity / approx mode (default HiFi2); fp32 DEST is always on
//   memory_config          interleaved output placement (default DRAM interleaved); sharded is refused
//   output_tensor          preallocated bf16 output matching the spec above; may not alias an input
Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim = -2,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    const std::optional<int32_t>& num_links = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn
