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

// Fused multi-device matmul + reduce-scatter (SUM) over one mesh axis: every device of a group along
// cluster_axis holds a K-slice A[g] (..., M, K) and W[g] (K, N); device p of the group gets block p of
// sum_g A[g] @ W[g], split along scatter_dim (-2: M, -1: N).
//
//   input_tensor           A, rank 2-4 with leading dims 1, bfloat16 / bfloat8_b, DRAM interleaved, tiled
//   weight                 W (K, N), bfloat16 / bfloat8_b / bfloat4_b, DRAM interleaved, tiled
//   cluster_axis           mesh axis of the groups (0 or 1)
//   scatter_dim            -2 or -1 (or their non-negative equivalents)
//   topology               Linear or Ring (a 2-device ring runs as a line)
//   num_links              1 or 2; std::nullopt uses every usable link
//   compute_kernel_config  math fidelity and approx mode (fp32 DEST and full sync are always on); default HiFi2
//   memory_config          output placement, interleaved only; default DRAM
//   output_tensor          preallocated output matching the spec the op would allocate (in place refused)
//
// Refusals throw UnsupportedAxisValue / NotImplementedRefusal (device/..._types.hpp) or std::invalid_argument.
Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim = -2,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    std::optional<int32_t> num_links = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn
