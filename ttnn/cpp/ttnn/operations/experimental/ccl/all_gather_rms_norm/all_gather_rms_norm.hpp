// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <tt-metalium/workload/legacy/program_descriptors.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Fused all-gather + RMSNorm / LayerNorm along cluster_axis of the input's mesh: every device of a group holds its
// (..., M, W_local) slice and returns the full normalized (..., M, G * W_local) row.
//
//   input_tensor           TILE, float32 / bfloat16 / bfloat8_b, DRAM interleaved or L1 HEIGHT / WIDTH sharded
//   cluster_axis           mesh axis of the group (0 or 1)
//   gamma, beta            optional full-W weights (TILE [1,1,1,W] or ROW_MAJOR [1,1,W/32,32]); beta: LayerNorm only
//   norm_type              "rms_norm" or "layer_norm"
//   epsilon                per-call scalar; changing it reuses the cached program
//   topology, num_links    fabric topology and links (None: min(usable links, 2))
//   compute_kernel_config  default HiFi4, fp32 DEST accumulation, exact math
//   memory_config          output placement when no output_tensor is given (default DRAM interleaved)
//   output_tensor          preallocated output
//   scratch_tensor         caller-held R_U staging (W_local % 32 != 0); see all_gather_rms_norm_scratch_tensor
Tensor all_gather_rms_norm(
    const Tensor& input_tensor,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma = std::nullopt,
    const std::optional<Tensor>& beta = std::nullopt,
    const std::string& norm_type = "rms_norm",
    double epsilon = 1e-6,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    std::optional<int32_t> num_links = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<Tensor>& scratch_tensor = std::nullopt);

// The staging tensor a call with these arguments uses (only W_local % 32 != 0 calls stage; others get nullopt).
std::optional<Tensor> all_gather_rms_norm_scratch_tensor(const Tensor& input_tensor, uint32_t cluster_axis);

}  // namespace ttnn::experimental
