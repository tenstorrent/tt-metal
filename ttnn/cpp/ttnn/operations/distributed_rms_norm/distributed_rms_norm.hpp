// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// RMSNorm over a hidden dim sharded across one mesh axis: every device p of a group of G devices along cluster_axis
// returns x[p] * rsqrt(Σ_g Σ_w x[g]^2 / (G * W_local) + epsilon) * gamma[p], the Σx² partials exchanged over the
// fabric.
//
//   cluster_axis           0 or 1, the mesh axis the hidden dim is split over
//   gamma                  optional, TILE (1,1,1,W) or ROW_MAJOR sticks (1,1,ceil(W/32),32), float32 / bfloat16
//   epsilon                per-call scalar; changing it reuses the cached program
//   topology               Linear or Ring
//   num_links              links per hop (None: every usable link)
//   compute_kernel_config  default HiFi4, fp32 DEST, exact math
//   memory_config          output placement (default: DRAM for an interleaved input, else the input's)
//   exchange_mode          "direct" (default), "relay" or "mcast"
//   output_tensor          preallocated output, matching the spec the op would allocate; not the input
//   scratch_tensor         the exchange landing (distributed_rms_norm_scratch_tensor); None: allocated per call
Tensor distributed_rms_norm(
    const Tensor& input_tensor,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma = std::nullopt,
    double epsilon = 1e-6,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    std::optional<int32_t> num_links = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::string>& exchange_mode = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<Tensor>& scratch_tensor = std::nullopt);

// The exchange landing a call with these arguments uses; a caller that holds it pays no allocation per call.
Tensor distributed_rms_norm_scratch_tensor(
    const Tensor& input_tensor,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma = std::nullopt,
    double epsilon = 1e-6,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    std::optional<int32_t> num_links = std::nullopt,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::string>& exchange_mode = std::nullopt);

}  // namespace ttnn
