// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/combine_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// `input` and `output` are caller-owned interleaved uint32 ROW_MAJOR DRAM mesh tensors, one row per
// token. `movements` says what moves where — one descriptor per fabric cable per device, i.e.
// 2 * num_links * num_devices in total. Returns the same output tensor it was handed.
ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice& device,
    const ttnn::Tensor& input,
    const ttnn::Tensor& output,
    const std::vector<CombineFabric2dMovement>& movements,
    uint32_t num_links = 2,
    uint32_t input_tokens_per_movement = 32,
    uint32_t output_tokens_per_movement = 100,
    uint32_t token_size_bytes = 14336,
    uint32_t axis = 0,
    uint32_t stall_telemetry = 0,
    std::optional<tt::tt_fabric::Topology> topology = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace ttnn {
using operations::experimental::deepseek_prefill::combine_fabric2d::combine_fabric2d;
}  // namespace ttnn
