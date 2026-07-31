// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d.hpp"
#include "device/combine_fabric2d_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice& device,
    const ttnn::Tensor& input,
    const ttnn::Tensor& output,
    const std::vector<CombineFabric2dMovement>& movements,
    uint32_t num_links,
    uint32_t tokens_per_movement,
    uint32_t token_size_bytes,
    uint32_t axis,
    uint32_t num_l1_slots,
    uint32_t fwd_bump_every,
    uint32_t stall_telemetry,
    std::optional<tt::tt_fabric::Topology> topology) {
    return ttnn::prim::combine_fabric2d(
        &device,
        input,
        output,
        movements,
        num_links,
        tokens_per_movement,
        token_size_bytes,
        axis,
        num_l1_slots,
        fwd_bump_every,
        stall_telemetry,
        topology.value_or(tt::tt_fabric::Topology::Mesh));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
