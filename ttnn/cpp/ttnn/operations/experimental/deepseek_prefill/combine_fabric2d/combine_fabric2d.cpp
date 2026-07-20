// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d.hpp"
#include "device/combine_fabric2d_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice& device,
    uint32_t num_links,
    uint32_t num_tokens,
    uint32_t chunk_size_bytes,
    uint32_t num_slots,
    uint32_t axis,
    uint32_t stall_telemetry,
    uint32_t variant,
    std::optional<tt::tt_fabric::Topology> topology) {
    return ttnn::prim::combine_fabric2d(
        &device,
        num_links,
        num_tokens,
        chunk_size_bytes,
        num_slots,
        axis,
        stall_telemetry,
        variant,
        topology.value_or(tt::tt_fabric::Topology::Mesh));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
