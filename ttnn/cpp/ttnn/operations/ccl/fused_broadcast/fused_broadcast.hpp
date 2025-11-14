// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn {
namespace operations::ccl {

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

struct ExecuteFusedBroadcast {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const MeshCoordinate& root_coord = MeshCoordinate{1, 0},
        const MeshCoordinate& mesh_shape = MeshCoordinate{4, 2},
        const tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt,
        uint32_t num_links = 1);
};

}  // namespace operations::ccl

constexpr auto fused_broadcast =
    ttnn::register_operation<"ttnn::fused_broadcast", ttnn::operations::ccl::ExecuteFusedBroadcast>();

}  // namespace ttnn
