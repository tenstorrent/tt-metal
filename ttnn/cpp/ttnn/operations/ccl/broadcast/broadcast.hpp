// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::ccl {

struct ExecuteBroadcast {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const MeshCoordinate& sender_coord,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);
};

}  // namespace operations::ccl

constexpr auto broadcast = ttnn::register_operation<"ttnn::broadcast", ttnn::operations::ccl::ExecuteBroadcast>();

}  // namespace ttnn
