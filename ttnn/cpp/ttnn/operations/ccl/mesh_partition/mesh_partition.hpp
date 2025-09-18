// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::ccl {

struct ExecuteMeshPartition {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        std::optional<uint32_t> cluster_axis,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::ccl

constexpr auto mesh_partition =
    ttnn::register_operation<"ttnn::mesh_partition", ttnn::operations::ccl::ExecuteMeshPartition>();  // namespace
                                                                                                      // ttnn

}  // namespace ttnn
