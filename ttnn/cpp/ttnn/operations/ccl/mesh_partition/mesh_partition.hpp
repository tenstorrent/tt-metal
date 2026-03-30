// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "ttnn/types.hpp"

namespace ttnn {

ttnn::Tensor mesh_partition(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
