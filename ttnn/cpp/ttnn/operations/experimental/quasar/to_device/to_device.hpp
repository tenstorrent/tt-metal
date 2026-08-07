// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<QueueId> queue_id = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
