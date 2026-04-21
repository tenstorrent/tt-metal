// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt::tt_metal {
class CoreRangeSet;
class Tensor;
}

namespace ttnn::experimental::core_subset_write {

void copy_to_device_filtered(
    const tt::tt_metal::Tensor& host_tensor,
    tt::tt_metal::Tensor& device_tensor,
    const tt::tt_metal::CoreRangeSet& logical_core_filter,
    std::optional<tt::tt_metal::QueueId> cq_id = std::nullopt);

}  // namespace ttnn::experimental::core_subset_write
