// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations::data_movement {

struct NonZeroIndicesOperation {
    static std::vector<ttnn::Tensor> invoke(
        QueueId queue_id, const ttnn::Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config);
};

}  // namespace operations::data_movement

constexpr auto nonzero =
    ttnn::register_operation<"ttnn::nonzero", ttnn::operations::data_movement::NonZeroIndicesOperation>();

}  // namespace ttnn
