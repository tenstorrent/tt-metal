// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <optional>

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

struct ReshardOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const MemoryConfig& memory_config,
        const std::optional<Tensor>& optional_output_tensor);
};

}  // namespace operations::data_movement

constexpr auto reshard =
    ttnn::register_operation_with_auto_launch_op<"ttnn::reshard", ttnn::operations::data_movement::ReshardOperation>();
}  // namespace ttnn
