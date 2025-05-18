// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

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

constexpr auto reshard = ttnn::register_operation<"ttnn::reshard", ttnn::operations::data_movement::ReshardOperation>();
}  // namespace ttnn
