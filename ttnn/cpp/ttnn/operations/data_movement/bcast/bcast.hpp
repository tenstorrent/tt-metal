// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "bcast_types.hpp"

namespace ttnn {

namespace operations::data_movement {

struct BcastOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        ttnn::BcastOpMath bcast_op,
        ttnn::BcastOpDim bcast_dim,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto bcast = ttnn::register_operation<"ttnn::bcast", ttnn::operations::data_movement::BcastOperation>();

}  // namespace ttnn
