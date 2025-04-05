// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "bcast_types.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
enum class BcastOpDim;
enum class BcastOpMath;

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
