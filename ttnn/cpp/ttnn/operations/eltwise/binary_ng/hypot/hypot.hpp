
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::binary_ng {

struct Hypot {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output_tensor = std::nullopt);
};

}  // namespace ttnn::operations::binary_ng

namespace ttnn::experimental {

constexpr auto hypot =
    ttnn::register_operation_with_auto_launch_op<"ttnn::experimental::hypot", ttnn::operations::binary_ng::Hypot>();
}  // namespace ttnn::experimental
