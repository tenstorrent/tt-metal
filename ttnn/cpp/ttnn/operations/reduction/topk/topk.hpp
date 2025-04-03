// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <functional>
#include <optional>
#include <tuple>
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
namespace operations::reduction {

struct ExecuteTopK {
    static std::vector<Tensor> invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const uint32_t k,
        const int8_t dim,
        const bool largest,
        const bool sorted,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors = std::nullopt);

    static std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs);
};

}  // namespace operations::reduction

constexpr auto topk =
    ttnn::register_operation_with_auto_launch_op<"ttnn::topk", ttnn::operations::reduction::ExecuteTopK>();

}  // namespace ttnn
