// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "llama_reduce_scatter.hpp"
#include "ttnn/run_operation.hpp"

namespace operations::experimental::ccl {
namespace detail {}  // namespace detail

ttnn::Tensor ExecuteLLamaReduceScatter::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
    return ttnn::prim(input_tensor, dim, memory_config);
}

}  // namespace operations::experimental::ccl
