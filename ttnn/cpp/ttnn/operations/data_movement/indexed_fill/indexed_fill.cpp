// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/indexed_fill.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_op.hpp"
#include "ttnn/common/queue_id.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor IndexedFillOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& batch_id,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    int64_t dim) {
    auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
    return operation::run_without_autoformat(
               IndexedFill{output_memory_config, dim}, {batch_id, input_tensor_a, input_tensor_b}, {}, {}, queue_id)
        .at(0);
}

}  // namespace ttnn::operations::data_movement
