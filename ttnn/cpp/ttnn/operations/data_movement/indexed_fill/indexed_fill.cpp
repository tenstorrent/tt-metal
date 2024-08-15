// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/indexed_fill.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_op.hpp"
#include "ttnn/multi_device.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::data_movement{

ttnn::Tensor IndexedFillOperation::invoke(uint8_t queue_id, const ttnn::Tensor& batch_id, const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b, const std::optional<ttnn::MemoryConfig>& memory_config, int64_t dim) {
    auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
    return operation::run_without_autoformat(IndexedFill{output_memory_config, dim}, {batch_id, input_tensor_a, input_tensor_b}, {}, {}, queue_id).at(0);
}

ttnn::Tensor IndexedFillOperation::invoke(const ttnn::Tensor& batch_id, const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b, const std::optional<ttnn::MemoryConfig>& memory_config, int64_t dim) {
    return invoke(DefaultQueueId, batch_id, input_tensor_a, input_tensor_b, memory_config, dim);
}

}  // namespace ttnn::operations::data_movement
