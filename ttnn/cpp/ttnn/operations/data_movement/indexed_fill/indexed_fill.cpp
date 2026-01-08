// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/indexed_fill.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor IndexedFillOperation::invoke(
    const ttnn::Tensor& batch_id,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    int64_t dim) {
    auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
    return ttnn::prim::indexed_fill(batch_id, input_tensor_a, input_tensor_b, output_memory_config, dim);
}

}  // namespace ttnn::operations::data_movement
