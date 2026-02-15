// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/non_zero_indices.hpp"

#include <tt_stl/vector_init.hpp>
#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation.hpp"

namespace ttnn::operations::data_movement::nonzero {

std::vector<Tensor> ExecuteNonZeroIndices::invoke(
    const Tensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    auto input_memory_config = memory_config.value_or(input_tensor.memory_config());
    auto [output_0, output_1] = ttnn::prim::nonzero(input_tensor, input_memory_config);
    return ttsl::vector_init<Tensor>(std::move(output_0), std::move(output_1));
}

}  // namespace ttnn::operations::data_movement::nonzero
