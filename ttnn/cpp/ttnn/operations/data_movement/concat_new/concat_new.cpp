// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_new.hpp"

#include "device/concat_new_device_operation.hpp"

namespace ttnn {

ttnn::Tensor concat_new(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    unsigned int groups,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids) {
    TT_FATAL(!input_tensors.empty(), "ttnn.concat_new: expected a non-empty list of Tensors!");
    TT_FATAL(!optional_output_tensor.has_value(), "optional output tensor currently unsupported!");

    const auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::operations::data_movement::concat_new_impl(input_tensors, dim, groups, mem_config, sub_core_grids);
}

}  // namespace ttnn
