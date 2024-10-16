// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {

struct ExecuteRepeatInterleave {
    // # This operation does not support the following cases:
    // #   - Shape([2[32], 2[32]]) -> repeats = 2, dim = 0
    // #   - Shape([2[32], 2[32]]) -> repeats = Tensor[1,2], dim = 1

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor,
                               uint32_t repeats,
                               int32_t dim,
                               std::optional<MemoryConfig> output_mem_config = std::nullopt);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto repeat_interleave =
    ttnn::register_operation_with_auto_launch_op<"ttnn::repeat_interleave",
                                                 ttnn::operations::data_movement::ExecuteRepeatInterleave>();

}  // namespace ttnn
