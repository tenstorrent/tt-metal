// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteTilize {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const ttnn::Tensor &input_tensor,
                               const std::optional<MemoryConfig> &memory_config = std::nullopt,
                               std::optional<DataType> output_dtype = std::nullopt,
                               bool use_multicore = false);

    static ttnn::Tensor invoke(const ttnn::Tensor &input_tensor,
                               const std::optional<MemoryConfig> &memory_config = std::nullopt,
                               std::optional<DataType> output_dtype = std::nullopt,
                               bool use_multicore = false);
};

}  // namespace operations::data_movement

constexpr auto tilize =
    ttnn::register_operation_with_auto_launch_op<"ttnn::tilize", ttnn::operations::data_movement::ExecuteTilize>();

}  // namespace ttnn
