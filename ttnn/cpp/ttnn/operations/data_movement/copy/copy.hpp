// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct CopyOperation {
    static ttnn::Tensor invoke(uint8_t queue_id, const Tensor& src_tensor, const Tensor& dst_tensor);

    static ttnn::Tensor invoke(const Tensor& src_tensor, const Tensor& dst_tensor);
};

struct AssignOperation {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const Tensor& input,
                               const MemoryConfig& output_mem_config,
                               std::optional<const DataType> output_dtype = std::nullopt,
                               std::optional<Tensor> optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(const Tensor& input,
                               const MemoryConfig& output_mem_config,
                               std::optional<const DataType> output_dtype = std::nullopt);

    static ttnn::Tensor invoke(uint8_t queue_id, const Tensor& input_a, const Tensor& input_b);

    static ttnn::Tensor invoke(const Tensor& input_a, const Tensor& input_b);
};

}  // namespace operations::data_movement

constexpr auto copy =
    ttnn::register_operation_with_auto_launch_op<"ttnn::copy", ttnn::operations::data_movement::CopyOperation>();
constexpr auto assign =
    ttnn::register_operation_with_auto_launch_op<"ttnn::assign", ttnn::operations::data_movement::AssignOperation>();

}  // namespace ttnn
