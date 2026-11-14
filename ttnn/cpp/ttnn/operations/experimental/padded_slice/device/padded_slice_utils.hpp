// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::detail {

uint32_t get_num_cores_channels_from_sharded_tensor(const Tensor& tensor);

// Bit pattern of one output element holding pad_value, in the low element_size bytes of the result.
// An unset pad_value is zero.
uint32_t pad_value_to_element_bits(const std::optional<float>& pad_value, DataType output_dtype);

}  // namespace ttnn::operations::experimental::detail
