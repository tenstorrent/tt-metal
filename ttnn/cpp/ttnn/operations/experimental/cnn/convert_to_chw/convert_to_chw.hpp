// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

ttnn::Tensor convert_to_chw(const Tensor& input, const std::optional<DataType>& dtype = std::nullopt);

}  // namespace ttnn::experimental
