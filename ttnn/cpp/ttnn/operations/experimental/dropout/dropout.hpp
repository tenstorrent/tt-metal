// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

Tensor dropout(const Tensor& input_tensor, float prob, float scale, uint32_t seed, bool use_per_device_seed = true);

}  // namespace ttnn::experimental
