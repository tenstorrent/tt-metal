// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

ttnn::Tensor narrow(const ttnn::Tensor& input_tensor, int32_t narrow_dim, int32_t narrow_start, uint32_t length);

}  // namespace ttnn
