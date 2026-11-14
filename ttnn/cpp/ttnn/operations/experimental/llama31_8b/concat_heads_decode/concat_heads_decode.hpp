// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

ttnn::Tensor concat_heads_decode(const ttnn::Tensor& input_tensor, uint32_t num_heads);

}  // namespace ttnn::experimental::llama31_8b
