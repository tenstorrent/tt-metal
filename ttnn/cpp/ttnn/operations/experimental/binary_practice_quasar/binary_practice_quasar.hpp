// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

// out = a + b for two 2D BFLOAT16 TILE tensors of the same shape (no broadcast).
ttnn::Tensor binary_practice_quasar(const ttnn::Tensor& a, const ttnn::Tensor& b);

}  // namespace ttnn::operations::experimental
