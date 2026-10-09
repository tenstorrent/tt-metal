// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::quasar {

// Quasar clone of ttnn::unsqueeze: inserts a size-1 dimension at `dim` via the quasar reshape.
ttnn::Tensor unsqueeze(const ttnn::Tensor& input_tensor, int dim);

}  // namespace ttnn::operations::experimental::quasar
