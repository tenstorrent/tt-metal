// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization::detail {

// The normalization kernels walk a TILE tensor's tiles on the default tile grid of its logical shape and reduce over
// its padded width or height. An interleaved tensor whose padded shape has whole tiles past that grid (e.g. from
// tilize_with_val_padding) is returned rebuilt on it; any other tensor is returned as is.
Tensor on_default_tile_grid(const Tensor& tensor);

}  // namespace ttnn::operations::normalization::detail
