// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/cumprod_device_operation.hpp"
#include "cumprod.hpp"

namespace ttnn::operations::experimental::reduction {

// TODO(jbbieniek): add doc
Tensor CumprodOperation::invoke(const Tensor& input_tensor, int64_t dim) {
    return ttnn::prim::cumprod(input_tensor, dim);
}

}  // namespace ttnn::operations::experimental::reduction
