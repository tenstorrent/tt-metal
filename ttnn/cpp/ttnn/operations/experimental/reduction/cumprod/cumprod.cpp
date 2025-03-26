// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/core/core.hpp"
#include "device/cumprod_device_operation.hpp"
#include "cumprod.hpp"

namespace ttnn::operations::experimental::reduction {

// TODO(jbbieniek): add doc
Tensor CumprodOperation::invoke(const Tensor& input_tensor, int64_t dim) {
    return ttnn::prim::cumprod(input_tensor, dim);
}

}  // namespace ttnn::operations::experimental::reduction
