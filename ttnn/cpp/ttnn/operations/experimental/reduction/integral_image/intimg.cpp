// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/intimg_device_operation.hpp"

#include "intimg.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn::experimental {

Tensor intimg(const Tensor& input_tensor) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::experimental::intimg");
    return ttnn::prim::intimg(input_tensor);
}

}  // namespace ttnn::experimental
