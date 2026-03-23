// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device/intimg_device_operation.hpp"

#include "intimg.hpp"

namespace ttnn::experimental {

Tensor intimg(const Tensor& input_tensor) { return ttnn::prim::intimg(input_tensor); }

}  // namespace ttnn::experimental
