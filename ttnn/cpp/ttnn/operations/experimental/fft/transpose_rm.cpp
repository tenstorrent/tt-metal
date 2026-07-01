// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fft/transpose_rm.hpp"

#include "device/transpose_rm_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor transpose_rm(const ttnn::Tensor& input) {
    return ttnn::prim::transpose_rm(input);
}

}  // namespace ttnn::operations::experimental
