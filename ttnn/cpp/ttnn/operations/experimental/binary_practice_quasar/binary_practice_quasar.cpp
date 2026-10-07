// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_practice_quasar.hpp"
#include "device/binary_practice_quasar_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor binary_practice_quasar(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    return ttnn::prim::binary_practice_quasar(a, b);
}

}  // namespace ttnn::operations::experimental
