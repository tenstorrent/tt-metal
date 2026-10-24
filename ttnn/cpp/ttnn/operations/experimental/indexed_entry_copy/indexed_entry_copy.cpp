// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/indexed_entry_copy/indexed_entry_copy.hpp"

#include "device/indexed_entry_copy_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor indexed_entry_copy(
    const Tensor& source, const Tensor& destination, const Tensor& index, uint32_t index_scale, uint32_t index_offset) {
    return ttnn::prim::indexed_entry_copy(source, destination, index, index_scale, index_offset);
}

}  // namespace ttnn::experimental
