// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/indexed_entry_copy/indexed_entry_copy.hpp"

#include "device/indexed_entry_copy_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor indexed_entry_copy(
    const Tensor& source,
    const Tensor& destination,
    const Tensor& index,
    uint32_t index_scale,
    uint32_t index_offset,
    const std::optional<Tensor>& zero_fill_at_zero) {
    return ttnn::experimental::prim::indexed_entry_copy(
        source, destination, index, index_scale, index_offset, zero_fill_at_zero);
}

}  // namespace ttnn::experimental::kda
