// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// In-place update of four UINT32 cache bundle metadata tensors; returns the input page_table.
// The caller must provide valid metadata and reserve sufficient capacity before calling.
Tensor update_cache_bundle_allocation(
    const Tensor& page_table,
    const Tensor& allocated_pages,
    const Tensor& free_list,
    const Tensor& free_count,
    uint32_t slot_id,
    uint32_t actual_start,
    uint32_t actual_end,
    uint32_t page_size = 32);

}  // namespace ttnn::experimental
