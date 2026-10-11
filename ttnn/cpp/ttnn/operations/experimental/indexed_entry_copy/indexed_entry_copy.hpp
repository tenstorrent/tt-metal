// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

ttnn::Tensor indexed_entry_copy(
    const Tensor& source,
    const Tensor& destination,
    const Tensor& index,
    uint32_t index_scale = 1,
    uint32_t index_offset = 0);

}  // namespace ttnn::experimental
