// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor indexed_entry_copy(
    const Tensor& source,
    const Tensor& destination,
    const Tensor& index,
    uint32_t index_scale,
    uint32_t index_offset);

}  // namespace ttnn::experimental::kda
