// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct IndexedEntryCopyParams {
    // entry = index[0] * index_scale + index_offset. Structural: part of the program hash, like
    // `num_layers` / `layer_idx` in update_padded_kv_cache, so one program serves every index value.
    uint32_t index_scale;
    uint32_t index_offset;
};

struct IndexedEntryCopyInputs {
    Tensor source;
    Tensor destination;
    Tensor index;
    // Read on device like `index`: 0 fills the destination entry with zeros instead of the source's.
    std::optional<Tensor> zero_fill_at_zero;
};

}  // namespace ttnn::experimental::prim
