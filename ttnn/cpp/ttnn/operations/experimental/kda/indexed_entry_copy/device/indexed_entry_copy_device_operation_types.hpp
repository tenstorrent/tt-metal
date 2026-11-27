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
    // A second (source, destination) pair copied by the same program with the same entry, e.g. a carry's two parts:
    // one dispatch instead of two.
    std::optional<Tensor> paired_source;
    std::optional<Tensor> paired_destination;
};

}  // namespace ttnn::experimental::prim
