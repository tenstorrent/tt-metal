// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::indexed_fused_update_cache {

struct IndexedFusedUpdateCacheParams {};

struct IndexedFusedUpdateCacheInputs {
    Tensor cache_tensor1;
    Tensor input_tensor1;
    Tensor cache_tensor2;
    Tensor input_tensor2;
    Tensor physical_update_idxs_tensor;
};

using IndexedFusedUpdateCacheResultSpec = std::array<tt::tt_metal::TensorSpec, 2>;
using IndexedFusedUpdateCacheResult = std::tuple<Tensor, Tensor>;

}  // namespace ttnn::experimental::prim::indexed_fused_update_cache
