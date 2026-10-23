// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "indexed_page_cache.hpp"

#include "device/indexed_fused_update_cache/indexed_fused_update_cache_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> indexed_fused_update_cache(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const Tensor& physical_update_idxs_tensor) {
    return ttnn::prim::indexed_fused_update_cache(
        cache_tensor1, input_tensor1, cache_tensor2, input_tensor2, physical_update_idxs_tensor);
}

}  // namespace ttnn::experimental
