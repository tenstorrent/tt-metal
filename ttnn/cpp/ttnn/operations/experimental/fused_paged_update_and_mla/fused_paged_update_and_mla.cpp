// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fused_paged_update_and_mla.hpp"

namespace ttnn::operations::experimental::fused_paged_update_and_mla {

// Empty placeholder implementation to satisfy compiler
ttnn::Tensor FusedPagedUpdateAndMlaOperation::invoke(const ttnn::Tensor& input_tensor, const ttnn::Tensor& kv_cache) {
    (void)kv_cache;
    return input_tensor;
}

} // namespace ttnn::operations::experimental::fused_paged_update_and_mla