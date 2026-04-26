// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::fused_paged_update_and_mla {

struct FusedPagedUpdateAndMlaOperation {
    static Tensor invoke(const Tensor& input_tensor, const Tensor& kv_cache);
};

} // namespace ttnn::operations::experimental::fused_paged_update_and_mla

namespace ttnn::experimental {
    constexpr auto fused_paged_update_and_mla = ttnn::register_operation<
        "ttnn::experimental::fused_paged_update_and_mla",
        ttnn::operations::experimental::fused_paged_update_and_mla::FusedPagedUpdateAndMlaOperation>();
}
