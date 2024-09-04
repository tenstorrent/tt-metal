// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::transformer {

struct SplitQueryKeyValueAndSplitHeadsOperation {
    static std::tuple<Tensor, Tensor, Tensor> invoke(
        const Tensor& input_tensor,
        const std::optional<Tensor>& input_tensor_kv,
        const uint32_t num_heads,
        const std::optional<uint32_t> num_kv_heads,
        const bool transpose_key,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace operations::transformer

namespace transformer {
constexpr auto split_query_key_value_and_split_heads = ttnn::register_operation_with_auto_launch_op<
    "ttnn::transformer::split_query_key_value_and_split_heads",
    ttnn::operations::transformer::SplitQueryKeyValueAndSplitHeadsOperation>();

} // namespace transformer
} // namespace ttnn
