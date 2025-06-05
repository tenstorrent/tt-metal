// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scatter_enums.hpp"

#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::experimental {

struct ScatterOperation {
    static Tensor invoke(
        const QueueId& queue_id,
        const Tensor& input_tensor,
        const int32_t& dim,
        const Tensor& index_tensor,
        const Tensor& source_tensor,
        const std::optional<MemoryConfig>& opt_out_memory_config,
        const std::optional<scatter::ScatterReductionType>& opt_reduction,
        std::optional<Tensor>& opt_output);
};

}  // namespace operations::experimental

namespace experimental {
constexpr auto scatter_ =
    ttnn::register_operation<"ttnn::experimental::scatter_", ttnn::operations::experimental::ScatterOperation>();
}  // namespace experimental

}  // namespace ttnn
