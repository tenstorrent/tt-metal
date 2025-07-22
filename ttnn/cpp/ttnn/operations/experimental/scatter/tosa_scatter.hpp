// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::experimental {

struct TOSAScatterOperation {
    static Tensor invoke(
        const QueueId& queue_id,
        const Tensor& input_tensor,
        const Tensor& index_tensor,
        const Tensor& source_tensor,
        const std::optional<MemoryConfig>& opt_out_memory_config);
};

}  // namespace operations::experimental

namespace experimental {
constexpr auto tosa_scatter = ttnn::
    register_operation<"ttnn::experimental::tosa_scatter", ttnn::operations::experimental::TOSAScatterOperation>();
}  // namespace experimental

}  // namespace ttnn
