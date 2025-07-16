// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scatter_enums.hpp"

#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::data_movement {

struct ScatterOperation {
    static Tensor invoke(
        const QueueId& queue_id,
        const Tensor& input_tensor,
        const int32_t& dim,
        const Tensor& index_tensor,
        const Tensor& source_tensor,
        const std::optional<MemoryConfig>& opt_out_memory_config,
        const std::optional<scatter::ScatterReductionType>& opt_reduction);
};

}  // namespace operations::data_movement

constexpr auto scatter =
    ttnn::register_operation<"ttnn::experimental::scatter", ttnn::operations::data_movement::ScatterOperation>();

}  // namespace ttnn
