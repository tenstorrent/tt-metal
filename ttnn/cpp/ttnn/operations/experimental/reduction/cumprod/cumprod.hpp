// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::reduction {

struct CumprodOperation {
    static constexpr uint32_t CHANNEL_DIMENSION{1};
    static constexpr uint32_t FIXED_CUMULATION_AXIS{CHANNEL_DIMENSION};
    static constexpr uint32_t FOUR_DIMENSIONS{4};
    using PermVec = SmallVector<int64_t>;
    static const PermVec NATURAL_AXIS_ORDER;

    static Tensor invoke(
        const Tensor& input_tensor,
        const int32_t dim,
        const std::optional<DataType>& input_dtype = std::nullopt,
        const std::optional<Tensor>& optional_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const QueueId& queue_id = DefaultQueueId);

    static bool validate_input(
        const Tensor& input_tensor,
        const int32_t dim,
        const DataType input_dtype,
        const std::optional<Tensor>& optional_out = std::nullopt);
    static std::tuple<Tensor, PermVec> permute_to_4d(const Tensor& input_tensor, const uint32_t& cum_axis);
    static Tensor reorder_from_4d(
        const Tensor& input_tensor,
        const PermVec& permutation,
        const uint32_t& rank,
        const std::optional<Tensor>& optional_out = std::nullopt);
};

}  // namespace operations::experimental::reduction

namespace experimental {
constexpr auto cumprod = ttnn::
    register_operation<"ttnn::experimental::cumprod", ttnn::operations::experimental::reduction::CumprodOperation>();

}  // namespace experimental
}  // namespace ttnn
