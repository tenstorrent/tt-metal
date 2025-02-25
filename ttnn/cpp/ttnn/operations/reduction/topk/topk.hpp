// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"

#include "device/topk_op.hpp"
#include "ttnn/types.hpp"

template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple) {
    return std::apply(
        [](auto&&... elems) { return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...}; },
        std::forward<Tuple>(tuple));
}
namespace ttnn {
namespace operations::reduction {

struct ExecuteTopK {
    static inline std::vector<Tensor> invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const uint16_t k,
        const int8_t dim,
        const bool largest,
        const bool sorted,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<CoreRangeSet>& sub_core_grids,
        std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors = std::nullopt) {
        CoreRangeSet sub_core_grids_value = sub_core_grids.value_or(ttnn::CoreRangeSet(
            ttnn::CoreRange(ttnn::CoreCoord(0, 0), input_tensor.device()->compute_with_storage_grid_size())));
        tt::log_info("sub_core_grids_value: {}", sub_core_grids_value);
        return operation::run(
            TopK{k, dim, largest, sorted, memory_config.value_or(input_tensor.memory_config()), sub_core_grids_value},
            {input_tensor},
            {},
            optional_output_tensors.has_value() ? tuple_to_vector_optional(optional_output_tensors.value())
                                                : std::vector<std::optional<Tensor>>{},
            queue_id);
    }

    static inline auto invoke(
        const Tensor& input_tensor,
        const uint16_t k,
        const int8_t dim,
        const bool largest,
        const bool sorted,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<CoreRangeSet>& sub_core_grids,
        std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
        return invoke(
            DefaultQueueId,
            input_tensor,
            k,
            dim,
            largest,
            sorted,
            memory_config,
            sub_core_grids,
            optional_output_tensors);
    }

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {
            Tensor(operation::get_workers_for_op_output({input_tensor})),
            Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }
};

}  // namespace operations::reduction

constexpr auto topk =
    ttnn::register_operation_with_auto_launch_op<"ttnn::topk", ttnn::operations::reduction::ExecuteTopK>();

}  // namespace ttnn
