// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort.hpp"

#include "ttnn/types.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "tt-metalium/logger.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"

namespace ttnn::operations::experimental::reduction {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple) {
    return std::apply(
        [](auto&&... elems) { return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...}; },
        std::forward<Tuple>(tuple));
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::vector<Tensor> ExecuteSort::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const int8_t dim,
    const bool descending,
    const bool stable,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
    // TODO: DEBUG:
    tt::log_error("1. invoke !!!!!!!!!!!!");
    // ---
    auto aa = optional_output_tensors.has_value()
                  ? CMAKE_UNIQUE_NAMESPACE::tuple_to_vector_optional(optional_output_tensors.value())
                  : std::vector<std::optional<Tensor>>{};
    std::vector<Tensor> a = {input_tensor, input_tensor};
    return a;
}

std::vector<Tensor> ExecuteSort::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& input_tensor = input_tensors.at(0);
    return {
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})),
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
}

}  // namespace ttnn::operations::experimental::reduction
