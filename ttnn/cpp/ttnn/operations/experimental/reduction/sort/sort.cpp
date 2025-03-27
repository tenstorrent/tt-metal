// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort.hpp"
#include "device/sort_device_operation.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"

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
    auto output_vectors = tt::tt_metal::operation::run(
        SortDeviceOperation{dim, descending, stable, memory_config.value_or(input_tensor.memory_config())},
        {input_tensor},
        {},
        optional_output_tensors.has_value()
            ? CMAKE_UNIQUE_NAMESPACE::tuple_to_vector_optional(optional_output_tensors.value())
            : std::vector<std::optional<Tensor>>{},
        queue_id);

    return output_vectors;
}

std::vector<Tensor> ExecuteSort::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& input_tensor = input_tensors.at(0);
    return {
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})),
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
}

}  // namespace ttnn::operations::experimental::reduction
