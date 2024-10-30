// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operation.hpp"
#include "tt_stl/concepts.hpp"
#include "tt_stl/type_name.hpp"

namespace tt::tt_metal {

namespace operation {

using ttnn::operations::experimental::auto_format::FormatParams;
template <typename ConcreteOperation>
auto generic_create_output_tensors(
    const ConcreteOperation& operation,
    const Tensors& input_tensors,
    const std::optional<DataType> output_dtype,
    const Layout output_layout,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tile>& tile = std::nullopt) -> ProgramOutputTensors<ConcreteOperation> {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_shapes = operation.compute_output_shapes(input_tensors);

    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    OutputTensors output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(create_device_tensor(
            output_shape,
            output_dtype.value_or(input_tensors.at(0).get_dtype()),
            output_layout,
            input_tensor.device(),
            output_mem_config.value_or(input_tensors.at(0).memory_config()), tile));
    }
    return output_tensors;
}


template<class OutputTensors=Tensors>
OutputTensors run(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0);

template<typename ConcreteOperation>
inline auto run(
    ConcreteOperation&& concrete_op,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors={},
    const OptionalTensors& optional_output_tensors={},
    uint8_t cq_id = 0
) -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        auto operation = DeviceOperation(concrete_op);
        return run<OutputTensors>(std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
    } else {
        static_assert(tt::stl::concepts::always_false_v<ConcreteOperation>, "Unsupported Operation");
    }
}

template<class OutputTensors=Tensors>
OutputTensors run_without_autoformat(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0
);
template <typename ConcreteOperation>
inline auto run_without_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {},
    uint8_t cq_id = 0)
    -> ProgramOutputTensors<ConcreteOperation>{
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_without_autoformat<OutputTensors>(std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
}

Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false,
    uint8_t cq_id = 0
);

template<typename ConcreteOperation>
inline auto run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false,
    uint8_t cq_id = 0
)-> Tensors {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<Tensors>(concrete_op);
    return run_with_autoformat(std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, pad_value, pad_c, cq_id);
}

Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const OptionalConstTensors& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0
);
template<typename ConcreteOperation>
inline auto run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0
)-> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_with_autoformat(std::move(operation), input_tensors, input_formatting, output_layouts, optional_input_tensors, optional_input_formatting, optional_output_tensors, cq_id);
}

template<class Callable, class OutputType=Tensors>
void launch_op(
    Callable&& op_func,
    const Tensors input_tensors,
    OutputType& output_tensors,
    const OptionalConstTensors optional_input_tensors = {},
    const OptionalTensors optional_output_tensors = {},
    bool enable_autoformat_device = true
);

void launch_with_autoformat(
    std::function<Tensors(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>&& op_func,
    const Tensors input_tensors,
    Tensors& output_tensors,
    const OptionalConstTensors optional_input_tensors = {},
    const OptionalTensors optional_output_tensors = {}
);

std::vector<Device*> get_workers_for_op_output(
    const std::vector<Tensor>& inputs,
    const std::vector<std::optional<const Tensor>>& optional_inputs = {},
    bool enable_autoformat_device = true);

namespace detail{
    Device* get_device(const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors = {});
}

} //namespace operation

} //namespace tt::tt_metal

#include"run_operation_inl.hpp"
