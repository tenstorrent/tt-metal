// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>
#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operation.hpp"
#include <tt-metalium/device.hpp>
#include <tt_stl/type_name.hpp>

namespace tt::tt_metal::operation {

template <class OutputTensors = Tensors>
OutputTensors run(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {});

template <typename ConcreteOperation>
inline auto run(
    ConcreteOperation&& concrete_op,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {}) -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        auto operation = DeviceOperation(std::forward<ConcreteOperation>(concrete_op));
        return run<OutputTensors>(std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors);
    } else {
        static_assert(tt::stl::concepts::always_false_v<ConcreteOperation>, "Unsupported Operation");
    }
}

namespace detail {
distributed::MeshDevice* get_device(
    const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors = {});
}

}  // namespace tt::tt_metal::operation
