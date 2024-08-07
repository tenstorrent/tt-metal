// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include "ttnn/operation.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include <type_traits>
#include <optional>

namespace tt::tt_metal::operation {

template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
constexpr bool is_optional_v = is_optional<T>::value;

template <class T>
Tensor* get_tensor(T& maybe_tensor) {
    Tensor* output_tensor = nullptr;
    if constexpr (is_optional_v<T>) {
        if (maybe_tensor.has_value())
            output_tensor = &maybe_tensor.value();
    } else {
        output_tensor = &maybe_tensor;
    }
    return output_tensor;
}


template<class OutputType>
OutputType launch_op(
    auto&& op_func,
    const Tensors input_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors,
    bool enable_autoformat_device) {
    return op_func(input_tensors, optional_input_tensors, optional_output_tensors);
}
}
