// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ReshapeOnDeviceParams {
    tt::tt_metal::Shape logical_output_shape;
    tt::tt_metal::Shape padded_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("logical_output_shape", "padded_output_shape", "output_mem_config");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(logical_output_shape), std::cref(padded_output_shape), std::cref(output_mem_config));
    }
};

struct ReshapeOnDeviceInputs {
    tt::tt_metal::Tensor input_tensor;

    ReshapeOnDeviceInputs() = default;
    explicit ReshapeOnDeviceInputs(tt::tt_metal::Tensor tensor) : input_tensor(std::move(tensor)) {}

    static constexpr auto attribute_names = std::forward_as_tuple("dtype", "memory_config", "layout", "padded_shape");
    auto attribute_values() const {
        return std::make_tuple(
            input_tensor.dtype(),
            std::cref(input_tensor.memory_config()),
            input_tensor.layout(),
            std::cref(input_tensor.padded_shape()));
    }
};

}  // namespace ttnn::prim
