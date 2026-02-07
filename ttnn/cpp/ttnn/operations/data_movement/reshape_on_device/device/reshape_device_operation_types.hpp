// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct ReshapeOnDeviceParams {
    tt::tt_metal::Shape logical_output_shape;
    tt::tt_metal::Shape padded_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ReshapeOnDeviceInputs {
    tt::tt_metal::Tensor input_tensor;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor); }
};

}  // namespace ttnn::prim
