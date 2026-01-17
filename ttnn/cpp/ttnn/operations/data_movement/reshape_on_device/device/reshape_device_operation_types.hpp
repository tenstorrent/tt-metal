// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_on_device {

struct ReshapeOnDeviceParams {
    tt::tt_metal::Shape logical_output_shape;
    tt::tt_metal::Shape padded_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ReshapeOnDeviceInputs {
    tt::tt_metal::Tensor input_tensor;
};

using tensor_return_value_t = tt::tt_metal::Tensor;

}  // namespace ttnn::operations::data_movement::reshape_on_device
