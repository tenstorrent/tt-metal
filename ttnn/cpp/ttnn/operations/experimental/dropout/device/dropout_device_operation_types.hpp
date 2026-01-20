// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DropoutParams {
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;

    // Specifies the seed for the dropout operation.
    // If `use_per_device_seed` is true, the seed is offset by device ID across devices in a mesh.
    uint32_t seed = 0;
    bool use_per_device_seed = false;

    const float prob = 0.0f;
    const float scale = 1.0f;
};

struct DropoutInputs {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
