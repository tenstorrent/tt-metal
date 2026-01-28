// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct PadParams {
    ttnn::Shape output_logical_shape;
    ttnn::Shape output_padded_shape;
    ttnn::Shape input_tensor_start;
    float pad_value{};
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_multicore{};
};

struct PadInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
