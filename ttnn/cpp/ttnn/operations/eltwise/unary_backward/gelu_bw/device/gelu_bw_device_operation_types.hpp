// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::operations::unary_backward::gelu_bw {

struct GeluBwParams {
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const operations::unary::GeluVariant variant = operations::unary::GeluVariant::ACCURATE;
};

struct GeluBwInputs {
    const Tensor& grad_output;
    const Tensor& input;
    std::optional<Tensor> preallocated_input_grad;
};

}  // namespace ttnn::operations::unary_backward::gelu_bw
