// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::ssm::repeat_mul {

struct RepeatMulParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
    const MathFidelity math_fidelity;
};

struct RepeatMulInputs {
    const Tensor& a;
    const Tensor& b;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::experimental::ssm::repeat_mul
