// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::ssm::hc_sum_reduce {

struct operation_attributes_t {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
    const MathFidelity math_fidelity;
};

struct tensor_args_t {
    Tensor input;
};

}  // namespace ttnn::operations::experimental::ssm::hc_sum_reduce
