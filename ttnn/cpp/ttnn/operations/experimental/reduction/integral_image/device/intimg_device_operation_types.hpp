// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

struct ReductionParams {};

struct ReductionInputs {
    const Tensor& input_tensor;
};

}  // namespace ttnn::operations::experimental::reduction
