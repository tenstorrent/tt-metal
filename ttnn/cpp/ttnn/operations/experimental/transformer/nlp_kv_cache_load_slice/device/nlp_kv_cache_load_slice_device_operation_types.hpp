// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct NlpKvCacheLoadSliceParams {
    ttnn::Shape output_tensor_start;
    ttnn::Shape output_tensor_end;
};

struct NlpKvCacheLoadSliceInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
