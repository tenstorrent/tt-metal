// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice {

struct NlpKvCacheLoadSliceParams {
    ttnn::Shape output_tensor_start;
    ttnn::Shape output_tensor_end;
};

struct NlpKvCacheLoadSliceInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice
