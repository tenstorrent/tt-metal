// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::operations::experimental::isin {

struct operation_attributes_t {
    const bool assume_unique;
    const bool invert;
    const uint32_t single_fetch_subchunk_size;
};

struct tensor_args_t {
    const Tensor elements_tensor;
    const Tensor test_elements_tensor;
    const std::optional<Tensor> optional_out;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::isin
