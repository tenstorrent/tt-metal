// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::plusone {

struct PlusoneParams {
    const std::optional<CoreRangeSet> sub_core_grids;
    const bool skip_negative_entries;
};

struct PlusoneInputs {
    const Tensor& input;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::plusone
