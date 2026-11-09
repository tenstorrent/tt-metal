// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct KvCacheLoadSliceParams {
    ttnn::Shape output_tensor_start;
    ttnn::Shape output_tensor_end;
};

struct KvCacheLoadSliceInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
