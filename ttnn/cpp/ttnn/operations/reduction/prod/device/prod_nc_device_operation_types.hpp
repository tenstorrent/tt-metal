// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ProdNcParams {
    int64_t dim;
};

struct ProdNcInputs {
    Tensor input;
    Tensor output;  // Note: output is passed as input (inplace pattern)
};

}  // namespace ttnn::prim
