// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::prod_nc {

struct ProdNcParams {
    int64_t dim;
};

struct ProdNcInputs {
    Tensor input;
    Tensor output;  // Note: output is passed as input (inplace pattern)
};

}  // namespace ttnn::operations::reduction::prod_nc
