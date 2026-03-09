// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RmScaledAddParams {
    const float scale;
};

struct RmScaledAddInputs {
    const Tensor input_a;
    const Tensor input_b;
};

}  // namespace ttnn::experimental::prim
