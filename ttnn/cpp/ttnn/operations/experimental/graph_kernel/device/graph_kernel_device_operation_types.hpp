// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GraphKernelParams {
    // Opaque graph description. Hashed into the program cache key by the default
    // device-operation hash, so distinct texts never share a cached program.
    const std::string text;
};

struct GraphKernelInputs {
    // Arbitrary number of inputs; inputs[0] defines the output spec.
    const std::vector<Tensor> inputs;
};

}  // namespace ttnn::experimental::prim
