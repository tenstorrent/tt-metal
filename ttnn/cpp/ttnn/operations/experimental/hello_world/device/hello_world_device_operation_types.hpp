// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// hello_world has no op-specific attributes: the program is fully determined by
// the input tensor's spec (shape/dtype/layout/memory config), which the default
// program hash (type + attributes + tensor args) already covers.
struct HelloWorldParams {};

// tensor_args must stay a plain reflectable aggregate: the device-operation
// framework walks it structurally to discover the Tensor leaves.
struct HelloWorldInputs {
    const Tensor& input;
};

}  // namespace ttnn::experimental::prim
