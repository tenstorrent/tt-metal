// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct OffsetCumsumParams {
    // Shape is derived from the input tensor; no extra params needed.
};

}  // namespace ttnn::experimental::prim
