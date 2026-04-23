// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::experimental::prim {

struct SpEqMulMaskParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
};

struct SpEqMulMaskInputs {
    Tensor a;
    Tensor b;
};

}  // namespace ttnn::experimental::prim
