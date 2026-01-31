// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn {
enum class PyDType {
    FLOAT32,
    FLOAT64,
    FLOAT16,
    BFLOAT16,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    BOOL
};
}
