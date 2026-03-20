
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::binary {

enum class BinaryCompositeOpType {
    NEXTAFTER,
    ISCLOSE,
    ATAN2,
    DIV_NO_NAN,
    FLOOR_DIV,
    OUTER,
    POLYVAL,
};

}  // namespace ttnn::operations::binary
