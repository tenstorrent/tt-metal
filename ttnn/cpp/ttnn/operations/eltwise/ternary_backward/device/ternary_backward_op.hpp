// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::ternary_backward {

enum class TernaryBackwardOpType {
    ADDCMUL_BW,
    ADDCDIV_BW,
    WHERE_BW
};


}  // namespace ttnn::operations::ternary
