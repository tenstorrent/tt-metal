// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::unary_backward {

constexpr uint8_t DefaultQueueId = 0;
enum class UnaryBackwardOpType {
    UNARY_MUL_BW,
};


}  // namespace ttnn::operations::unary
