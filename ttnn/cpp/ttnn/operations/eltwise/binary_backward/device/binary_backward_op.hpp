// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::binary_backward {

constexpr uint8_t DefaultQueueId = 0;
enum class BinaryBackwardOpType {
    ATAN2_BW,
    EMBEDDING_BW,
    ADDALPHA_BW,
    SUBALPHA_BW,
    SUB_BW,
    XLOGY_BW,
    HYPOT_BW,
    LDEXP_BW,
    LOGADDEXP_BW,
    LOGADDEXP2_BW,
    SQUARED_DIFFERENCE_BW,
    ADD_BW,
    BINARY_EQ_BW,
    BINARY_ASSIGN_BW,
    CONCAT_BW,
    BINARY_LE_BW,
    RSUB_BW,
    BIAS_GELU_BW,
    BINARY_GT_BW,
    BINARY_NE_BW,
    BINARY_GE_BW,
    MIN_BW,
    MAX_BW,
    DIV_BW,
};


}  // namespace ttnn::operations::binary
