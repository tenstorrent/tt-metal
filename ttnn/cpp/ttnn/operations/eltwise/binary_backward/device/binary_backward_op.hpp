// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::binary_backward {

enum class BinaryBackwardOpType {
    ATAN2_BW,
    EMBEDDING_BW
};


}  // namespace ttnn::operations::binary
