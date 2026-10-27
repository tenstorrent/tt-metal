// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::binary_backward {

// The enum is the migration list for ops routed through the shared binary_backward
// device operation, and the discriminator for the program cache key.
enum class BinaryBackwardOpType : uint8_t {
    MUL_BW,
};

}  // namespace ttnn::operations::binary_backward
