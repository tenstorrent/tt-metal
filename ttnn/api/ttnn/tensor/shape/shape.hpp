// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/shape.hpp>

namespace ttnn {
using tt::tt_metal::Shape;

inline uint32_t get_batch_size(const tt::tt_metal::Shape& shape) {
    uint32_t result = 1;
    for (int i = 0; i < (int)shape.rank() - 2; i++) {
        result *= shape[i];
    }
    return result;
}

}  // namespace ttnn
