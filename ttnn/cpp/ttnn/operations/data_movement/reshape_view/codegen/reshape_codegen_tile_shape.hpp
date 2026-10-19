// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Physical (C, H, W) model of a TILE shape: leading dims fold into C.
struct Shape3 {
    uint64_t c = 1;
    uint64_t h = 1;
    uint64_t w = 1;
};

inline std::optional<Shape3> normalize(const ttnn::Shape& shape) {
    if (shape.rank() == 0) {
        return std::nullopt;
    }
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (shape[i] == 0) {
            return std::nullopt;
        }
    }
    Shape3 out;
    out.w = shape[-1];
    if (shape.rank() >= 2) {
        out.h = shape[-2];
    }
    for (size_t i = 0; i + 2 < shape.rank(); ++i) {
        out.c *= shape[i];
    }
    return out;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
