// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

// Fold a row-major logical shape into (M, H): M = product of all leading dims, H = the last dim.
// Shared by the plain and masked program factories (both compiled into the same op library, so a
// per-file anonymous-namespace copy would collide under the Unity build).
inline std::pair<uint32_t, uint32_t> fold_M_H(const ttnn::Shape& shape) {
    uint64_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        M *= static_cast<uint64_t>(shape[i]);
    }
    return {static_cast<uint32_t>(M), static_cast<uint32_t>(shape[shape.size() - 1])};
}

}  // namespace ttnn::experimental::prim::per_token_cast_back
