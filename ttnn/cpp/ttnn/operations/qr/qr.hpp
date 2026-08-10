// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

#include <optional>
#include <tuple>

namespace ttnn {

// Reduced QR factorization of a rank-2 fp32 matrix with both dimensions at
// most 32 (a single TILE). Returns (Q, R) with Q (m x k) and R (k x n),
// k = min(m, n), following the LAPACK sign convention (sign(0) = 1).
std::tuple<Tensor, Tensor> qr(
    const Tensor& input, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
