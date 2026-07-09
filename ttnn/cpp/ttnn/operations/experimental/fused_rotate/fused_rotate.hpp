// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor fused_rotate(
    const ttnn::Tensor& x_flat,
    const ttnn::Tensor& coef_exp,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t W,
    const std::vector<uint32_t>& deg,
    const std::vector<uint32_t>& ks,
    const std::vector<uint32_t>& js);

}  // namespace ttnn::operations::experimental
