// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor fused_ln_bw(
    const ttnn::Tensor& gy,
    const ttnn::Tensor& x,
    const ttnn::Tensor& red,
    const ttnn::Tensor& n,
    const ttnn::Tensor& gamma,
    uint32_t W,
    uint32_t eps_bits);

}  // namespace ttnn::operations::experimental
