// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor fused_gate(
    const ttnn::Tensor& a,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& b,
    uint32_t Wt,
    uint32_t Gt,
    uint32_t Ht,
    uint32_t mode);

}  // namespace ttnn::operations::experimental
