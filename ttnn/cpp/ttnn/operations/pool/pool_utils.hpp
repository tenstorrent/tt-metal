// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ttnn/operations/pool/generic/device/pool_op.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"

namespace ttnn::operations::pool {
tt::tt_metal::ReduceOpMath get_reduce_op(Pool2DType pool_type);
uint32_t get_bf16_pool_scalar(Pool2DType pool_type, uint32_t kernel_size_hw);
uint32_t get_bf16_pool_init_value(Pool2DType pool_type);

}  // namespace ttnn::operations::pool
