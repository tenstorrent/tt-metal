// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// Rows [src_row0, src_row0 + n) of `src` (bf16 ROW_MAJOR, last dim 512, rows flattened; interleaved or ND-sharded)
// -> bfloat4_b row records [row0, row0 + n) of `rows` ([.., R', 320] uint8 ROW_MAJOR, written in place). n % 32 == 0;
// n == 0 launches nothing. max_grid cuts the device's compute grid to (x, y). output_tensor, when given, must be
// `rows` itself (the op's output is the record table). Refusals throw ContractAssertion (Python AssertionError).
void ckv_bfp4_rows(
    const Tensor& src,
    const Tensor& rows,
    uint32_t src_row0,
    uint32_t n,
    uint32_t row0,
    const std::optional<CoreCoord>& max_grid = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn
