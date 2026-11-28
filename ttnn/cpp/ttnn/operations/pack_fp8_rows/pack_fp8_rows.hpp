// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// e4m3 [.., M, 128] FP8_E4M3 ROW_MAJOR + scale [.., M, 1] FLOAT32 ROW_MAJOR (both DRAM interleaved, equal leading
// dims) -> [.., M, 192] UINT8 ROW_MAJOR rows [values | fp32 scale | 60 zero B] (all zero where the values are all
// +-0). output_tensor: the caller's buffer of exactly that spec (DRAM interleaved), returned as is. max_grid: the
// cores are taken from the rectangle (0, 0) .. (max_grid - 1). Refusals throw PackFp8RowsAssertion.
Tensor pack_fp8_rows(
    const Tensor& e4m3,
    const Tensor& scale,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::CoreCoord>& max_grid = std::nullopt);

}  // namespace ttnn
