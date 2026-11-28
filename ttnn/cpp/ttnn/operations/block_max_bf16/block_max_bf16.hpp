// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// strip_major == false: block_max [.., R, NI] fp32 ROW_MAJOR DRAM -> [.., R, NI / factor] bf16 ROW_MAJOR, column b the
// max of input columns [b factor, (b + 1) factor), for the first n output columns (rounded up to 8).
// strip_major == true: block_max [.., S NB / 8, 256] bf16 ROW_MAJOR -> output_tensor [.., 32 S, NB] bf16 (required),
// a transpose of each strip, the first n blocks (rounded up to 8); factor 1.
// max_grid: a CoreCoord (an (x, y) pair converts) capping the worker grid.
// Exactly one of n (a runtime value) and n_tensor ([1] uint32 ROW_MAJOR DRAM, read on the device).
Tensor block_max_bf16(
    const Tensor& block_max,
    uint32_t factor = 1,
    std::optional<uint32_t> n = std::nullopt,
    const std::optional<Tensor>& n_tensor = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreCoord>& max_grid = std::nullopt,
    bool strip_major = false);

}  // namespace ttnn
