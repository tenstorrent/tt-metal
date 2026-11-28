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

// scores [.., R, W] bf16 ROW_MAJOR + block ids [.., R, K] uint32 ROW_MAJOR (equal leading dims, DRAM interleaved)
// -> [.., R, K block] bf16 ROW_MAJOR: row r holds scores[r, ids[r, j] block + o] at j block + o (an id >= W / block:
// -inf). `block` a power of two <= 32 dividing W (a multiple of 32); signed, as the Python int it was, so a negative
// block is refused like any other. `max_grid`: the cores' rectangle.
Tensor block_gather(
    const Tensor& scores,
    const Tensor& ids,
    int64_t block,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreCoord>& max_grid = std::nullopt);

}  // namespace ttnn
