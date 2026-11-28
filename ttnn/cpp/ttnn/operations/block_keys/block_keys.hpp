// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// pos [.., R, KP] uint32 ROW_MAJOR (positions in block_gather's [.., R, K block] row) + the block ids [.., R, K]
// uint32 ROW_MAJOR it read -> [.., R, KP] uint32 ROW_MAJOR key ids ids[pos / block] block + pos % block
// (0xFFFFFFFF where pos is none or its block was: an id >= width / block). `width`: the scores' W.
// Every tensor DRAM interleaved; the output defaults to DRAM. max_grid caps the worker grid.
Tensor block_keys(
    const Tensor& pos,
    const Tensor& ids,
    uint32_t block,
    uint32_t width,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreCoord>& max_grid = std::nullopt);

}  // namespace ttnn
