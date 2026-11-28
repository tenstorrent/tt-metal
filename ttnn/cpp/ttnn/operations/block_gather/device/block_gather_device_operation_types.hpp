// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::block_gather {

// What the public entry resolved. The program key (compute_program_hash) hashes all of these plus the scores and ids
// specs, which fix K, W, rows, parts and the scratch size: every attribute is structural. output_memory_config is
// replaced in the key by the caller's output's memory config when one is given (the call ignores it then).
//
// The Python refuses with plain asserts (AssertionError) and has no _op_contract refusals, so the port declares no
// refusal types: every refusal is a TT_FATAL.
struct BlockGatherParams {
    uint32_t block = 1;                               // B
    tt::tt_metal::MemoryConfig output_memory_config;  // memory_config or DRAM interleaved
    // The worker rectangle: device grid clamped by max_grid (each side max(1, min(grid, max_grid))), as _grid().
    uint32_t grid_x = 1;
    uint32_t grid_y = 1;
};

struct BlockGatherInputs {
    const Tensor& scores;                 // [.., R, W] bf16 ROW_MAJOR, DRAM interleaved
    const Tensor& ids;                    // [.., R, K] uint32 ROW_MAJOR, DRAM interleaved
    const std::optional<Tensor>& output;  // preallocated [.., R, K B] bf16 ROW_MAJOR, returned as is
};

}  // namespace ttnn::operations::block_gather
