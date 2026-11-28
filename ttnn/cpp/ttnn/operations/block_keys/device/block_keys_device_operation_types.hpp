// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::block_keys {

// What the public entry resolved. Every field is structural: compute_program_hash covers them all plus the pos / ids
// specs (K, KP and the row count), leaving out only whether the output is the caller's. Only the three buffer
// addresses vary between hits; the factory's override_runtime_arguments writes them.
struct BlockKeysParams {
    uint32_t block = 1;   // B, a power of two <= 32
    uint32_t nbw = 0;     // width / B (Python: int(width) // B)
    uint32_t grid_x = 1;  // worker grid clamped by max_grid (Python _grid): kernel GX, core count gx * gy
    uint32_t grid_y = 1;
    tt::tt_metal::MemoryConfig output_memory_config;  // memory_config or DRAM_MEMORY_CONFIG
};

struct BlockKeysInputs {
    const Tensor& pos;
    const Tensor& ids;
    const std::optional<Tensor>& output;  // preallocated output (returned as is)
};

// The Python op refuses only through asserts (AssertionError); the port refuses with TT_FATAL and declares no
// _op_contract refusal types.

}  // namespace ttnn::operations::block_keys
