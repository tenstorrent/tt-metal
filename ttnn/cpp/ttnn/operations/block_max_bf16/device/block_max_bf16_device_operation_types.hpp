// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::block_max_bf16 {

// What the public entry resolved. Key: everything but `n`, a per-call value that the factories'
// override_runtime_arguments writes into common_arg::N (compute_program_hash leaves it out). Whether n_tensor is
// given is in the key through tensor_args_t (its presence and spec).
struct BlockMaxBf16Params {
    bool strip_major = false;
    uint32_t factor = 1;  // reduce: a power of two <= 16; strip_major: 1
    uint32_t n = 0;       // output columns / blocks (0 when n_tensor is given); NOT in the key
    // The caller's grid cap (x, y), clamped to the device grid by the factory; nullopt: the whole grid.
    std::optional<std::array<uint32_t, 2>> max_grid;
    tt::tt_metal::MemoryConfig output_memory_config;  // memory_config or DRAM interleaved (unused with output)
};

struct BlockMaxBf16Inputs {
    const Tensor& input;                    // block_max
    const std::optional<Tensor>& n_tensor;  // [1] uint32 ROW_MAJOR DRAM
    const std::optional<Tensor>& output;    // preallocated output (required with strip_major)
};

// No refusal types: the Python op refuses only with assertions, which the port raises as TT_FATAL.

}  // namespace ttnn::operations::block_max_bf16
