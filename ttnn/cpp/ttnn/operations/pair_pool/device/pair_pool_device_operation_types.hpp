// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::pair_pool {

// Everything the public entry resolved; all structural, so the default program hash (every attribute plus the
// tensor specs) is the key. The Python key was (mesh, x padded shape, memory configs, max_grid): the program
// cache is per mesh, the specs carry shape and memory configs, and grid_x / grid_y are max_grid resolved
// against the device grid (max(1, min(g, max_grid))).
struct PairPoolParams {
    uint32_t grid_x = 1;  // grid width the op uses (PAIR_POOL_GX)
    uint32_t grid_y = 1;
    tt::tt_metal::MemoryConfig output_memory_config;  // memory_config or DRAM, or the output_tensor's
};

struct PairPoolInputs {
    const Tensor& input;
    const std::optional<Tensor>& output;  // caller's output_tensor, returned as is
};

// The Python op refused with `assert` (AssertionError, messages the tests match). The binding's translator raises
// this type as Python's builtin AssertionError with the same message.
struct PairPoolAssertion : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::pair_pool
