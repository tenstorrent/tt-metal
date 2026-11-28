// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::pack_fp8_rows {

// What the public entry resolved. Every field is structural: the key (compute_program_hash) is every attribute
// plus the e4m3 and scale specs; the three buffer addresses are rewritten by override_runtime_arguments.
struct PackFp8RowsParams {
    // The output's memory config (memory_config, else DRAM interleaved; the output tensor's when given).
    tt::tt_metal::MemoryConfig output_memory_config;
    // The grid the op uses: the device's compute grid cut to max_grid (each at least 1), as Python _grid().
    uint32_t grid_x = 1;
    uint32_t grid_y = 1;
};

struct PackFp8RowsInputs {
    const Tensor& e4m3;
    const Tensor& scale;
    // Preallocated output (uint8 ROW_MAJOR [.., M, 192], DRAM interleaved), returned as is.
    const std::optional<Tensor>& output;
};

// The Python op refuses with `assert` (AssertionError, the tests match it). The binding's translator raises
// this type as Python's builtin AssertionError with the same message.
struct PackFp8RowsAssertion : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::pack_fp8_rows
