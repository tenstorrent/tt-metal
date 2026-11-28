// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ckv_bfp4_rows {

// What the public entry resolved. src_row0 / num_bands / row0 vary per call and stay out of the cache key
// (compute_program_hash); override_runtime_arguments re-applies them on every hit. grid_x / grid_y (the device grid
// cut to max_grid, every core of it runs the kernels) are structural and in the key.
struct CkvBfp4RowsParams {
    uint32_t src_row0 = 0;
    uint32_t num_bands = 0;  // n / 32, > 0 (the entry returns before launching for n == 0)
    uint32_t row0 = 0;
    uint32_t grid_x = 1;
    uint32_t grid_y = 1;
};

struct CkvBfp4RowsInputs {
    const Tensor& src;   // bf16 ROW_MAJOR, last dim 512, interleaved or ND-sharded
    const Tensor& rows;  // uint8 ROW_MAJOR, last dim 320: the record table written in place
    // Preallocated output: the op's output is the record table itself, so a given output_tensor must be `rows`
    // (same spec and same buffer); it may never alias src.
    const std::optional<Tensor>& output;
};

// The Python op refuses with `assert` (AssertionError, matched by message in the tests). Thrown where the Python
// asserts, with the Python's message; the binding's translator raises it as Python's AssertionError.
struct ContractAssertion : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::ckv_bfp4_rows
