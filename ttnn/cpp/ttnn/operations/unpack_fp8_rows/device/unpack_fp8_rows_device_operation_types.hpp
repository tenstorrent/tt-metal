// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <stdexcept>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::unpack_fp8_rows {

// What the public entry resolved (fp8_rows.py unpack_fp8_rows). The key (compute_program_hash) is every
// structural field plus the rows and out specs; the per-call values (src_page0, out_page0, nbands) are left
// out of it and rewritten by override_runtime_arguments, with the two buffer addresses.
struct UnpackFp8RowsParams {
    // Per call, not in the key (override_runtime_arguments re-applies them):
    uint32_t src_page0 = 0;  // batch * R + row_lo: the first row's page of rows
    uint32_t out_page0 = 0;  // out_row0 / 32 * (VALUES / 32): the first tile of out
    uint32_t nbands = 0;     // (row_hi - row_lo) / 32, > 0 (an empty range returns before launch)
    // Structural, in the key:
    uint32_t grid_x = 1;         // _grid(): the device grid cut to max_grid, at least 1 (GX)
    uint32_t grid_y = 1;         // the cores are y * grid_x + x < ncores
    uint32_t ncores = 1;         // whole_grid ? grid_x * grid_y : max(1, min(grid_x * grid_y, nbands))
    bool common_values = false;  // the per-call values as common args (define UNPACK_COMMON_VALUES)
};

struct UnpackFp8RowsInputs {
    const Tensor& rows;  // [B, .., R, 192] uint8 ROW_MAJOR, DRAM (interleaved or ND-sharded)
    const Tensor& out;   // [.., M, 128] bfloat8_b TILE, DRAM interleaved: written in place and returned
};

// The Python op refuses with `assert` (AssertionError, the tests match its message). The binding's translator
// raises this type as Python's builtin AssertionError with the same message.
struct UnpackFp8RowsAssertion : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::unpack_fp8_rows
