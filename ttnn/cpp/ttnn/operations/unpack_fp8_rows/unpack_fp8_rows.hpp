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

// The inverse of pack_fp8_rows onto the bfloat8_b grid: rows [row_lo, row_hi) of batch `batch` of `rows`
// ([B, .., R, 192] uint8 ROW_MAJOR in DRAM, interleaved or ND-sharded) decoded as e4m3 x scale, rounded to nearest
// even onto bfloat8_b, written in place into `out` ([.., M, 128] bfloat8_b TILE, DRAM interleaved) at rows
// [out_row0, out_row0 + row_hi - row_lo) (default out_row0 = row_lo). Row bounds are whole tile rows. max_grid: the
// cores of the rectangle (0, 0) .. max_grid - 1. whole_grid: every call on the whole grid (one program for every row
// count). common_values: the per-call values as common runtime args. Returns `out`. Refusals throw
// UnpackFp8RowsAssertion.
Tensor unpack_fp8_rows(
    const Tensor& rows,
    const Tensor& out,
    int64_t row_lo,
    int64_t row_hi,
    int64_t batch = 0,
    const std::optional<int64_t>& out_row0 = std::nullopt,
    const std::optional<tt::tt_metal::CoreCoord>& max_grid = std::nullopt,
    bool whole_grid = false,
    bool common_values = false);

}  // namespace ttnn
