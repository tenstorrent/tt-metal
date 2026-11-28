// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unpack_fp8_rows.hpp"

#include <algorithm>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>

#include "device/kernels/unpack_fp8_rows_args.hpp"
#include "device/unpack_fp8_rows_device_operation.hpp"

namespace ttnn {

Tensor unpack_fp8_rows(
    const Tensor& rows,
    const Tensor& out,
    int64_t row_lo,
    int64_t row_hi,
    int64_t batch,
    const std::optional<int64_t>& out_row0,
    const std::optional<tt::tt_metal::CoreCoord>& max_grid,
    bool whole_grid,
    bool common_values) {
    using operations::unpack_fp8_rows::UnpackFp8RowsAssertion;
    using operations::unpack_fp8_rows::UnpackFp8RowsParams;
    constexpr int64_t tile_rows = tt::constants::TILE_HEIGHT;

    // The Python asserts, in their order: dtypes / layouts / widths and placement first.
    operations::unpack_fp8_rows::check_support({.rows = rows, .out = out});

    const auto& rs = rows.logical_shape();
    // (Python's rows.shape[-2] fails on a rank-1 tensor.)
    TT_FATAL(rs.rank() >= 2, "unpack_fp8_rows: rows must have rank >= 2, got {}", rs);
    const int64_t r = rs[-2];
    int64_t lead = 1;
    for (size_t i = 0; i + 2 < rs.rank(); ++i) {
        lead *= rs[i];
    }
    const auto& os = out.logical_shape();
    int64_t m = 1;
    for (size_t i = 0; i + 1 < os.rank(); ++i) {
        m *= os[i];
    }
    // Python ints: the range checks see negative values as the Python did (refused as "outside").
    const int64_t first_out_row = out_row0.has_value() ? *out_row0 : row_lo;
    if (!(0 <= batch && batch < lead && 0 <= row_lo && row_lo <= row_hi && row_hi <= r)) {
        throw UnpackFp8RowsAssertion(fmt::format(
            "unpack_fp8_rows: rows [{}, {}) of batch {} outside [{}, .., {}]", row_lo, row_hi, batch, lead, r));
    }
    const int64_t n = row_hi - row_lo;
    if (!(row_lo % tile_rows == 0 && n % tile_rows == 0 && first_out_row % tile_rows == 0)) {
        throw UnpackFp8RowsAssertion(fmt::format(
            "unpack_fp8_rows: whole tile rows only (row_lo {}, row_hi {}, out_row0 {})",
            row_lo,
            row_hi,
            first_out_row));
    }
    if (first_out_row + n > m) {
        throw UnpackFp8RowsAssertion(
            fmt::format("unpack_fp8_rows: out rows [{}, {}) past its {} rows", first_out_row, first_out_row + n, m));
    }
    if (n == 0) {
        return out;
    }
    // (The Python passed a negative first out page on to its runtime-arg patch, which cannot hold it.)
    TT_FATAL(first_out_row >= 0, "unpack_fp8_rows: out_row0 must be >= 0, got {}", first_out_row);
    const auto nbands = static_cast<uint32_t>(n / tile_rows);

    // Python _grid(): the device's compute grid, cut to max_grid (each at least 1).
    auto* device = rows.device();
    TT_FATAL(device != nullptr, "unpack_fp8_rows: rows must be an allocated device tensor");
    const CoreCoord g = device->compute_with_storage_grid_size();
    uint32_t gx = g.x;
    uint32_t gy = g.y;
    if (max_grid.has_value()) {
        gx = std::max<uint32_t>(1, std::min<uint32_t>(g.x, max_grid->x));
        gy = std::max<uint32_t>(1, std::min<uint32_t>(g.y, max_grid->y));
    }
    const uint32_t grid_cores = gx * gy;

    const UnpackFp8RowsParams params{
        .src_page0 = static_cast<uint32_t>((batch * r) + row_lo),
        .out_page0 =
            static_cast<uint32_t>(first_out_row / tile_rows) * (::unpack_fp8_rows::VALUES / tt::constants::TILE_WIDTH),
        .nbands = nbands,
        .grid_x = gx,
        .grid_y = gy,
        .ncores = whole_grid ? grid_cores : std::max<uint32_t>(1, std::min(grid_cores, nbands)),
        .common_values = common_values,
    };
    return ttnn::prim::unpack_fp8_rows(rows, out, params);
}

}  // namespace ttnn
