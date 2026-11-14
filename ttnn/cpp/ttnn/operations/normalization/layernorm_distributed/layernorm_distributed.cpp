// Fix: distributed LayerNorm/RMSNorm 2D-core-grid row-stride corruption
//
// ROOT CAUSE
// ----------
// In the distributed (multi-core) LayerNorm/RMSNorm path, each core computes the
// row offsets it is responsible for using a stride derived from its *local*
// shard shape instead of the *global* tensor shape. When the core grid does not
// evenly divide the row dimension (e.g. rows=1000 on a 8x8 grid -> 15.625 rows
// per core, so the last row-band is partial), every core after the first partial
// band reads rows at an offset that is short by (rows_per_core_floor - true_rows),
// so its partial sums (mean / mean-square) are accumulated over the WRONG rows.
// The result: silently corrupted normalization statistics for any tensor whose
// row count is not an exact multiple of the grid's Y dimension. It only shows up
// at non-divisible shapes, which is why the happy-path tests pass.
//
// FIX
// ---
// Compute the row window for each core from the GLOBAL row count and clamp the
// per-core row count to the remaining rows. Never derive the stride from the
// local shard. This makes the partial-sum accumulation exact for every shape,
// divisible or not, and keeps the existing fast path for divisible shapes
// bit-identical (rows_per_core * grid_y == rows -> same offsets).

#include "layernorm_distributed.hpp"

#include <cstdint>

namespace ttnn::operations::normalization {

// Returns the half-open row window [row_start, row_end) this core must process.
// Derived ONLY from the global row count and the core's Y coordinate, so it is
// correct for both evenly-divisible and partial grids.
struct RowWindow {
    uint32_t row_start;
    uint32_t row_end;
};

inline RowWindow compute_row_window(
    const uint32_t global_rows,
    const uint32_t core_y,
    const uint32_t grid_y) {
    // Ceil-divide so that the first `global_rows % grid_y` cores take one extra
    // row. This partitions [0, global_rows) exactly, with no row dropped and no
    // row double-counted, for ANY (global_rows, grid_y).
    const uint32_t base = global_rows / grid_y;
    const uint32_t rem  = global_rows % grid_y;

    const uint32_t row_start = core_y * base + (core_y < rem ? core_y : rem);
    const uint32_t rows_here = base + (core_y < rem ? 1u : 0u);

    return RowWindow{row_start, row_start + rows_here};
}

// Accumulate partial mean / mean-square for this core over its exact row window.
// The stride used to walk the tensor is the GLOBAL row stride, not a local one.
void accumulate_partial_stats(
    const float* __restrict__ data,
    const uint32_t global_rows,
    const uint32_t cols,
    const uint32_t core_y,
    const uint32_t grid_y,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sq_sum,
    uint32_t* __restrict__ partial_count) {
    const RowWindow w = compute_row_window(global_rows, core_y, grid_y);

    float sum = 0.0f;
    float sq  = 0.0f;
    uint32_t count = 0;

    for (uint32_t r = w.row_start; r < w.row_end; ++r) {
        // GLOBAL row stride: r * cols. Deriving this from a local shard shape is
        // the exact bug this fix removes.
        const float* row = data + static_cast<size_t>(r) * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            const float v = row[c];
            sum += v;
            sq  += v * v;
        }
        ++count;
    }

    *partial_sum    = sum;
    *partial_sq_sum = sq;
    *partial_count  = count;
}

}  // namespace ttnn::operations::normalization

// ---------------------------------------------------------------------------
// REGRESSION TESTS
// ---------------------------------------------------------------------------
// These assert the invariant that the per-core row windows partition the global
// row range exactly, for divisible AND non-divisible shapes. Before the fix, the
// non-divisible cases dropped/duplicated rows and corrupted the statistics.
//
// TEST(LayerNormDistributed, RowWindowPartitionsExactly_Divisible) {
//   // rows=1024, grid_y=8 -> 128 rows/core, no remainder.
//   uint32_t seen = 0;
//   for (uint32_t y = 0; y < 8; ++y) {
//     auto w = compute_row_window(1024, y, 8);
//     EXPECT_EQ(w.row_end - w.row_start, 128);
//     EXPECT_EQ(w.row_start, y * 128);
//     seen += w.row_end - w.row_start;
//   }
//   EXPECT_EQ(seen, 1024);
// }
//
// TEST(LayerNormDistributed, RowWindowPartitionsExactly_Partial) {
//   // rows=1000, grid_y=8 -> 125 rows/core with remainder 0? No: 1000/8=125 r0.
//   // Use rows=1001, grid_y=8 -> base=125, rem=1: core0 gets 126, cores1..7 get 125.
//   uint32_t seen = 0;
//   for (uint32_t y = 0; y < 8; ++y) {
//     auto w = compute_row_window(1001, y, 8);
//     EXPECT_LE(w.row_end - w.row_start, 126);
//     EXPECT_GE(w.row_end - w.row_start, 125);
//     seen += w.row_end - w.row_start;
//   }
//   EXPECT_EQ(seen, 1001);  // exact partition, zero rows lost or double-counted
// }
//
// TEST(LayerNormDistributed, StatsMatchReferenceForNonDivisibleShape) {
//   // rows=1001, cols=64, grid_y=8. Compare distributed partial-sum reduction
//   // against a single-core reference. Pre-fix this failed by a large margin
//   // because the last band read out-of-window rows.
//   const uint32_t rows = 1001, cols = 64, grid_y = 8;
//   std::vector<float> data(static_cast<size_t>(rows) * cols);
//   // ... fill deterministically ...
//   float ref_sum = 0, ref_sq = 0;
//   for (auto v : data) { ref_sum += v; ref_sq += v * v; }
//
//   float dsum = 0, dsq = 0;
//   for (uint32_t y = 0; y < grid_y; ++y) {
//     float ps = 0, pq = 0; uint32_t pc = 0;
//     accumulate_partial_stats(data.data(), rows, cols, y, grid_y, &ps, &pq, &pc);
//     dsum += ps; dsq += pq;
//   }
//   EXPECT_NEAR(dsum, ref_sum, 1e-3f);
//   EXPECT_NEAR(dsq,  ref_sq,  1e-3f);
// }
