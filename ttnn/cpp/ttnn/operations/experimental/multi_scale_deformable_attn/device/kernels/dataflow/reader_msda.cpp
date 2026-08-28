// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for fused multi-scale deformable attention.
//
// Stages the grid and attn for a block of up to 32 queries as column-0 tiles the
// compute kernel solves the sampling geometry from, then gathers the value
// sticks for the corners it hands back. One output tile carries up to 32 queries
// from a single batch index n.
//
// The geometry itself is not here. It is per-point float work over 32 queries —
// vector work on a core with no vector unit and no FPU, where every operation
// costs about 140 cycles of soft-float emulation. The compute kernel owns it.
// What stays is integer: decoding the floored corner out of bf16, bounds-testing
// it, and turning it into a page index.
//
// Tile face layout (bf16, 32x32 tile = 4 faces of 16x16, 512 B per face,
// 2048 B per tile):
//   TL face: offset    0..511   (rows  0..15, cols  0..15)
//   TR face: offset  512..1023  (rows  0..15, cols 16..31)
//   BL face: offset 1024..1535  (rows 16..31, cols  0..15)
//   BR face: offset 1536..2047  (rows 16..31, cols 16..31)
// Row r spans a low half (cols 0..15) and a high half (cols 16..31) at
// non-contiguous offsets; both are 32-byte aligned, so the NoC writes value
// sticks into them directly.
//
// Per-tile runtime args (3 per tile): (n, q_start, v_rows). 1 <= v_rows <= 32.
// Zero-fill contract:
//   * grid/attn tiles: column 0 is written for all 32 rows, zero past v_rows, so
//     a tail row's weight comes out zero.
//   * input tile: only valid rows (r < v_rows AND corner in bounds) are written;
//     tail and out-of-bounds rows keep whatever the CB slot held. Safe because
//     the matching scalar lane is zero.

#include <cstdint>
#include <cstring>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/kernels/msda_tile_layout.hpp"

namespace {

// Byte-identical to the `bfloat16_to_float` / `float_to_bfloat16` helpers in
// ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/grid_sample_reader_common.hpp
// and a handful of other reader kernels; duplicated here to keep the kernel
// dependency-free.
// TODO(#45742): consolidate these per-op copies into one shared kernel header.
// x0/y0 arrive as bf16 holding exact integers (the compute kernel floored them),
// so they decode with shifts. Going through float here would put soft-float back
// on a core that has no FPU, which is the whole point of moving the geometry.
inline int32_t bf16_to_int(uint16_t bf16) {
    const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
    if (exp < 0) {
        return 0;  // |v| < 1, and v is integral, so v == 0
    }
    const uint32_t mant = (bits & 0x7FFFFFu) | 0x800000u;
    const int32_t v = static_cast<int32_t>(mant >> (23 - exp));
    return (bits & 0x80000000u) ? -v : v;
}

}  // namespace

constexpr uint32_t grid_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t attn_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t input_tile_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t grid_x_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t grid_y_cb_index = get_compile_time_arg_val(4);
constexpr uint32_t attn_tile_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t x0_cb_index = get_compile_time_arg_val(6);
constexpr uint32_t y0_cb_index = get_compile_time_arg_val(7);

constexpr uint32_t D = get_compile_time_arg_val(8);
constexpr uint32_t Q = get_compile_time_arg_val(9);
constexpr uint32_t P = get_compile_time_arg_val(10);
constexpr uint32_t h_in = get_compile_time_arg_val(11);
constexpr uint32_t w_in = get_compile_time_arg_val(12);
constexpr uint32_t value_stick_nbytes = get_compile_time_arg_val(13);
constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(14);
constexpr uint32_t attn_stick_nbytes = get_compile_time_arg_val(15);
// Points sharing one grid page: P when the caller folds the point axis into the
// last dimension, 1 for the (N, Q*P, 1, 2) form. Divides P either way.
constexpr uint32_t grid_pts_per_stick = get_compile_time_arg_val(16);
// Heads packed into the value stick. n indexes (batch, head) jointly, so the batch
// picks the page and the head picks a byte range inside it.
constexpr uint32_t num_heads = get_compile_time_arg_val(17);
// attn addressing. Legacy (N, Q, P): n_div 1, head stride 0, so the page carries the
// head and the offset is zero. Packed (B, Q, num_heads*stride): n_div num_heads, and
// the head plus point_offset pick a P-point range inside the row.
constexpr uint32_t attn_page_nbytes = get_compile_time_arg_val(18);
constexpr uint32_t attn_n_div = get_compile_time_arg_val(19);
constexpr uint32_t attn_head_stride_pts = get_compile_time_arg_val(20);
constexpr uint32_t attn_point_offset = get_compile_time_arg_val(21);
// grid addressing, same three-way shape as attn's: n_div 1 and head stride 0 leave the
// page carrying the head, which is both rank-4 forms.
constexpr uint32_t grid_page_nbytes = get_compile_time_arg_val(22);
constexpr uint32_t grid_n_div = get_compile_time_arg_val(23);
constexpr uint32_t grid_head_stride_pts = get_compile_time_arg_val(24);
constexpr uint32_t grid_point_offset = get_compile_time_arg_val(25);
constexpr auto value_args = TensorAccessorArgs<26>();
constexpr auto grid_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
constexpr auto attn_args = TensorAccessorArgs<grid_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_MAX_ROWS = 32;
constexpr uint32_t HALF_STICK_NBYTES = 32;  // one face-row half: 16 bf16 (TL or TR portion of one row)
constexpr uint32_t HALF_WORDS = HALF_STICK_NBYTES / sizeof(uint32_t);
constexpr uint32_t TILE_NBYTES = 2048;  // bf16 32x32 tile
constexpr uint32_t GRID_STICKS_PER_QUERY = P / grid_pts_per_stick;
// A NoC read's source offset must sit on a 32-byte boundary, and a head's level run
// need not. The read is rounded down to the boundary and the wanted points indexed
// from there, so the scratch row has to hold the worst-case 30-byte lead-in too.
constexpr uint32_t NOC_OFF_ALIGN = 32;
constexpr uint32_t ATTN_READ_NBYTES = ((NOC_OFF_ALIGN - 2) + P * 2 + NOC_OFF_ALIGN - 1) / NOC_OFF_ALIGN * NOC_OFF_ALIGN;

// A value stick carries D bf16 values (= D/2 uint32 words). One tile row
// holds 32 values (lo + hi face halves), so a stick spans N_D_TILES tiles
// laid side by side; the trailing tile is half-filled when D % 32 == 16.
// Derived from D (not value_stick_nbytes, which is alignment-padded).
constexpr uint32_t STICK_WORDS = D / 2;
constexpr uint32_t WORDS_PER_TILE_ROW = 2 * HALF_WORDS;
constexpr uint32_t N_D_TILES = (STICK_WORDS + WORDS_PER_TILE_ROW - 1) / WORDS_PER_TILE_ROW;
constexpr uint32_t STICK_NBYTES = D * 2;  // one head's slice, before alignment padding
constexpr uint32_t HEAD_STRIDE_NBYTES = STICK_NBYTES;
constexpr uint32_t TILE_ROW_NBYTES = 2 * HALF_STICK_NBYTES;  // 32 bf16 = one tile row
static_assert(D % 16 == 0 && D > 0, "D must be a positive multiple of 16");
// x0/y0 cross from the compute kernel as bf16, which is exact for integers up to
// 256. Past that the decoded corner would silently be wrong, not out of bounds.
static_assert(h_in <= 256 && w_in <= 256, "bf16 corner indices are exact only up to 256");

void kernel_main() {
    const uint32_t value_addr = get_arg_val<uint32_t>(0);
    const uint32_t grid_addr = get_arg_val<uint32_t>(1);
    const uint32_t attn_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(3);

    const auto value_acc = TensorAccessor(value_args, value_addr, value_stick_nbytes);
    const auto grid_acc = TensorAccessor(grid_args, grid_addr, grid_page_nbytes);
    const auto attn_acc = TensorAccessor(attn_args, attn_addr, attn_page_nbytes);

    Noc noc;
    CircularBuffer grid_cb(grid_cb_index);
    CircularBuffer attn_cb(attn_cb_index);
    CircularBuffer input_tile_cb(input_tile_cb_index);
    CircularBuffer grid_x_cb(grid_x_cb_index);
    CircularBuffer grid_y_cb(grid_y_cb_index);
    CircularBuffer attn_tile_cb(attn_tile_cb_index);
    CircularBuffer x0_cb(x0_cb_index);
    CircularBuffer y0_cb(y0_cb_index);

    constexpr int32_t h_in_i = static_cast<int32_t>(h_in);
    constexpr int32_t w_in_i = static_cast<int32_t>(w_in);

    // Reserve scratch CBs once and treat them as fixed linear L1 arenas.
    grid_cb.reserve_back(TILE_MAX_ROWS * GRID_STICKS_PER_QUERY);
    const uint32_t grid_scratch_l1 = grid_cb.get_write_ptr();
    attn_cb.reserve_back(TILE_MAX_ROWS);
    const uint32_t attn_scratch_l1 = attn_cb.get_write_ptr();

    // Per-(p, corner) precompute scratch (one entry per row in the current tile).
    int32_t x0_arr[TILE_MAX_ROWS];
    int32_t y0_arr[TILE_MAX_ROWS];
    bool x0v_arr[TILE_MAX_ROWS];
    bool x1v_arr[TILE_MAX_ROWS];
    bool y0v_arr[TILE_MAX_ROWS];
    bool y1v_arr[TILE_MAX_ROWS];

    uint32_t arg_idx = 4;
    for (uint32_t t = 0; t < num_output_tiles; ++t) {
        const uint32_t n = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t q_start = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t v_rows = get_arg_val<uint32_t>(arg_idx++);

        // Stage attn for the v_rows queries (one P-wide run each).
        const uint32_t attn_off = ((n % attn_n_div) * attn_head_stride_pts + attn_point_offset) * 2;
        const uint32_t attn_read_off = attn_off & ~(NOC_OFF_ALIGN - 1);
        const uint32_t attn_lane = (attn_off - attn_read_off) / 2;
        const uint32_t attn_row_base = (n / attn_n_div) * Q + q_start;
        // A run near the end of the row would make the padded read overrun the page; the
        // tail past the wanted points is only there to satisfy the offset alignment.
        const uint32_t attn_avail = attn_page_nbytes - attn_read_off;
        const uint32_t attn_bytes = ATTN_READ_NBYTES < attn_avail ? ATTN_READ_NBYTES : attn_avail;
        for (uint32_t r = 0; r < v_rows; ++r) {
            CoreLocalMem<uint32_t> dst(attn_scratch_l1 + r * ATTN_READ_NBYTES);
            noc.async_read(
                attn_acc, dst, attn_bytes, {.page_id = attn_row_base + r, .offset_bytes = attn_read_off}, {});
        }
        // Stage grid for v_rows queries: GRID_STICKS_PER_QUERY sticks each, holding
        // grid_pts_per_stick points of (x, y). Packed, that is one stick reached by the same
        // rounded-down offset attn uses; unpacked, the head is already in the page index.
        const uint32_t grid_off = ((n % grid_n_div) * grid_head_stride_pts + grid_point_offset) * 4;
        const uint32_t grid_read_off = grid_off & ~(NOC_OFF_ALIGN - 1);
        const uint32_t grid_lane = (grid_off - grid_read_off) / 2;
        const uint32_t grid_avail = grid_page_nbytes - grid_read_off;
        const uint32_t grid_bytes = grid_stick_nbytes < grid_avail ? grid_stick_nbytes : grid_avail;
        for (uint32_t r = 0; r < v_rows; ++r) {
            const uint32_t base = ((n / grid_n_div) * Q + (q_start + r)) * GRID_STICKS_PER_QUERY;
            for (uint32_t g = 0; g < GRID_STICKS_PER_QUERY; ++g) {
                CoreLocalMem<uint32_t> dst(grid_scratch_l1 + (r * GRID_STICKS_PER_QUERY + g) * grid_stick_nbytes);
                noc.async_read(grid_acc, dst, grid_bytes, {.page_id = base + g, .offset_bytes = grid_read_off}, {});
            }
        }
        noc.async_read_barrier();

        const uint32_t n_off = (n / num_heads) * static_cast<uint32_t>(h_in_i * w_in_i);
        const uint32_t head_off = (n % num_heads) * HEAD_STRIDE_NBYTES;

        // Hand the geometry to the compute kernel: one column-0 tile per point for
        // gx, gy and attn. All P points go out before any corner comes back — a
        // reader that interleaved the two would block on a full input_tile_cb while
        // compute waited on the next point's grid.
        for (uint32_t p = 0; p < P; ++p) {
            grid_x_cb.reserve_back(1);
            grid_y_cb.reserve_back(1);
            attn_tile_cb.reserve_back(1);
            const uint32_t gx_l1 = grid_x_cb.get_write_ptr();
            const uint32_t gy_l1 = grid_y_cb.get_write_ptr();
            const uint32_t at_l1 = attn_tile_cb.get_write_ptr();

            // The geometry runs over the whole tile, not just column 0, so the
            // other 31 columns must be finite. A CB slot is uninitialised L1 on
            // its first use and the bit pattern there can decode to inf or NaN,
            // which the SFPU then propagates. Later blocks reuse the slot and
            // find this pass's finite values, so once is enough.
            if (t < P) {
                for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                    const auto row = msda_tile_layout::tile_row_offsets(r);
                    CoreLocalMem<volatile uint32_t> gx_lo(gx_l1 + row.lo);
                    CoreLocalMem<volatile uint32_t> gx_hi(gx_l1 + row.hi);
                    CoreLocalMem<volatile uint32_t> gy_lo(gy_l1 + row.lo);
                    CoreLocalMem<volatile uint32_t> gy_hi(gy_l1 + row.hi);
                    CoreLocalMem<volatile uint32_t> at_lo(at_l1 + row.lo);
                    CoreLocalMem<volatile uint32_t> at_hi(at_l1 + row.hi);
                    for (uint32_t i = 0; i < HALF_WORDS; ++i) {
                        gx_lo[i] = 0;
                        gx_hi[i] = 0;
                        gy_lo[i] = 0;
                        gy_hi[i] = 0;
                        at_lo[i] = 0;
                        at_hi[i] = 0;
                    }
                }
            }

            for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                // Tail rows are written as zero: their weight ends up zero, which
                // is the same contract the reduction already relies on.
                uint16_t gx = 0;
                uint16_t gy = 0;
                uint16_t av = 0;
                if (r < v_rows) {
                    const uint32_t stick = r * GRID_STICKS_PER_QUERY + p / grid_pts_per_stick;
                    const uint32_t lane = grid_lane + 2 * (p % grid_pts_per_stick);
                    CoreLocalMem<volatile uint16_t> grid_ptr(grid_scratch_l1 + stick * grid_stick_nbytes);
                    CoreLocalMem<volatile uint16_t> attn_ptr(attn_scratch_l1 + r * ATTN_READ_NBYTES);
                    gx = grid_ptr[lane];
                    gy = grid_ptr[lane + 1];
                    av = attn_ptr[attn_lane + p];
                }
                const uint32_t col0 = msda_tile_layout::tile_col0_offset(r);
                CoreLocalMem<volatile uint16_t> gx_dst(gx_l1 + col0);
                CoreLocalMem<volatile uint16_t> gy_dst(gy_l1 + col0);
                CoreLocalMem<volatile uint16_t> at_dst(at_l1 + col0);
                gx_dst[0] = gx;
                gy_dst[0] = gy;
                at_dst[0] = av;
            }

            grid_x_cb.push_back(1);
            grid_y_cb.push_back(1);
            attn_tile_cb.push_back(1);
        }

        for (uint32_t p = 0; p < P; ++p) {
            // Corners solved on the SFPU. Only the bounds test and the page index
            // stay here, both in integer arithmetic.
            x0_cb.wait_front(1);
            y0_cb.wait_front(1);
            const uint32_t x0_l1 = x0_cb.get_read_ptr();
            const uint32_t y0_l1 = y0_cb.get_read_ptr();
            for (uint32_t r = 0; r < v_rows; ++r) {
                const uint32_t col0 = msda_tile_layout::tile_col0_offset(r);
                CoreLocalMem<volatile uint16_t> x0_src(x0_l1 + col0);
                CoreLocalMem<volatile uint16_t> y0_src(y0_l1 + col0);
                const int32_t x0 = bf16_to_int(x0_src[0]);
                const int32_t y0 = bf16_to_int(y0_src[0]);
                x0_arr[r] = x0;
                y0_arr[r] = y0;
                x0v_arr[r] = (x0 >= 0) && (x0 < w_in_i);
                x1v_arr[r] = (x0 + 1 >= 0) && (x0 + 1 < w_in_i);
                y0v_arr[r] = (y0 >= 0) && (y0 < h_in_i);
                y1v_arr[r] = (y0 + 1 >= 0) && (y0 + 1 < h_in_i);
            }
            x0_cb.pop_front(1);
            y0_cb.pop_front(1);

            for (uint32_t c = 0; c < 4; ++c) {
                // Hoist all c-invariant selectors out of the per-r loops below:
                // c picks which y/x validity array, which corner-weight array,
                // and the (dy, dx) offset to the corner.
                const int32_t dy_off = (c < 2) ? 0 : 1;
                const int32_t dx_off = (c & 1) ? 1 : 0;
                const bool* yv_arr = (c < 2) ? y0v_arr : y1v_arr;
                const bool* xv_arr = (c & 1) ? x1v_arr : x0v_arr;

                // ---- INPUT TILES (N_D_TILES per (p, corner)) ----
                input_tile_cb.reserve_back(N_D_TILES);
                const uint32_t tile_l1 = input_tile_cb.get_write_ptr();

                // Read each valid row's stick straight into its face-row halves.
                //
                // A tile row is two 32-byte halves at non-contiguous L1 offsets, so a 64-byte
                // stick cannot land in one transfer — but both halves are 32-byte aligned on
                // either side, so the NoC can place them directly. Staging into a scratch stick
                // and copying word by word costs more than the gather it serves.
                //
                // Only the logical D*2 bytes are requested, never value_stick_nbytes: the padded
                // tail would spill past the row into the next one.
                for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                    if (r >= v_rows || !(yv_arr[r] && xv_arr[r])) {
                        // No corner to gather: zero the row. The reduction weights
                        // come from the compute kernel, which does not know which
                        // corners fell outside the feature map, so a stale row here
                        // would be multiplied by a live weight.
                        const auto off = msda_tile_layout::tile_row_offsets(r);
                        for (uint32_t k = 0; k < N_D_TILES; ++k) {
                            const uint32_t ktile_l1 = tile_l1 + k * TILE_NBYTES;
                            CoreLocalMem<volatile uint32_t> zlo(ktile_l1 + off.lo);
                            CoreLocalMem<volatile uint32_t> zhi(ktile_l1 + off.hi);
                            for (uint32_t i = 0; i < HALF_WORDS; ++i) {
                                zlo[i] = 0;
                                zhi[i] = 0;
                            }
                        }
                        continue;
                    }
                    const uint32_t cy = static_cast<uint32_t>(y0_arr[r] + dy_off);
                    const uint32_t cx = static_cast<uint32_t>(x0_arr[r] + dx_off);
                    const uint32_t stick_idx = n_off + cy * w_in_i + cx;
                    const auto off = msda_tile_layout::tile_row_offsets(r);
                    for (uint32_t k = 0; k < N_D_TILES; ++k) {
                        // Length is measured inside this head's slice; head_off only moves
                        // the address, so it must not enter the remaining-bytes maths.
                        const uint32_t tile_off = k * TILE_ROW_NBYTES;
                        const uint32_t src_off = head_off + tile_off;
                        const uint32_t bytes_k =
                            (STICK_NBYTES - tile_off < TILE_ROW_NBYTES) ? (STICK_NBYTES - tile_off) : TILE_ROW_NBYTES;
                        const uint32_t lo_bytes = bytes_k < HALF_STICK_NBYTES ? bytes_k : HALF_STICK_NBYTES;
                        const uint32_t ktile_l1 = tile_l1 + k * TILE_NBYTES;
                        CoreLocalMem<uint32_t> dl(ktile_l1 + off.lo);
                        noc.async_read(value_acc, dl, lo_bytes, {.page_id = stick_idx, .offset_bytes = src_off}, {});
                        if (bytes_k > lo_bytes) {
                            CoreLocalMem<uint32_t> dh(ktile_l1 + off.hi);
                            noc.async_read(
                                value_acc,
                                dh,
                                bytes_k - lo_bytes,
                                {.page_id = stick_idx, .offset_bytes = src_off + HALF_STICK_NBYTES},
                                {});
                        }
                    }
                }
                noc.async_read_barrier();

                // Tail rows (r ≥ v_rows) and OOB-corner rows are left untouched:
                // their scalar entry is zero (see scalar tile below), so any stale
                // bytes in input row r contribute 0 to L1 accumulation. Saves a
                // 16-row × 64-byte memset for tail tiles and skips work on full
                // tiles entirely. The same contract covers the unused hi halves of
                // a trailing half-filled d-tile (D % 32 == 16): the writer never
                // reads those lanes back.
                input_tile_cb.push_back(N_D_TILES);

            }
        }
    }
}
