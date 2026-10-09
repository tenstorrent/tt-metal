// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the modulated deformable im2col.
//
// Per block of 32 output rows (pixels) it stages the rows' offsets and masks and
// hands compute one tile each of the y and x offsets, the mask, and the kernel
// window's base coordinates, laid out [row, kernel position]. Compute returns
// every sample's top-left corner (y0, x0) and its four masked bilinear weights
// in the same layout. For each kernel position the reader then hands compute
// the four corners as one row-major block, 32 rows of
// [corner 0 | corner 1 | corner 2 | corner 3] C-wide input sticks, landed by the
// NoC reads themselves, plus one scalar tile per corner whose column 0 holds the
// row's weight. Compute tilizes the block and sums the four corners times their
// scalars, which is the masked bilinear sample.
//
// The geometry runs on compute because the data-movement RISC-Vs have no FPU.
// The reader only moves bits: a bfloat16 widens to fp32 by a shift. Offsets
// saturate at +-OFFSET_CLAMP pixels, which keeps every coordinate inside int32
// and puts a non-finite offset outside the input. The next block's geometry is
// handed over before this block's corners are read, so compute works on it while
// the reader waits on the NoC.
//
// A corner is read only when it lies inside the input and its weight is
// nonzero. Every other row, and every row past the block's end, is zeroed
// rather than skipped: a stale NaN or Inf times a zero scalar would still be
// NaN.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t offset_arena_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t mask_arena_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t corner_block_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t scalar_tile_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t C = get_compile_time_arg_val(4);
constexpr int32_t IN_H = get_compile_time_arg_val(5);
constexpr int32_t IN_W = get_compile_time_arg_val(6);
constexpr uint32_t OUT_H = get_compile_time_arg_val(7);
constexpr uint32_t OUT_W = get_compile_time_arg_val(8);
constexpr uint32_t KERNEL_W = get_compile_time_arg_val(9);
constexpr uint32_t kernel_positions = get_compile_time_arg_val(10);
constexpr int32_t STRIDE_H = get_compile_time_arg_val(11);
constexpr int32_t STRIDE_W = get_compile_time_arg_val(12);
constexpr int32_t PAD_H = get_compile_time_arg_val(13);
constexpr int32_t PAD_W = get_compile_time_arg_val(14);
constexpr int32_t DILATION_H = get_compile_time_arg_val(15);
constexpr int32_t DILATION_W = get_compile_time_arg_val(16);
// Page strides of the three row-major tensors, as their buffers lay them out.
constexpr uint32_t input_page_nbytes = get_compile_time_arg_val(17);
constexpr uint32_t offset_page_nbytes = get_compile_time_arg_val(18);
constexpr uint32_t mask_page_nbytes = get_compile_time_arg_val(19);
// The host validates that the input size, padding and kernel span stay below this.
constexpr uint32_t OFFSET_CLAMP = get_compile_time_arg_val(20);
constexpr uint32_t geom_in_cb_index = get_compile_time_arg_val(21);
constexpr uint32_t geom_index_cb_index = get_compile_time_arg_val(22);
constexpr uint32_t geom_weight_cb_index = get_compile_time_arg_val(23);

constexpr auto input_args = TensorAccessorArgs<24>();
constexpr auto offset_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto mask_args = TensorAccessorArgs<offset_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t CHANNEL_TILES = C / 32;
constexpr uint32_t CORNERS = 4;
constexpr uint32_t STICK_NBYTES = C * sizeof(uint16_t);  // the C channels read per corner
constexpr uint32_t BLOCK_ROW_NBYTES = CORNERS * STICK_NBYTES;

// Tiles of geom_in_cb, in compute's order: fp32 [row, kernel position].
constexpr uint32_t IN_DY = 0;
constexpr uint32_t IN_DX = 1;
constexpr uint32_t IN_MASK = 2;
constexpr uint32_t IN_BASE_Y = 3;
constexpr uint32_t IN_BASE_X = 4;
constexpr uint32_t GEOM_IN_TILES = 5;
constexpr uint32_t FP32_TILE_NBYTES = 32 * 32 * sizeof(uint32_t);
constexpr uint32_t BF16_TILE_NBYTES = 32 * 32 * sizeof(uint16_t);

static_assert(kernel_positions <= 32, "the geometry tiles hold one kernel position per column");
static_assert((OFFSET_CLAMP & (OFFSET_CLAMP - 1)) == 0, "the offset clamp works on the bfloat16 exponent");

namespace {

// bfloat16 exponent field of OFFSET_CLAMP, a power of two, and OFFSET_CLAMP itself.
constexpr uint32_t CLAMP_EXPONENT = 127 + (31 - __builtin_clz(OFFSET_CLAMP));
constexpr uint16_t CLAMP_BF16 = static_cast<uint16_t>(CLAMP_EXPONENT << 7);

// An offset of magnitude OFFSET_CLAMP or more, inf and NaN, become +-OFFSET_CLAMP.
inline uint32_t offset_to_fp32(uint16_t bf16) {
    if (((bf16 >> 7) & 0xFFu) >= CLAMP_EXPONENT) {
        const bool negative = (bf16 & 0x8000u) && (bf16 & 0x7FFFu) <= 0x7F80u;  // NaN goes positive
        return static_cast<uint32_t>(negative ? (0x8000u | CLAMP_BF16) : CLAMP_BF16) << 16;
    }
    return static_cast<uint32_t>(bf16) << 16;
}

// An integer of magnitude below 2^24 as fp32 bits.
inline uint32_t int_to_fp32(int32_t value) {
    if (value == 0) {
        return 0;
    }
    const uint32_t sign = value < 0 ? 0x80000000u : 0;
    const uint32_t magnitude = static_cast<uint32_t>(value < 0 ? -value : value);
    const int32_t msb = 31 - __builtin_clz(magnitude);
    return sign | (static_cast<uint32_t>(127 + msb) << 23) | ((magnitude << (23 - msb)) & 0x7FFFFFu);
}

// Element (row, col) of a 32 x 32 tile in its four 16 x 16 faces.
inline uint32_t tile_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
}

inline void zero_stick(uint32_t l1_addr) {
    CoreLocalMem<volatile uint32_t> dst(l1_addr);
    for (uint32_t i = 0; i < STICK_NBYTES / sizeof(uint32_t); ++i) {
        dst[i] = 0;
    }
}

inline bool inside(int32_t index, int32_t extent) { return index >= 0 && index < extent; }

inline uint32_t valid_rows_of(uint32_t block, uint32_t num_rows) {
    const uint32_t row0 = block * TILE_ROWS;
    return num_rows - row0 < TILE_ROWS ? num_rows - row0 : TILE_ROWS;
}

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t offset_addr = get_arg_val<uint32_t>(1);
    const uint32_t mask_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_block = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);

    const auto input_acc = TensorAccessor(input_args, input_addr, input_page_nbytes);
    const auto offset_acc = TensorAccessor(offset_args, offset_addr, offset_page_nbytes);
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, mask_page_nbytes);

    Noc noc;
    CircularBuffer offset_arena_cb(offset_arena_cb_index);
    CircularBuffer mask_arena_cb(mask_arena_cb_index);
    CircularBuffer corner_block_cb(corner_block_cb_index);
    CircularBuffer scalar_tile_cb(scalar_tile_cb_index);
    CircularBuffer geom_in_cb(geom_in_cb_index);
    CircularBuffer index_cb(geom_index_cb_index);
    CircularBuffer weight_cb(geom_weight_cb_index);

    // The arenas are reserved once and used as flat L1; nothing waits on them.
    offset_arena_cb.reserve_back(TILE_ROWS);
    const uint32_t offset_l1 = offset_arena_cb.get_write_ptr();
    mask_arena_cb.reserve_back(TILE_ROWS);
    const uint32_t mask_l1 = mask_arena_cb.get_write_ptr();

    // b * IN_H per row, the row's image in batch-wide input rows: of the block being read, and of
    // the block whose geometry is being staged.
    int32_t batch_row[TILE_ROWS];
    int32_t staged_batch_row[TILE_ROWS];

    // Stages a block's offsets and masks and hands compute its geometry tiles.
    auto stage_geometry = [&](uint32_t block) {
        const uint32_t row0 = block * TILE_ROWS;
        const uint32_t valid_rows = valid_rows_of(block, num_rows);
        for (uint32_t r = 0; r < valid_rows; ++r) {
            CoreLocalMem<uint32_t> off_dst(offset_l1 + r * offset_page_nbytes);
            noc.async_read(offset_acc, off_dst, offset_page_nbytes, {.page_id = row0 + r}, {.offset_bytes = 0});
            CoreLocalMem<uint32_t> mask_dst(mask_l1 + r * mask_page_nbytes);
            noc.async_read(mask_acc, mask_dst, mask_page_nbytes, {.page_id = row0 + r}, {.offset_bytes = 0});
        }
        geom_in_cb.reserve_back(GEOM_IN_TILES);
        const uint32_t geom_l1 = geom_in_cb.get_write_ptr();
        noc.async_read_barrier();

        // (b, h_out, w_out) of the block's first row, then stepped row by row.
        uint32_t b = row0 / (OUT_H * OUT_W);
        uint32_t h_o = (row0 / OUT_W) % OUT_H;
        uint32_t w_o = row0 % OUT_W;
        for (uint32_t r = 0; r < valid_rows; ++r) {
            staged_batch_row[r] = static_cast<int32_t>(b) * IN_H;
            const int32_t base_y = static_cast<int32_t>(h_o) * STRIDE_H - PAD_H;
            const int32_t base_x = static_cast<int32_t>(w_o) * STRIDE_W - PAD_W;
            CoreLocalMem<volatile uint16_t> off(offset_l1 + r * offset_page_nbytes);
            CoreLocalMem<volatile uint16_t> msk(mask_l1 + r * mask_page_nbytes);
            for (uint32_t kk = 0; kk < kernel_positions; ++kk) {
                const uint32_t element = tile_element(r, kk) * sizeof(uint32_t);
                const int32_t k_y = static_cast<int32_t>(kk / KERNEL_W) * DILATION_H;
                const int32_t k_x = static_cast<int32_t>(kk % KERNEL_W) * DILATION_W;
                CoreLocalMem<volatile uint32_t> dy(geom_l1 + IN_DY * FP32_TILE_NBYTES + element);
                CoreLocalMem<volatile uint32_t> dx(geom_l1 + IN_DX * FP32_TILE_NBYTES + element);
                CoreLocalMem<volatile uint32_t> m(geom_l1 + IN_MASK * FP32_TILE_NBYTES + element);
                CoreLocalMem<volatile uint32_t> by(geom_l1 + IN_BASE_Y * FP32_TILE_NBYTES + element);
                CoreLocalMem<volatile uint32_t> bx(geom_l1 + IN_BASE_X * FP32_TILE_NBYTES + element);
                dy[0] = offset_to_fp32(off[2 * kk]);
                dx[0] = offset_to_fp32(off[2 * kk + 1]);
                m[0] = static_cast<uint32_t>(msk[kk]) << 16;
                by[0] = int_to_fp32(base_y + k_y);
                bx[0] = int_to_fp32(base_x + k_x);
            }
            if (++w_o == OUT_W) {
                w_o = 0;
                if (++h_o == OUT_H) {
                    h_o = 0;
                    ++b;
                }
            }
        }
        geom_in_cb.push_back(GEOM_IN_TILES);
    };

    if (num_blocks > 0) {
        stage_geometry(start_block);
    }

    uint32_t y0[TILE_ROWS];
    uint32_t x0[TILE_ROWS];
    uint16_t weight[CORNERS][TILE_ROWS];  // zero for a corner outside the input

    for (uint32_t block = start_block; block < start_block + num_blocks; ++block) {
        const uint32_t valid_rows = valid_rows_of(block, num_rows);
        for (uint32_t r = 0; r < valid_rows; ++r) {
            batch_row[r] = staged_batch_row[r];
        }
        index_cb.wait_front(2);
        weight_cb.wait_front(CORNERS);
        const uint32_t index_l1 = index_cb.get_read_ptr();
        const uint32_t weight_l1 = weight_cb.get_read_ptr();
        if (block + 1 < start_block + num_blocks) {
            stage_geometry(block + 1);
        }

        for (uint32_t kk = 0; kk < kernel_positions; ++kk) {
            for (uint32_t r = 0; r < valid_rows; ++r) {
                const uint32_t element = tile_element(r, kk);
                CoreLocalMem<volatile uint32_t> y0_tile(index_l1);
                CoreLocalMem<volatile uint32_t> x0_tile(index_l1 + FP32_TILE_NBYTES);
                y0[r] = y0_tile[element];
                x0[r] = x0_tile[element];
                const int32_t y = static_cast<int32_t>(y0[r]);
                const int32_t x = static_cast<int32_t>(x0[r]);
                const bool in_input[CORNERS] = {
                    inside(y, IN_H) && inside(x, IN_W),
                    inside(y, IN_H) && inside(x + 1, IN_W),
                    inside(y + 1, IN_H) && inside(x, IN_W),
                    inside(y + 1, IN_H) && inside(x + 1, IN_W)};
                for (uint32_t c = 0; c < CORNERS; ++c) {
                    CoreLocalMem<volatile uint16_t> w(weight_l1 + c * BF16_TILE_NBYTES);
                    weight[c][r] = in_input[c] ? w[element] : 0;
                }
            }

            corner_block_cb.reserve_back(CORNERS * CHANNEL_TILES);
            const uint32_t block_l1 = corner_block_cb.get_write_ptr();
            for (uint32_t c = 0; c < CORNERS; ++c) {
                const int32_t dy = c >> 1;
                const int32_t dx = c & 1;
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    const uint32_t dst_l1 = block_l1 + r * BLOCK_ROW_NBYTES + c * STICK_NBYTES;
                    // The page index below is valid only for a corner inside the input.
                    if (r >= valid_rows || weight[c][r] == 0) {
                        zero_stick(dst_l1);
                        continue;
                    }
                    const int32_t row = batch_row[r] + static_cast<int32_t>(y0[r]) + dy;
                    const uint32_t x = static_cast<uint32_t>(static_cast<int32_t>(x0[r]) + dx);
                    const uint32_t page = static_cast<uint32_t>(row) * IN_W + x;
                    CoreLocalMem<uint32_t> dst(dst_l1);
                    noc.async_read(input_acc, dst, STICK_NBYTES, {.page_id = page}, {.offset_bytes = 0});
                }
            }
            noc.async_read_barrier();
            corner_block_cb.push_back(CORNERS * CHANNEL_TILES);

            // Only column 0 of each row is read by the column broadcast.
            for (uint32_t c = 0; c < CORNERS; ++c) {
                scalar_tile_cb.reserve_back(1);
                const uint32_t scalar_l1 = scalar_tile_cb.get_write_ptr();
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    CoreLocalMem<volatile uint16_t> lane(scalar_l1 + tile_element(r, 0) * sizeof(uint16_t));
                    lane[0] = r < valid_rows ? weight[c][r] : 0;
                }
                scalar_tile_cb.push_back(1);
            }
        }
        index_cb.pop_front(2);
        weight_cb.pop_front(CORNERS);
    }
}
