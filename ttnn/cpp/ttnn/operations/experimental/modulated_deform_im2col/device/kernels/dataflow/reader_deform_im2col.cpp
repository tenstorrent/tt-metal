// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the modulated deformable im2col.
//
// Per block of 32 output rows (pixels) it stages the rows' offsets and mask
// logits, one stick per row, and hands compute one tile each of the y and x
// offsets, the mask logits and the rows' window origins, laid out [row, kernel
// position]; the kernel offsets of the positions it hands over once. Compute
// returns every sample's top-left corner page within its image and its four
// weights, zero for a corner outside the input, in the same layout. For each kernel position the reader then hands
// compute the four corners as one row-major block, 32 rows of
// [corner 0 | corner 1 | corner 2 | corner 3] C-wide input sticks, landed by the
// NoC reads themselves, plus one scalar tile per corner whose column 0 holds the
// row's weight. Compute tilizes the block and sums the four corners times their
// scalars, which is the masked bilinear sample.
//
// The geometry runs on compute because the data-movement RISC-Vs have no FPU;
// the reader only moves bits. Offsets saturate at +-OFFSET_CLAMP pixels, which
// keeps every coordinate exact in fp32 and puts a non-finite offset outside the
// input. Mask logits saturate the same way: a NaN one would give its row NaN
// weights, nonzero even for corners outside the input, whose pages the reader
// would then read. The next block's geometry is handed over before this block's corners
// are read, so compute works on it while the reader waits on the NoC.
//
// Two copies of this kernel run per core, one on each data-movement RISC-V and
// NoC, as issuing the corner reads is most of the work: the first takes the even
// kernel positions, the second the odd ones, one fewer, and stages the geometry.
// Each gets its own copy of the geometry from compute and its own corner, scalar
// and output buffers, which compute serves in kernel-position order, and writes
// the output of its own positions.
//
// A corner is read only when its weight is nonzero. Every other row, and every
// row past the block's end, is zeroed rather than skipped: a stale NaN or Inf
// times a zero scalar would still be NaN.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t offset_mask_arena_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t corner_block_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t scalar_tile_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t C = get_compile_time_arg_val(3);
constexpr int32_t IN_H = get_compile_time_arg_val(4);
constexpr int32_t IN_W = get_compile_time_arg_val(5);
constexpr uint32_t OUT_H = get_compile_time_arg_val(6);
constexpr uint32_t OUT_W = get_compile_time_arg_val(7);
constexpr uint32_t KERNEL_W = get_compile_time_arg_val(8);
constexpr uint32_t kernel_positions = get_compile_time_arg_val(9);
constexpr int32_t STRIDE_H = get_compile_time_arg_val(10);
constexpr int32_t STRIDE_W = get_compile_time_arg_val(11);
constexpr int32_t PAD_H = get_compile_time_arg_val(12);
constexpr int32_t PAD_W = get_compile_time_arg_val(13);
constexpr int32_t DILATION_H = get_compile_time_arg_val(14);
constexpr int32_t DILATION_W = get_compile_time_arg_val(15);
// Page strides of the two row-major tensors, as their buffers lay them out.
constexpr uint32_t input_page_nbytes = get_compile_time_arg_val(16);
constexpr uint32_t offset_mask_page_nbytes = get_compile_time_arg_val(17);
// The host validates that the input size, padding and kernel span stay below this.
constexpr uint32_t OFFSET_CLAMP = get_compile_time_arg_val(18);
constexpr uint32_t geom_in_cb_index = get_compile_time_arg_val(19);
constexpr uint32_t geom_index_cb_index = get_compile_time_arg_val(20);
constexpr uint32_t geom_weight_cb_index = get_compile_time_arg_val(21);
constexpr uint32_t geom_kernel_cb_index = get_compile_time_arg_val(22);
// This copy's role: whether it stages the geometry, and its first kernel position (it takes
// every other one).
constexpr bool STAGES_GEOMETRY = get_compile_time_arg_val(23);
constexpr uint32_t FIRST_POSITION = get_compile_time_arg_val(24);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(25);
constexpr uint32_t output_tile_nbytes = get_compile_time_arg_val(26);
// The two halves of the core's input band: the first copy fills the first, the second the other.
constexpr uint32_t band_a_cb_index = get_compile_time_arg_val(27);
constexpr uint32_t band_b_cb_index = get_compile_time_arg_val(28);

constexpr auto input_args = TensorAccessorArgs<29>();
constexpr auto offset_mask_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto output_args = TensorAccessorArgs<offset_mask_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t CHANNEL_TILES = C / 32;
constexpr uint32_t CORNERS = 4;
constexpr uint32_t STICK_NBYTES = C * sizeof(uint16_t);  // the C channels read per corner
constexpr uint32_t BLOCK_ROW_NBYTES = CORNERS * STICK_NBYTES;
constexpr uint32_t OUTPUT_TILES_PER_BLOCK = kernel_positions * CHANNEL_TILES;
// This copy's last kernel position, when it has any.
constexpr uint32_t LAST_POSITION =
    FIRST_POSITION < kernel_positions ? FIRST_POSITION + (kernel_positions - 1 - FIRST_POSITION) / 2 * 2 : 0;

// Tiles of geom_in_cb, per block, and of geom_kernel_cb, once, in compute's order: fp32
// [row, kernel position].
constexpr uint32_t IN_DY = 0;
constexpr uint32_t IN_DX = 1;
constexpr uint32_t IN_MASK = 2;  // the logits; compute applies the sigmoid
constexpr uint32_t IN_ORIGIN_Y = 3;
constexpr uint32_t IN_ORIGIN_X = 4;
constexpr uint32_t GEOM_IN_TILES = 5;
constexpr uint32_t KERNEL_Y = 0;
constexpr uint32_t KERNEL_X = 1;
constexpr uint32_t TILE_ELEMENTS = 32 * 32;

static_assert(kernel_positions <= 32, "the geometry tiles hold one kernel position per column");
static_assert((OFFSET_CLAMP & (OFFSET_CLAMP - 1)) == 0, "the offset clamp works on the bfloat16 exponent");

namespace {

// bfloat16 exponent field of OFFSET_CLAMP, a power of two, and OFFSET_CLAMP itself.
constexpr uint32_t CLAMP_EXPONENT = 127 + (31 - __builtin_clz(OFFSET_CLAMP));
constexpr uint16_t CLAMP_BF16 = static_cast<uint16_t>(CLAMP_EXPONENT << 7);

// A value of magnitude OFFSET_CLAMP or more, inf and NaN, become +-OFFSET_CLAMP.
inline uint32_t saturate_to_fp32(uint16_t bf16) {
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

// Zeros a stick with NoC reads of the core's zero region, which cost one read issue each where
// storing the words would cost one store per word; corners outside the input are many on cores
// whose rows lie along an image edge, and the slowest core sets the op's time.
inline void zero_stick(uint32_t l1_addr) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t offset = 0; offset < STICK_NBYTES; offset += MEM_ZEROS_SIZE) {
        const uint32_t nbytes = STICK_NBYTES - offset < MEM_ZEROS_SIZE ? STICK_NBYTES - offset : MEM_ZEROS_SIZE;
        noc_async_read(zeros, l1_addr + offset, nbytes);
    }
}

inline uint32_t valid_rows_of(uint32_t block, uint32_t num_rows) {
    const uint32_t row0 = block * TILE_ROWS;
    return num_rows - row0 < TILE_ROWS ? num_rows - row0 : TILE_ROWS;
}

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t offset_mask_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_block = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t num_rows = get_arg_val<uint32_t>(4);
    const uint32_t output_addr = get_arg_val<uint32_t>(5);
    // The core's band of input pages, [band_first_page, band_first_page + band_pages).
    const uint32_t band_first_page = get_arg_val<uint32_t>(6);
    const uint32_t band_pages = get_arg_val<uint32_t>(7);

    const auto input_acc = TensorAccessor(input_args, input_addr, input_page_nbytes);
    const auto offset_mask_acc = TensorAccessor(offset_mask_args, offset_mask_addr, offset_mask_page_nbytes);
    const auto output_acc = TensorAccessor(output_args, output_addr, output_tile_nbytes);

    Noc noc;
    CircularBuffer corner_block_cb(corner_block_cb_index);
    CircularBuffer scalar_tile_cb(scalar_tile_cb_index);
    CircularBuffer geom_in_cb(geom_in_cb_index);
    CircularBuffer index_cb(geom_index_cb_index);
    CircularBuffer weight_cb(geom_weight_cb_index);
    CircularBuffer output_cb(output_cb_index);

    uint32_t offset_mask_l1 = 0;
    if constexpr (STAGES_GEOMETRY) {
        // The arena is reserved once and used as flat L1; nothing waits on it.
        CircularBuffer offset_mask_arena_cb(offset_mask_arena_cb_index);
        offset_mask_arena_cb.reserve_back(TILE_ROWS);
        offset_mask_l1 = offset_mask_arena_cb.get_write_ptr();

        CircularBuffer geom_kernel_cb(geom_kernel_cb_index);
        geom_kernel_cb.reserve_back(2);
        const uint32_t kernel_l1 = geom_kernel_cb.get_write_ptr();
        for (uint32_t r = 0; r < TILE_ROWS; ++r) {
            for (uint32_t kk = 0; kk < kernel_positions; ++kk) {
                const uint32_t element = tile_element(r, kk) * sizeof(uint32_t);
                CoreLocalMem<volatile uint32_t> ky(kernel_l1 + KERNEL_Y * TILE_ELEMENTS * sizeof(uint32_t) + element);
                CoreLocalMem<volatile uint32_t> kx(kernel_l1 + KERNEL_X * TILE_ELEMENTS * sizeof(uint32_t) + element);
                ky[0] = int_to_fp32(static_cast<int32_t>(kk / KERNEL_W) * DILATION_H);
                kx[0] = int_to_fp32(static_cast<int32_t>(kk % KERNEL_W) * DILATION_W);
            }
        }
        geom_kernel_cb.push_back(2);
    }

    // Stages a block's offsets and mask logits and hands compute its geometry tiles.
    auto stage_geometry = [&](uint32_t block) {
        const uint32_t row0 = block * TILE_ROWS;
        const uint32_t valid_rows = valid_rows_of(block, num_rows);
        for (uint32_t r = 0; r < valid_rows; ++r) {
            CoreLocalMem<uint32_t> dst(offset_mask_l1 + r * offset_mask_page_nbytes);
            noc.async_read(offset_mask_acc, dst, offset_mask_page_nbytes, {.page_id = row0 + r}, {.offset_bytes = 0});
        }
        geom_in_cb.reserve_back(GEOM_IN_TILES);
        const uint32_t geom_l1 = geom_in_cb.get_write_ptr();
        noc.async_read_barrier();

        // (h_out, w_out) of the block's first row, then stepped row by row.
        uint32_t h_o = (row0 / OUT_W) % OUT_H;
        uint32_t w_o = row0 % OUT_W;
        tt_l1_ptr uint32_t* geom = reinterpret_cast<tt_l1_ptr uint32_t*>(geom_l1);
        for (uint32_t r = 0; r < valid_rows; ++r) {
            const uint32_t origin_y = int_to_fp32(static_cast<int32_t>(h_o) * STRIDE_H - PAD_H);
            const uint32_t origin_x = int_to_fp32(static_cast<int32_t>(w_o) * STRIDE_W - PAD_W);
            const tt_l1_ptr uint16_t* off =
                reinterpret_cast<const tt_l1_ptr uint16_t*>(offset_mask_l1 + r * offset_mask_page_nbytes);
            // The mask logits follow the row's 2*K*K offsets.
            const tt_l1_ptr uint16_t* msk = off + 2 * kernel_positions;
            tt_l1_ptr uint32_t* row = geom + tile_element(r, 0);
            for (uint32_t kk = 0; kk < kernel_positions; ++kk) {
                tt_l1_ptr uint32_t* dst = row + tile_element(0, kk);
                dst[IN_DY * TILE_ELEMENTS] = saturate_to_fp32(off[2 * kk]);
                dst[IN_DX * TILE_ELEMENTS] = saturate_to_fp32(off[2 * kk + 1]);
                dst[IN_MASK * TILE_ELEMENTS] = saturate_to_fp32(msk[kk]);
                dst[IN_ORIGIN_Y * TILE_ELEMENTS] = origin_y;
                dst[IN_ORIGIN_X * TILE_ELEMENTS] = origin_x;
            }
            if (++w_o == OUT_W) {
                w_o = 0;
                if (++h_o == OUT_H) {
                    h_o = 0;
                }
            }
        }
        geom_in_cb.push_back(GEOM_IN_TILES);
    };

    // Writes the block's output of one of this copy's kernel positions.
    auto write_output = [&](uint32_t block, uint32_t kk) {
        output_cb.wait_front(CHANNEL_TILES);
        const uint32_t tiles_l1 = output_cb.get_read_ptr();
        const uint32_t first_tile = block * OUTPUT_TILES_PER_BLOCK + kk * CHANNEL_TILES;
        for (uint32_t t = 0; t < CHANNEL_TILES; ++t) {
            CoreLocalMem<uint32_t> src(tiles_l1 + t * output_tile_nbytes);
            noc.async_write(src, output_acc, output_tile_nbytes, {.offset_bytes = 0}, {.page_id = first_tile + t});
        }
        // The tiles only need to have left L1 before compute refills the slots.
        noc.async_writes_flushed();
        output_cb.pop_front(CHANNEL_TILES);
    };

    if constexpr (STAGES_GEOMETRY) {
        if (num_blocks > 0) {
            stage_geometry(start_block);
        }
    }

    // Each copy brings in its half of the band, then both wait for the whole of it.
    const uint32_t band_half = (band_pages + 1) / 2;
    CircularBuffer band_a_cb(band_a_cb_index);
    CircularBuffer band_b_cb(band_b_cb_index);
    {
        CircularBuffer& own = FIRST_POSITION == 0 ? band_a_cb : band_b_cb;
        const uint32_t begin = FIRST_POSITION == 0 ? 0 : band_half;
        const uint32_t end = FIRST_POSITION == 0 ? band_half : band_pages;
        own.reserve_back(1);
        const uint32_t own_l1 = own.get_write_ptr();
        for (uint32_t p = begin; p < end; ++p) {
            CoreLocalMem<uint32_t> dst(own_l1 + (p - begin) * input_page_nbytes);
            noc.async_read(input_acc, dst, STICK_NBYTES, {.page_id = band_first_page + p}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        own.push_back(1);
    }
    band_a_cb.wait_front(1);
    band_b_cb.wait_front(1);
    const uint32_t band_a_l1 = band_a_cb.get_read_ptr();
    const uint32_t band_b_l1 = band_b_cb.get_read_ptr();

    uint32_t image_page[TILE_ROWS];  // first page of each row's image
    uint32_t page[TILE_ROWS];
    uint16_t weight[CORNERS][TILE_ROWS];

    for (uint32_t block = start_block; block < start_block + num_blocks; ++block) {
        const uint32_t valid_rows = valid_rows_of(block, num_rows);
        uint32_t b = block * TILE_ROWS / (OUT_H * OUT_W);
        uint32_t rows_left_in_image = (b + 1) * OUT_H * OUT_W - block * TILE_ROWS;
        for (uint32_t r = 0; r < valid_rows; ++r) {
            image_page[r] = b * static_cast<uint32_t>(IN_H * IN_W);
            if (--rows_left_in_image == 0) {
                ++b;
                rows_left_in_image = OUT_H * OUT_W;
            }
        }
        index_cb.wait_front(1);
        weight_cb.wait_front(CORNERS);
        const uint32_t index_l1 = index_cb.get_read_ptr();
        const uint32_t weight_l1 = weight_cb.get_read_ptr();
        if constexpr (STAGES_GEOMETRY) {
            if (block + 1 < start_block + num_blocks) {
                stage_geometry(block + 1);
            }
        }

        for (uint32_t kk = FIRST_POSITION; kk < kernel_positions; kk += 2) {
            // Only column 0 of each row is read by the column broadcast.
            scalar_tile_cb.reserve_back(CORNERS);
            const uint32_t scalar_l1 = scalar_tile_cb.get_write_ptr();
            const tt_l1_ptr uint32_t* index =
                reinterpret_cast<const tt_l1_ptr uint32_t*>(index_l1) + tile_element(0, kk);
            const tt_l1_ptr uint16_t* w = reinterpret_cast<const tt_l1_ptr uint16_t*>(weight_l1) + tile_element(0, kk);
            tt_l1_ptr uint16_t* lane = reinterpret_cast<tt_l1_ptr uint16_t*>(scalar_l1);
            for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                const uint32_t row = tile_element(r, 0);
                page[r] = image_page[r] + index[row];
                for (uint32_t c = 0; c < CORNERS; ++c) {
                    const uint16_t value = r < valid_rows ? w[c * TILE_ELEMENTS + row] : 0;
                    weight[c][r] = value;
                    lane[c * TILE_ELEMENTS + row] = value;
                }
            }
            scalar_tile_cb.push_back(CORNERS);

            corner_block_cb.reserve_back(CORNERS * CHANNEL_TILES);
            const uint32_t block_l1 = corner_block_cb.get_write_ptr();
            for (uint32_t c = 0; c < CORNERS; ++c) {
                const uint32_t corner_page_offset = (c >> 1) * IN_W + (c & 1);
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    const uint32_t dst_l1 = block_l1 + r * BLOCK_ROW_NBYTES + c * STICK_NBYTES;
                    // A zero weight also marks a corner outside the input, whose page is not one.
                    if (weight[c][r] == 0) {
                        zero_stick(dst_l1);
                        continue;
                    }
                    const uint32_t page_id = page[r] + corner_page_offset;
                    // Unsigned, so a page before the band also falls outside it.
                    const uint32_t in_band = page_id - band_first_page;
                    if (in_band < band_pages) {
                        const uint32_t src_l1 = in_band < band_half
                                                    ? band_a_l1 + in_band * input_page_nbytes
                                                    : band_b_l1 + (in_band - band_half) * input_page_nbytes;
                        noc_async_read(get_noc_addr(src_l1), dst_l1, STICK_NBYTES);
                    } else {
                        CoreLocalMem<uint32_t> dst(dst_l1);
                        noc.async_read(input_acc, dst, STICK_NBYTES, {.page_id = page_id}, {.offset_bytes = 0});
                    }
                }
            }
            noc.async_read_barrier();
            corner_block_cb.push_back(CORNERS * CHANNEL_TILES);

            // This copy's previous position, now that its next corners are on their way to compute.
            if (kk >= FIRST_POSITION + 2) {
                write_output(block, kk - 2);
            }
        }
        if constexpr (FIRST_POSITION < kernel_positions) {
            write_output(block, LAST_POSITION);
        }
        index_cb.pop_front(1);
        weight_cb.pop_front(CORNERS);
    }
    noc.async_write_barrier();
}
