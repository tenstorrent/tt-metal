// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the modulated deformable conv: it gathers the bilinear corners each block's columns are
// sampled from, for a matmul that runs in the same op, so the columns never leave L1.
//
// The kernel positions run outermost and each core's blocks of 32 output rows inside them, so one
// position's slice of the weight serves all of the core's blocks before the next slice replaces it.
// An input wider than a chunk of channels takes one pass per chunk at each position, with that
// chunk's rows of the weight; a pass reads only its chunk of each corner. Two copies of this kernel
// run per core, one on each data-movement RISC-V and NoC: the first takes the core's even blocks
// and loads the weight slices and the bias, the second the odd blocks and stages the geometry.
// Compute returns each block's geometry, every sample's top-left corner page within its image and
// its four weights, to the copy that owns the block, which keeps it for all kernel positions.
//
// The geometry is staged during the first kernel position, two blocks ahead of compute, and compute
// derives it just before the block's corners. For each (pass, block) a copy hands compute the four
// corners as one row-major block, 32 rows of [corner 0 | corner 1 | corner 2 | corner 3] chunk-wide
// input sticks, landed by the NoC reads, and one scalar tile per corner whose column 0 holds the
// row's weight. After the last position it writes its blocks' C_out-wide output.
//
// A corner is read only when its weight is nonzero. Every other row, and every row past the block's
// end, is zeroed rather than skipped: a stale NaN or Inf times a zero scalar would still be NaN.
//
// The geometry runs on compute because the data-movement RISC-Vs have no FPU; the reader only moves
// bits. Offsets saturate at +-OFFSET_CLAMP pixels, which keeps every coordinate exact in fp32 and
// puts a non-finite offset outside the input. Mask logits saturate the same way: a NaN one would
// give its row NaN weights, nonzero even for corners outside the input, whose pages the reader
// would then read.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "hostdevcommon/common_values.hpp"
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
// This copy's role: the parity of the blocks it takes. The first copy (even blocks) loads the
// weight and the bias, the second stages the geometry.
constexpr uint32_t BLOCK_PARITY = get_compile_time_arg_val(23);
constexpr bool STAGES_GEOMETRY = BLOCK_PARITY == 1;
constexpr bool LOADS_WEIGHT = BLOCK_PARITY == 0;
constexpr uint32_t output_cb_index = get_compile_time_arg_val(24);
constexpr uint32_t tile_nbytes = get_compile_time_arg_val(25);
// The two halves of the core's input band: the first copy fills the first, the second the other.
constexpr uint32_t band_a_cb_index = get_compile_time_arg_val(26);
constexpr uint32_t band_b_cb_index = get_compile_time_arg_val(27);
constexpr uint32_t conv_weight_cb_index = get_compile_time_arg_val(28);
constexpr uint32_t bias_cb_index = get_compile_time_arg_val(29);
constexpr uint32_t OUT_TILES = get_compile_time_arg_val(30);  // C_out / 32
constexpr bool HAS_BIAS = get_compile_time_arg_val(31);
// One core reads each weight slice from DRAM and multicasts it to the others, so the slices cost
// one DRAM read instead of one per core; the per-core reads otherwise congest the corner gather.
constexpr bool MCAST_WEIGHT = get_compile_time_arg_val(32);
constexpr uint32_t MCAST_NUM_DESTS = get_compile_time_arg_val(33);
constexpr uint32_t weight_sender_sem_id = get_compile_time_arg_val(34);
constexpr uint32_t weight_receiver_sem_id = get_compile_time_arg_val(35);
constexpr uint32_t CHUNK_TILES = get_compile_time_arg_val(36);  // input channels per pass / 32
constexpr auto input_args = TensorAccessorArgs<37>();
constexpr auto offset_mask_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto output_args = TensorAccessorArgs<offset_mask_args.next_compile_time_args_offset()>();
constexpr auto weight_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t CHANNEL_TILES = C / 32;
constexpr uint32_t CHUNKS = CHANNEL_TILES / CHUNK_TILES;
constexpr uint32_t PASSES = kernel_positions * CHUNKS;
constexpr uint32_t CORNERS = 4;
constexpr uint32_t PAGE_NBYTES = C * sizeof(uint16_t);  // an input page's channels, as the band holds them
constexpr uint32_t STICK_NBYTES = CHUNK_TILES * 32 * sizeof(uint16_t);  // the channels a pass reads per corner
constexpr uint32_t BLOCK_ROW_NBYTES = CORNERS * STICK_NBYTES;
// One pass's slice of the (K*K*C, C_out) weight, CHUNK_TILES x OUT_TILES tiles. The weight's rows
// run (kernel position, channel), so pass p's slice is the p-th run of WEIGHT_SLICE_TILES tiles.
constexpr uint32_t WEIGHT_SLICE_TILES = CHUNK_TILES * OUT_TILES;
static_assert(CHANNEL_TILES % CHUNK_TILES == 0, "the channels split into whole chunks");

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
    const uint32_t weight_addr = get_arg_val<uint32_t>(8);
    const uint32_t bias_addr = get_arg_val<uint32_t>(9);
    // The multicast's sender and its destination rectangle, in NoC coordinates.
    const bool is_weight_sender = get_arg_val<uint32_t>(10);
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(11);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(12);
    uint32_t mcast_x0 = get_arg_val<uint32_t>(13);
    uint32_t mcast_y0 = get_arg_val<uint32_t>(14);
    uint32_t mcast_x1 = get_arg_val<uint32_t>(15);
    uint32_t mcast_y1 = get_arg_val<uint32_t>(16);
    if (noc_index == 1) {
        // NoC 1 runs the other way, so its rectangle starts at the far corner.
        const uint32_t x = mcast_x0, y = mcast_y0;
        mcast_x0 = mcast_x1;
        mcast_y0 = mcast_y1;
        mcast_x1 = x;
        mcast_y1 = y;
    }

    const auto input_acc = TensorAccessor(input_args, input_addr, input_page_nbytes);
    const auto offset_mask_acc = TensorAccessor(offset_mask_args, offset_mask_addr, offset_mask_page_nbytes);
    const auto output_acc = TensorAccessor(output_args, output_addr, tile_nbytes);
    const auto weight_acc = TensorAccessor(weight_args, weight_addr, tile_nbytes);
    const auto bias_acc = TensorAccessor(bias_args, bias_addr, tile_nbytes);

    Noc noc;
    CircularBuffer corner_block_cb(corner_block_cb_index);
    CircularBuffer scalar_tile_cb(scalar_tile_cb_index);
    CircularBuffer geom_in_cb(geom_in_cb_index);
    CircularBuffer index_cb(geom_index_cb_index);
    CircularBuffer corner_weight_cb(geom_weight_cb_index);
    CircularBuffer output_cb(output_cb_index);
    CircularBuffer conv_weight_cb(conv_weight_cb_index);

    Semaphore<> weight_sender_sem(weight_sender_sem_id);
    Semaphore<> weight_receiver_sem(weight_receiver_sem_id);
    if (LOADS_WEIGHT && MCAST_WEIGHT && is_weight_sender) {
        // The value the sender multicasts into every receiver's flag once a slice has landed.
        weight_receiver_sem.set(VALID);
    }

    // Loads one pass's slice of the weight, tiles in row-major order, in two steps so the wait for
    // it can trail the pass's corners: begin reserves the slot and, on a receiver, reports it free;
    // finish reads or awaits the data and hands it to compute, which needs it only after the pass's
    // last block. Every core reserves the same slot in the same
    // order, so the slot's address matches on all of them.
    uint32_t weight_slot_l1 = 0;
    auto begin_weight_slice = [&]() {
        conv_weight_cb.reserve_back(WEIGHT_SLICE_TILES);
        weight_slot_l1 = conv_weight_cb.get_write_ptr();
        if (MCAST_WEIGHT && !is_weight_sender) {
            weight_receiver_sem.set(INVALID);
            weight_sender_sem.up(noc, sender_noc_x, sender_noc_y, 1);
        }
    };
    auto finish_weight_slice = [&](uint32_t pass) {
        if (!MCAST_WEIGHT || is_weight_sender) {
            const uint32_t first_tile = pass * WEIGHT_SLICE_TILES;
            for (uint32_t t = 0; t < WEIGHT_SLICE_TILES; ++t) {
                CoreLocalMem<uint32_t> dst(weight_slot_l1 + t * tile_nbytes);
                noc.async_read(weight_acc, dst, tile_nbytes, {.page_id = first_tile + t}, {.offset_bytes = 0});
            }
            noc.async_read_barrier();
            if constexpr (MCAST_WEIGHT) {
                // Every receiver has freed the slot.
                weight_sender_sem.wait(MCAST_NUM_DESTS);
                weight_sender_sem.set(0);
                const MulticastEndpoint mcast_dst;
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(weight_slot_l1),
                    mcast_dst,
                    WEIGHT_SLICE_TILES * tile_nbytes,
                    MCAST_NUM_DESTS,
                    {},
                    {.noc_x_start = mcast_x0,
                     .noc_y_start = mcast_y0,
                     .noc_x_end = mcast_x1,
                     .noc_y_end = mcast_y1,
                     .addr = weight_slot_l1},
                    true);
                noc.async_writes_flushed();
                weight_receiver_sem.set_multicast(noc, mcast_x0, mcast_y0, mcast_x1, mcast_y1, MCAST_NUM_DESTS);
            }
        } else {
            weight_receiver_sem.wait(VALID);
        }
        conv_weight_cb.push_back(WEIGHT_SLICE_TILES);
    };
    auto load_weight_slice = [&](uint32_t pass) {
        begin_weight_slice();
        finish_weight_slice(pass);
    };

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

    if constexpr (STAGES_GEOMETRY) {
        // During the first kernel position compute derives each block's geometry two blocks
        // ahead of its corners; staging keeps the next two blocks ready, the depth of geom_in_cb.
        for (uint32_t j = 0; j < num_blocks && j < 2; ++j) {
            stage_geometry(start_block + j);
        }
    }
    if constexpr (LOADS_WEIGHT) {
        load_weight_slice(0);
        if constexpr (HAS_BIAS) {
            CircularBuffer bias_cb(bias_cb_index);
            bias_cb.reserve_back(OUT_TILES);
            const uint32_t dst_l1 = bias_cb.get_write_ptr();
            for (uint32_t t = 0; t < OUT_TILES; ++t) {
                CoreLocalMem<uint32_t> dst(dst_l1 + t * tile_nbytes);
                noc.async_read(bias_acc, dst, tile_nbytes, {.page_id = t}, {.offset_bytes = 0});
            }
            noc.async_read_barrier();
            bias_cb.push_back(OUT_TILES);
        }
    }

    // Each copy brings in its half of the band, then both wait for the whole of it.
    const uint32_t band_half = (band_pages + 1) / 2;
    CircularBuffer band_a_cb(band_a_cb_index);
    CircularBuffer band_b_cb(band_b_cb_index);
    {
        CircularBuffer& own = BLOCK_PARITY == 0 ? band_a_cb : band_b_cb;
        const uint32_t begin = BLOCK_PARITY == 0 ? 0 : band_half;
        const uint32_t end = BLOCK_PARITY == 0 ? band_half : band_pages;
        own.reserve_back(1);
        const uint32_t own_l1 = own.get_write_ptr();
        for (uint32_t p = begin; p < end; ++p) {
            CoreLocalMem<uint32_t> dst(own_l1 + (p - begin) * input_page_nbytes);
            noc.async_read(input_acc, dst, PAGE_NBYTES, {.page_id = band_first_page + p}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        own.push_back(1);
    }
    band_a_cb.wait_front(1);
    band_b_cb.wait_front(1);
    const uint32_t band_a_l1 = band_a_cb.get_read_ptr();
    const uint32_t band_b_l1 = band_b_cb.get_read_ptr();

    const uint32_t index_tile_nbytes = get_tile_size(geom_index_cb_index);
    const uint32_t corner_weight_slot_nbytes = CORNERS * get_tile_size(geom_weight_cb_index);
    uint32_t image_page[TILE_ROWS];  // first page of each row's image
    uint32_t page[TILE_ROWS];
    uint16_t corner_weight[CORNERS][TILE_ROWS];

    for (uint32_t pass = 0; pass < PASSES; ++pass) {
        const uint32_t kk = pass / CHUNKS;
        const uint32_t chunk_offset = (pass % CHUNKS) * STICK_NBYTES;
        if constexpr (LOADS_WEIGHT) {
            // The next slice loads while compute works through this one; the CB holds two.
            if (pass + 1 < PASSES) {
                begin_weight_slice();
            }
        }
        uint32_t slot = 0;  // this copy's blocks, in order; their geometry stays in its CBs
        for (uint32_t j = BLOCK_PARITY; j < num_blocks; j += 2, ++slot) {
            const uint32_t block = start_block + j;
            const uint32_t valid_rows = valid_rows_of(block, num_rows);
            if (pass == 0) {
                if constexpr (STAGES_GEOMETRY) {
                    for (uint32_t ahead = j + 1; ahead <= j + 2 && ahead < num_blocks; ++ahead) {
                        if (ahead >= 2) {
                            stage_geometry(start_block + ahead);
                        }
                    }
                }
                index_cb.wait_front(slot + 1);
                corner_weight_cb.wait_front(CORNERS * (slot + 1));
            }
            const uint32_t index_l1 = index_cb.get_read_ptr() + slot * index_tile_nbytes;
            const uint32_t corner_weight_l1 = corner_weight_cb.get_read_ptr() + slot * corner_weight_slot_nbytes;

            uint32_t b = block * TILE_ROWS / (OUT_H * OUT_W);
            uint32_t rows_left_in_image = (b + 1) * OUT_H * OUT_W - block * TILE_ROWS;
            for (uint32_t r = 0; r < valid_rows; ++r) {
                image_page[r] = b * static_cast<uint32_t>(IN_H * IN_W);
                if (--rows_left_in_image == 0) {
                    ++b;
                    rows_left_in_image = OUT_H * OUT_W;
                }
            }

            // Only column 0 of each row is read by the column broadcast.
            scalar_tile_cb.reserve_back(CORNERS);
            const uint32_t scalar_l1 = scalar_tile_cb.get_write_ptr();
            const tt_l1_ptr uint32_t* index =
                reinterpret_cast<const tt_l1_ptr uint32_t*>(index_l1) + tile_element(0, kk);
            const tt_l1_ptr uint16_t* w =
                reinterpret_cast<const tt_l1_ptr uint16_t*>(corner_weight_l1) + tile_element(0, kk);
            tt_l1_ptr uint16_t* lane = reinterpret_cast<tt_l1_ptr uint16_t*>(scalar_l1);
            for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                const uint32_t row = tile_element(r, 0);
                page[r] = r < valid_rows ? image_page[r] + index[row] : 0;
                for (uint32_t c = 0; c < CORNERS; ++c) {
                    const uint16_t value = r < valid_rows ? w[c * TILE_ELEMENTS + row] : 0;
                    corner_weight[c][r] = value;
                    lane[c * TILE_ELEMENTS + row] = value;
                }
            }
            scalar_tile_cb.push_back(CORNERS);

            corner_block_cb.reserve_back(CORNERS * CHUNK_TILES);
            const uint32_t block_l1 = corner_block_cb.get_write_ptr();
            for (uint32_t c = 0; c < CORNERS; ++c) {
                const uint32_t corner_page_offset = (c >> 1) * IN_W + (c & 1);
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    const uint32_t dst_l1 = block_l1 + r * BLOCK_ROW_NBYTES + c * STICK_NBYTES;
                    // A zero weight also marks a corner outside the input, whose page is not one.
                    if (corner_weight[c][r] == 0) {
                        zero_stick(dst_l1);
                        continue;
                    }
                    const uint32_t page_id = page[r] + corner_page_offset;
                    // Unsigned, so a page before the band also falls outside it.
                    const uint32_t in_band = page_id - band_first_page;
                    if (in_band < band_pages) {
                        const uint32_t src_l1 =
                            chunk_offset + (in_band < band_half
                                                ? band_a_l1 + in_band * input_page_nbytes
                                                : band_b_l1 + (in_band - band_half) * input_page_nbytes);
                        noc_async_read(get_noc_addr(src_l1), dst_l1, STICK_NBYTES);
                    } else {
                        CoreLocalMem<uint32_t> dst(dst_l1);
                        noc.async_read(
                            input_acc,
                            dst,
                            STICK_NBYTES,
                            {.page_id = page_id, .offset_bytes = chunk_offset},
                            {.offset_bytes = 0});
                    }
                }
            }
            noc.async_read_barrier();
            corner_block_cb.push_back(CORNERS * CHUNK_TILES);
        }
        if constexpr (LOADS_WEIGHT) {
            if (pass + 1 < PASSES) {
                finish_weight_slice(pass + 1);
            }
        }
    }

    // The conv's output of this copy's blocks, C_out wide.
    for (uint32_t j = BLOCK_PARITY; j < num_blocks; j += 2) {
        output_cb.wait_front(OUT_TILES);
        const uint32_t tiles_l1 = output_cb.get_read_ptr();
        const uint32_t first_tile = (start_block + j) * OUT_TILES;
        for (uint32_t t = 0; t < OUT_TILES; ++t) {
            CoreLocalMem<uint32_t> src(tiles_l1 + t * tile_nbytes);
            noc.async_write(src, output_acc, tile_nbytes, {.offset_bytes = 0}, {.page_id = first_tile + t});
        }
        noc.async_writes_flushed();
        output_cb.pop_front(OUT_TILES);
    }
    noc.async_write_barrier();
}
