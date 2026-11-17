// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 reader (NCRISC / NoC0): per M-block, fill_halo_slab then gather_im2col_blocks.
//
// No kernel_lib helper expresses the gather: read_sticks_for_tilize reads consecutive sticks at one
// offset, while an im2col row is up to 3 taps from 3 pixels with zero-fill for out-of-image taps and
// K padding. A "gather rows by index list with zero-fill into tilize-layout pages" dataflow helper
// would close the gap. Fencing uses the local_copy helpers' transaction-id calls.
//
// Slab layout: a pixel with flat index v of band kh (kernel row kh's input pixels for this block)
// lives at slab pixel v + slab_x_base[kh].
//   ROW_MAJOR: the reader fills the slab from DRAM pages; band kh starts at slab pixel
//     kh * band_stride_px. With band_stride_px == dilation * W the bands coincide where they overlap,
//     so each input pixel is read once.
//   TILE: the reader streams the band's input tile-rows to compute, which untilizes them into the
//     slab (compute -> reader CB). A band starting inside the previous band's tile-rows shares them.
// Both fills read page runs through one TensorAccessor: on a HEIGHT-sharded input a run is the
// contiguous part of one owner core's shard (own or remote), so a band costs one NoC read per shard it
// crosses; on an interleaved input every page is its own read.
// Channel slices (BLOCK_SHARDED, K split over the grid's columns): the core reads only its column's
// slice of each input row (pixel or tile-row). Those pages are not consecutive page ids (the other
// columns' slices sit between them), but one owner shard stores its rows of the slice back to back, so
// a run is the part of a band inside one shard.
// With combine_split_nocs the reader also sends the higher grid columns' N-slices of each M-block's
// partials on NoC0 (see the writer's file comment): it waits for the writer's send_go, writes, fences and
// raises send_done. It never waits on or pops the partials CB; the writer owns that CB's lifecycle.
//
// im2col layout: by default one row per output pixel ([32, block_k_cols] per tile-row), costing one read
// (or one per tap) per pixel per K-block. Tap-major (stride 1, whole unpadded taps): each tap of a tile-row
// is its own [32, C_in_p] block; the pixels of one output image row read consecutive slab pixels, so a
// tap costs one read per image row it spans, plus zero-fills for padding.
//
// Split slab fill (HEIGHT-sharded ROW_MAJOR): a read response travels the requesting NoC's direction from the
// owner core, so on NoC0 a run owned by a higher-index shard (to the right / below) wraps around the ring. The
// reader reads the runs of its own and lower shards; the writer reads the higher shards' runs on NoC1 into the
// same slab and raises slab_fwd_done per M-block. The reader raises slab_free once a block's gather has
// landed, so the writer never refills a slab still being read.
//
// Face-direct im2col (see conv2d_3x3_gather_dataflow.hpp): the reader reserves the K-block's in0 tiles and pushes
// them in row groups of in0_stream_rows tile-rows, in order, so compute multiplies each group as it lands. With a
// split gather it raises gather_go once the block is reserved and the slab complete; the writer then gathers every
// odd row group and counts them in gather_done, while the reader gathers the even ones. Only the reader touches
// the CB's pointers; the writer writes into the reserved block, which starts at the CB base.
//
// Lanes C_in..C_in_p of a slab pixel are not data (DRAM page padding or tile padding) and may hold
// NaN. The gather copies only the C_in real channels of each tap; the lanes of an im2col row are
// zero either from a one-time zeroing of cb_im2col_rm (when every K-block has the same column
// layout, so data never lands where another K-block keeps lanes or K padding) or from per-row
// zero-fills.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "api/semaphore.h"
#include "conv2d_3x3_combine_dataflow.hpp"
#include "conv2d_3x3_slab_dataflow.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t ELEM_BYTES = 2;        // bf16 slab and im2col rows
constexpr uint32_t TRID_GATHER = 4;       // ROW_MAJOR bands use trids 1..3
constexpr uint32_t TRID_INPUT_TILES = 5;  // TILE footprint batches

using conv2d_3x3::TaggedReads;

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_halo_slab = get_compile_time_arg_val(0);
    constexpr uint32_t cb_im2col_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(2);
    constexpr int32_t input_height = get_compile_time_arg_val(3);
    constexpr int32_t input_width = get_compile_time_arg_val(4);
    constexpr uint32_t output_width = get_compile_time_arg_val(5);
    constexpr uint32_t num_out_px = get_compile_time_arg_val(6);
    constexpr uint32_t c_in = get_compile_time_arg_val(7);
    constexpr uint32_t c_in_p = get_compile_time_arg_val(8);
    constexpr int32_t padding = get_compile_time_arg_val(9);
    constexpr int32_t stride = get_compile_time_arg_val(10);
    constexpr int32_t dilation = get_compile_time_arg_val(11);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t tensor_kkh_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(15);
    constexpr int32_t band_stride_px = get_compile_time_arg_val(16);
    constexpr bool tile_input = get_compile_time_arg_val(17) != 0;
    constexpr uint32_t input_page_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t slab_tile_rows = get_compile_time_arg_val(19);
    constexpr uint32_t untilize_batch_rows = get_compile_time_arg_val(20);
    constexpr uint32_t im2col_rm_depth_rows = get_compile_time_arg_val(21);
    constexpr bool channel_slice = get_compile_time_arg_val(22) != 0;
    constexpr uint32_t input_shard_rows = get_compile_time_arg_val(23);
    constexpr uint32_t row_page_stride = get_compile_time_arg_val(24);
    constexpr bool combine_send = get_compile_time_arg_val(25) != 0;
    constexpr uint32_t k_cols = get_compile_time_arg_val(26);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(27);
    constexpr uint32_t out_n_tiles = get_compile_time_arg_val(28);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(29);
    constexpr uint32_t cb_partials = get_compile_time_arg_val(30);
    constexpr uint32_t cb_combine_slots = get_compile_time_arg_val(31);
    constexpr uint32_t combine_send_go_sem_id = get_compile_time_arg_val(32);
    constexpr uint32_t combine_send_done_sem_id = get_compile_time_arg_val(33);
    constexpr bool tap_major = get_compile_time_arg_val(34) != 0;
    constexpr bool face_direct = get_compile_time_arg_val(35) != 0;
    constexpr int32_t band_last_tap = get_compile_time_arg_val(36);
    constexpr uint32_t cb_im2col_tiles = get_compile_time_arg_val(37);
    constexpr bool slab_split = get_compile_time_arg_val(38) != 0;
    constexpr uint32_t slab_fwd_done_sem_id = get_compile_time_arg_val(39);
    constexpr uint32_t slab_free_sem_id = get_compile_time_arg_val(40);
    constexpr bool gather_split = get_compile_time_arg_val(41) != 0;
    constexpr uint32_t in0_stream_rows = get_compile_time_arg_val(42);
    constexpr uint32_t gather_go_sem_id = get_compile_time_arg_val(43);
    constexpr uint32_t gather_done_sem_id = get_compile_time_arg_val(44);
    constexpr auto input_args = TensorAccessorArgs<45>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t actual_m_tiles = get_arg_val<uint32_t>(2);
    const uint32_t num_m_blocks_this_core = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t read_rotation = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t slice_page_offset = get_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint32_t my_col = get_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t my_shard = get_arg_val<uint32_t>(7);
    constexpr uint32_t row_peer_coords_arg = 8;  // (x, y) of grid columns 0..k_cols-1 of this row

    constexpr uint32_t pixel_bytes = c_in_p * ELEM_BYTES;
    constexpr int32_t num_in_px = input_height * input_width;
    constexpr uint32_t block_k_cols = block_k_tiles * TILE_DIM;
    constexpr uint32_t im2col_row_bytes = block_k_cols * ELEM_BYTES;
    constexpr uint32_t tap_cols = 3 * c_in_p;
    constexpr bool lanes_padded = c_in < c_in_p;
    constexpr bool k_padded = tensor_kkh_tiles * TILE_DIM > tap_cols;
    // One K-block per kernel row: every K-block puts taps, lanes and K padding at the same row offsets.
    constexpr bool one_k_block_per_kh = block_k_tiles == tensor_kkh_tiles;
    constexpr bool zero_im2col_once = !face_direct && one_k_block_per_kh && (lanes_padded || k_padded);
    constexpr uint32_t c_in_tiles = c_in_p / TILE_DIM;
    static_assert(
        (tensor_kkh_tiles % block_k_tiles) == 0 || (face_direct && block_k_tiles % tensor_kkh_tiles == 0),
        "a K-block never straddles a kernel row, unless it is whole kernel rows (face-direct)");
    static_assert(num_k_blocks * block_k_tiles == 3 * tensor_kkh_tiles, "K-blocks tile the kh blocks exactly");
    static_assert((c_in * ELEM_BYTES) % 16 == 0, "per-tap copies move whole 16 B L1 words");
    static_assert(!tile_input || (c_in_p % TILE_DIM) == 0, "TILE slab rows are whole tile columns");
    static_assert(tile_input || input_page_bytes == pixel_bytes, "ROW_MAJOR: one input page is one slab pixel");
    static_assert(!tile_input || (slab_tile_rows % untilize_batch_rows) == 0, "slab is whole untilize batches");
    static_assert(!channel_slice || input_shard_rows > 0, "a channel slice is read in owner-shard runs");
    static_assert(
        !slab_split || (!tile_input && !channel_slice && input_shard_rows > 0),
        "the slab fill is split by owner shard of ROW_MAJOR HEIGHT pages");
    static_assert(
        !tap_major || (stride == 1 && !lanes_padded && !k_padded && block_k_cols % c_in_p == 0),
        "tap-major rows are whole unpadded taps read in unit-stride pixel runs");

    using FaceGather = conv2d_3x3::FaceGather<
        output_width,
        input_height,
        input_width,
        stride,
        padding,
        dilation,
        c_in_p,
        tensor_kkh_tiles,
        block_k_tiles,
        band_last_tap>;
    constexpr uint32_t FACE_DIM = FaceGather::FACE_DIM;
    constexpr uint32_t in0_tile_bytes = FaceGather::tile_bytes;
    constexpr int32_t face_taps = face_direct ? FaceGather::face_taps : 1;
    constexpr uint32_t kh_per_k_block = face_direct ? FaceGather::kh_per_k_block : 1;
    // A tile column of a kernel row holds two face columns of face_taps taps each; past tap 2 it is K padding.
    constexpr bool padding_only_faces = face_direct && 2 * tensor_kkh_tiles > (3 + face_taps - 1) / face_taps;
    static_assert(
        !face_direct || (!tile_input && !lanes_padded && stride * c_in_p == FACE_DIM &&
                         face_taps * c_in_p == FACE_DIM && (face_taps == 1 || dilation == 1)),
        "face-direct: one output pixel steps the slab by one face row of adjacent, unpadded taps");
    static_assert(face_direct || band_last_tap == 2, "only face-direct reads a tap past the kernel row's last one");
    constexpr uint32_t in0_groups = block_m_tiles / in0_stream_rows;
    static_assert(in0_groups * in0_stream_rows == block_m_tiles, "row groups tile the in0 block");
    static_assert(in0_groups == 1 || (face_direct && num_k_blocks == 1), "only a whole-K face-direct block streams");
    static_assert(!gather_split || (slab_split && in0_groups >= 2), "the writer gathers odd row groups");
    // The writer's row groups of every earlier K-block, for the cumulative gather_done count.
    constexpr uint32_t writer_groups_per_k_block = gather_split ? in0_groups / 2 : 0;

    const auto input = TensorAccessor(input_args, input_addr, input_page_bytes);
    // TILE: compute pushes exactly slab_tile_rows tile-rows per M-block and the reader pops them all,
    // so the slab always starts at the CB base.
    const uint32_t slab_l1 = tile_input ? get_read_ptr(cb_halo_slab) : get_write_ptr(cb_halo_slab);
    const uint64_t slab_noc = get_noc_addr(slab_l1);
    TaggedReads reads;

    if constexpr (zero_im2col_once) {
        const uint32_t im2col_rm_bytes = im2col_rm_depth_rows * block_k_tiles * get_tile_size(cb_im2col_rm);
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_rm), im2col_rm_bytes);
        reads.noc.write_zeros_l1_barrier();
    }
    if constexpr (padding_only_faces) {
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_tiles), block_m_tiles * block_k_tiles * in0_tile_bytes);
        reads.noc.write_zeros_l1_barrier();
    }

    auto zero_fill = [&](uint32_t dst_l1, uint32_t bytes) { reads.zero_fill(dst_l1, bytes); };

    // Input pages [page, end_page) into consecutive L1 page slots from l1. With the split fill, runs of higher
    // shards are the writer's.
    constexpr uint32_t owner_shard_rows = input_shard_rows > 0 ? input_shard_rows : 1;
    auto read_pages = [&](uint32_t page, uint32_t end_page, uint32_t l1) {
        while (page < end_page) {
            // Interleaved runs step by the bank count, not by 1; only unit-stride runs are contiguous here.
            const uint32_t run = input.contiguous_page_stride() == 1 ? input.num_contiguous_pages(page, end_page) : 1;
            if (!slab_split || page / owner_shard_rows <= my_shard) {
                reads.read(input.get_noc_addr(page), l1, run * input_page_bytes);
            }
            page += run;
            l1 += run * input_page_bytes;
        }
    };

    // Input rows [row, end_row) of this core's channels (pixels on ROW_MAJOR, tile-rows on TILE) into
    // consecutive L1 from l1.
    constexpr uint32_t pages_per_row = tile_input ? c_in_tiles : 1;
    constexpr uint32_t row_bytes = pages_per_row * input_page_bytes;
    auto read_rows = [&](uint32_t row, uint32_t end_row, uint32_t l1) {
        if constexpr (channel_slice) {
            while (row < end_row) {
                const uint32_t shard_end = (row / input_shard_rows + 1) * input_shard_rows;
                const uint32_t run_end = end_row < shard_end ? end_row : shard_end;
                reads.read(
                    input.get_noc_addr(row * row_page_stride + slice_page_offset), l1, (run_end - row) * row_bytes);
                l1 += (run_end - row) * row_bytes;
                row = run_end;
            }
        } else {
            read_pages(row * pages_per_row, end_row * pages_per_row, l1);
        }
    };

    // ROW_MAJOR: slab pixels [x_from, x_to) are flat input pixels x + v_minus_x.
    [[maybe_unused]] auto read_slab_pixels = [&](int32_t x_from, int32_t x_to, int32_t v_minus_x) {
        read_rows(x_from + v_minus_x, x_to + v_minus_x, slab_l1 + x_from * pixel_bytes);
    };

    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        const uint32_t block_tiles_done = m_block_idx * block_m_tiles;
        const uint32_t block_valid_tiles =
            (actual_m_tiles - block_tiles_done) < block_m_tiles ? (actual_m_tiles - block_tiles_done) : block_m_tiles;
        const uint32_t o_first = (m_tile_start + block_tiles_done) * TILE_DIM;
        const uint32_t o_end_tiles = o_first + block_valid_tiles * TILE_DIM;
        const uint32_t o_end = o_end_tiles < num_out_px ? o_end_tiles : num_out_px;

        const conv2d_3x3::SlabFootprint footprint =
            conv2d_3x3::block_footprint<output_width, input_width, stride, padding, dilation, band_last_tap>(
                o_first, o_end);
        const int32_t f = footprint.f;
        const int32_t band_len = footprint.band_len;

        int32_t slab_x_base[3];
        [[maybe_unused]] uint32_t slab_ready_pages[3];  // TILE: slab pages that hold band kh once compute pushed them

        if constexpr (tile_input) {
            // ---- fill_halo_slab (TILE): plan the slab rows, then stream input tile-rows in slab order.
            uint32_t seg_first[3];
            uint32_t seg_count[3];
            uint32_t num_segs = 0;
            int32_t next_free = 0;
            bool have_prev = false;
            int32_t prev_t_lo = 0, prev_t_hi = 0, prev_row = 0;
            for (int32_t kh = 0; kh < 3; ++kh) {
                const int32_t v_lo = f + kh * dilation * input_width;
                const int32_t lo = v_lo > 0 ? v_lo : 0;
                const int32_t hi = v_lo + band_len < num_in_px ? v_lo + band_len : num_in_px;
                if (lo >= hi) {
                    slab_x_base[kh] = 0;
                    slab_ready_pages[kh] = next_free * c_in_tiles;
                    continue;
                }
                const int32_t t_lo = lo / static_cast<int32_t>(TILE_DIM);
                const int32_t t_hi = (hi + static_cast<int32_t>(TILE_DIM) - 1) / static_cast<int32_t>(TILE_DIM);
                const bool shares = have_prev && t_lo < prev_t_hi;
                const int32_t row = shares ? prev_row + (t_lo - prev_t_lo) : next_free;
                const int32_t first_new = shares ? prev_t_hi : t_lo;
                if (t_hi > first_new) {
                    seg_first[num_segs] = first_new;
                    seg_count[num_segs] = t_hi - first_new;
                    ++num_segs;
                }
                slab_x_base[kh] = static_cast<int32_t>(TILE_DIM) * (row - t_lo);
                next_free = row + (t_hi - t_lo);
                slab_ready_pages[kh] = next_free * c_in_tiles;
                have_prev = true;
                prev_t_lo = t_lo;
                prev_t_hi = t_hi;
                prev_row = row;
            }

            // Slab rows past the plan's last row are pushed unread: compute untilizes a fixed count.
            constexpr uint32_t batch_pages = untilize_batch_rows * c_in_tiles;
            uint32_t seg = 0;
            uint32_t seg_row = 0;
            for (uint32_t batch = 0; batch < slab_tile_rows / untilize_batch_rows; ++batch) {
                cb_reserve_back(cb_input_tiles, batch_pages);
                uint32_t l1 = get_write_ptr(cb_input_tiles);
                reads.begin(TRID_INPUT_TILES);
                for (uint32_t r = 0; r < untilize_batch_rows && seg < num_segs;) {
                    // The batch's rows of one segment are consecutive input tile-rows: consecutive pages.
                    const uint32_t seg_left = seg_count[seg] - seg_row;
                    const uint32_t rows = untilize_batch_rows - r < seg_left ? untilize_batch_rows - r : seg_left;
                    const uint32_t row0 = seg_first[seg] + seg_row;
                    read_rows(row0, row0 + rows, l1);
                    l1 += rows * row_bytes;
                    r += rows;
                    seg_row += rows;
                    if (seg_row == seg_count[seg]) {
                        ++seg;
                        seg_row = 0;
                    }
                }
                reads.wait(TRID_INPUT_TILES);
                cb_push_back(cb_input_tiles, batch_pages);
            }
        } else {
            // ---- fill_halo_slab (ROW_MAJOR): band kh tagged trid kh + 1; overlapping bands read only their
            // new tail.
            if constexpr (slab_split) {
                if (m_block_idx > 0) {
                    // The writer refills the slab only after this block's gather no longer reads it.
                    Semaphore<>(slab_free_sem_id).up(1);
                }
            }
            conv2d_3x3::slab_bands<input_width, num_in_px, dilation, band_stride_px>(
                footprint, slab_x_base, [&](int32_t kh, int32_t x_lo, int32_t x_hi, int32_t v_minus_x) {
                    reads.begin(kh + 1);
                    if (x_lo < x_hi) {
                        // Each core starts its band at a different pixel so the cores do not all open on one bank.
                        const int32_t x_split =
                            x_lo + static_cast<int32_t>(read_rotation % static_cast<uint32_t>(x_hi - x_lo));
                        read_slab_pixels(x_split, x_hi, v_minus_x);
                        read_slab_pixels(x_lo, x_split, v_minus_x);
                    }
                });
            if constexpr (slab_split) {
                Semaphore<>(slab_fwd_done_sem_id).wait_min(m_block_idx + 1);
            }
        }

        // ---- gather_im2col_blocks: one K-block = block_m_tiles tile-rows x block_k_cols of kernel row kh.
        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            const uint32_t k_tile0 = k_block_idx * block_k_tiles;
            const int32_t kh = k_tile0 / tensor_kkh_tiles;
            const uint32_t col_lo = (k_tile0 - kh * tensor_kkh_tiles) * TILE_DIM;
            const uint32_t col_hi = col_lo + block_k_cols;
            if (col_lo == 0) {
                for (int32_t band = kh; band < kh + static_cast<int32_t>(kh_per_k_block); ++band) {
                    if constexpr (tile_input) {
                        if (slab_ready_pages[band] > 0) {
                            cb_wait_front(cb_halo_slab, slab_ready_pages[band]);
                        }
                    } else {
                        reads.wait(band + 1);
                    }
                }
            }

            if constexpr (face_direct) {
                cb_reserve_back(cb_im2col_tiles, block_m_tiles * block_k_tiles);
                const uint32_t tiles_l1 = get_write_ptr(cb_im2col_tiles);
                if constexpr (gather_split) {
                    Semaphore<>(gather_go_sem_id).up(1);
                }
                const uint32_t writer_groups_before =
                    (m_block_idx * num_k_blocks + k_block_idx) * writer_groups_per_k_block;
                // Push row groups [pushed, end_group) in order, a writer's group once the writer counted it.
                uint32_t pushed = 0;
                auto push_through = [&](uint32_t end_group) {
                    for (; pushed < end_group; ++pushed) {
                        if (gather_split && pushed % 2 == 1) {
                            Semaphore<>(gather_done_sem_id).wait_min(writer_groups_before + (pushed + 1) / 2);
                        }
                        cb_push_back(cb_im2col_tiles, in0_stream_rows * block_k_tiles);
                    }
                };
                for (uint32_t group = 0; group < in0_groups; group += gather_split ? 2 : 1) {
                    reads.begin(TRID_GATHER);
                    FaceGather::tile_rows(
                        reads,
                        tiles_l1,
                        group * in0_stream_rows,
                        (group + 1) * in0_stream_rows,
                        o_first,
                        o_end,
                        kh,
                        slab_l1,
                        slab_x_base);
                    reads.wait(TRID_GATHER);
                    push_through(group + 1);
                }
                push_through(in0_groups);
                continue;
            }
            const uint32_t data_hi = col_hi < tap_cols ? col_hi : tap_cols;
            const int32_t kw_first = col_lo < tap_cols ? col_lo / c_in_p : 3;
            const int32_t kw_last = data_hi > col_lo ? (data_hi - 1) / c_in_p : -1;
            const int32_t slab_x_base_kh = slab_x_base[kh];

            uint32_t o = o_first;
            for (uint32_t tile_row = 0; tile_row < block_m_tiles; ++tile_row) {
                cb_reserve_back(cb_im2col_rm, block_k_tiles);
                const uint32_t rows_l1 = get_write_ptr(cb_im2col_rm);
                reads.begin(TRID_GATHER);

                if constexpr (tap_major) {
                    // Tap kw of this tile-row is a [32 pixels, c_in_p] block. At stride 1 the pixels of one
                    // output image row read consecutive slab pixels, so each such run is one read.
                    const uint32_t row_end = o + TILE_DIM < o_end ? o + TILE_DIM : o_end;
                    for (int32_t kw = kw_first; kw <= kw_last; ++kw) {
                        uint32_t dst = rows_l1 + (kw - kw_first) * TILE_DIM * pixel_bytes;
                        int32_t ho = o / output_width;
                        int32_t wo = o % output_width;
                        for (uint32_t p = o; p < row_end;) {
                            const int32_t run = row_end - p < output_width - wo ? row_end - p : output_width - wo;
                            const int32_t h_in = ho + kh * dilation - padding;
                            const int32_t w_lo = wo - padding + kw * dilation;
                            // Pixels [in_lo, in_hi) of the run are inside the image row; the rest are padding.
                            int32_t in_lo = w_lo < 0 ? (-w_lo < run ? -w_lo : run) : 0;
                            int32_t in_hi = input_width - w_lo < run ? input_width - w_lo : run;
                            if (h_in < 0 || h_in >= input_height || in_hi < in_lo) {
                                in_lo = in_hi = run;
                            }
                            if (in_lo > 0) {
                                zero_fill(dst, in_lo * pixel_bytes);
                            }
                            if (in_hi > in_lo) {
                                reads.read(
                                    slab_noc + (slab_x_base_kh + h_in * input_width + w_lo + in_lo) * pixel_bytes,
                                    dst + in_lo * pixel_bytes,
                                    (in_hi - in_lo) * pixel_bytes);
                            }
                            if (run > in_hi) {
                                zero_fill(dst + in_hi * pixel_bytes, (run - in_hi) * pixel_bytes);
                            }
                            dst += run * pixel_bytes;
                            p += run;
                            wo = 0;
                            ++ho;
                        }
                    }
                    o = row_end;
                } else {
                    int32_t ho = o / output_width;
                    int32_t wo = o % output_width;
                    for (uint32_t r = 0; r < TILE_DIM && o < o_end; ++r, ++o) {
                        const uint32_t row_l1 = rows_l1 + r * im2col_row_bytes;
                        const int32_t h_in = ho * stride + kh * dilation - padding;
                        const int32_t w_in0 = wo * stride - padding;
                        const bool row_in_image = h_in >= 0 && h_in < input_height;
                        const int32_t slab_x0 = slab_x_base_kh + h_in * input_width + w_in0;

                        const bool combined = !lanes_padded && dilation == 1 && row_in_image && w_in0 + kw_first >= 0 &&
                                              w_in0 + kw_last < input_width && kw_first <= kw_last;
                        if (combined) {
                            // d = 1, no lanes: the touched taps are consecutive slab pixels, so one read covers them.
                            reads.read(
                                slab_noc + slab_x0 * pixel_bytes + col_lo * ELEM_BYTES,
                                row_l1,
                                (data_hi - col_lo) * ELEM_BYTES);
                        } else {
                            for (int32_t kw = kw_first; kw <= kw_last; ++kw) {
                                const uint32_t tap_lo = kw * c_in_p;
                                const uint32_t seg_lo = col_lo > tap_lo ? col_lo : tap_lo;
                                const uint32_t seg_hi = data_hi < tap_lo + c_in_p ? data_hi : tap_lo + c_in_p;
                                const uint32_t real_hi = seg_hi < tap_lo + c_in ? seg_hi : tap_lo + c_in;
                                const uint32_t dst = row_l1 + (seg_lo - col_lo) * ELEM_BYTES;
                                const int32_t w_in = w_in0 + kw * dilation;
                                if (row_in_image && w_in >= 0 && w_in < input_width) {
                                    const int32_t slab_x = slab_x0 + kw * dilation;
                                    if (real_hi > seg_lo) {
                                        reads.read(
                                            slab_noc + slab_x * pixel_bytes + (seg_lo - tap_lo) * ELEM_BYTES,
                                            dst,
                                            (real_hi - seg_lo) * ELEM_BYTES);
                                    }
                                    if constexpr (lanes_padded && !zero_im2col_once) {
                                        const uint32_t lane_lo = real_hi > seg_lo ? real_hi : seg_lo;
                                        if (seg_hi > lane_lo) {
                                            zero_fill(
                                                row_l1 + (lane_lo - col_lo) * ELEM_BYTES,
                                                (seg_hi - lane_lo) * ELEM_BYTES);
                                        }
                                    }
                                } else {
                                    zero_fill(dst, (seg_hi - seg_lo) * ELEM_BYTES);
                                }
                            }
                        }
                        if constexpr (!zero_im2col_once) {
                            if (col_hi > tap_cols) {
                                const uint32_t pad_lo = col_lo > tap_cols ? col_lo : tap_cols;
                                zero_fill(row_l1 + (pad_lo - col_lo) * ELEM_BYTES, (col_hi - pad_lo) * ELEM_BYTES);
                            }
                        }

                        if (++wo == static_cast<int32_t>(output_width)) {
                            wo = 0;
                            ++ho;
                        }
                    }
                }
                // Rows past the block's last valid pixel keep stale bytes: every phase downstream is
                // row-independent and the writer never stores them.
                reads.wait(TRID_GATHER);
                cb_push_back(cb_im2col_rm, block_k_tiles);
            }
        }

        if constexpr (tile_input) {
            constexpr uint32_t slab_pages = slab_tile_rows * c_in_tiles;
            cb_wait_front(cb_halo_slab, slab_pages);
            cb_pop_front(cb_halo_slab, slab_pages);
        }

        if constexpr (combine_send) {
            constexpr uint32_t num_owning_cols = (tensor_n_tiles + out_n_tiles - 1) / out_n_tiles;
            static_assert(num_owning_cols <= k_cols);
            Semaphore<> send_go(combine_send_go_sem_id);
            Semaphore<> send_done(combine_send_done_sem_id);
            send_go.wait_min(m_block_idx + 1);
            conv2d_3x3::scatter_partial_slices<block_m_tiles, block_n_tiles, out_n_tiles, tensor_n_tiles>(
                get_read_ptr(cb_partials),
                get_write_ptr(cb_combine_slots),
                get_tile_size(cb_partials),
                my_col,
                my_col + 1,
                num_owning_cols,
                row_peer_coords_arg);
            noc_async_write_barrier();
            send_done.up(1);
        }
    }
    dataflow_kernel_lib::set_read_trid(reads.noc, 0);
}
