// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 reader (NCRISC / NoC0): per M-block, fill_halo_slab then gather_im2col_blocks.
// (perf experiment copy: split_row_gather)
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
// them in row groups of in0_stream_rows tile-rows (the last may be shorter), in order, so compute multiplies each group
// as it lands. With a split gather it raises gather_go once the block is reserved and the slab complete; the writer
// then gathers every odd row group and counts them in gather_done, while the reader gathers the even ones. Only the
// reader touches the CB's pointers; the writer writes into the reserved block, which starts at the CB base.
//
// Split row gather (perf experiment split_row_gather; per-pixel row gather only, block_m_tiles > 1). The writer gathers
// some tile-rows of cb_im2col_rm on NoC1 while the reader gathers the rest; the reader alone moves the CB's pointers.
// Tile-row s of the kernel's sequence (M-block, K-block, tile-row) lives in slot s % depth (every push is one
// tile-row and the CB holds exactly `depth` of them), so a row is placed by index, not by the write pointer.
//   Dynamic (the recommended mode): at each of its rows the reader reserves rows [rows_pushed, s] together, takes s,
//   and hands the writer the following rows of the same K-block while the writer has fewer than row_queue
//   outstanding and has published (row_limit) that it serves them: their weights are issued and their M-block
//   planned. A hand-out is a mailbox word (the row index) plus row_go; row_assigned tells the writer how far the
//   assignment got. The reader pushes in sequence order, waiting on row_done for writer rows. So a writer busy with a
//   weight round or its store gets no rows, and an idle one takes about every other row.
//   Static: the writer owns rows s % period >= period - writer_rows, granted grant-ahead rows early.
// No helper covers this either: the CB API has no "reserve for another RISC", hence the index-addressed slots.
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
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_combine_dataflow.hpp"
#include "conv2d_3x3_slab_dataflow.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t ELEM_BYTES = 2;        // bf16 slab and im2col rows
constexpr uint32_t TRID_GATHER = 4;       // ROW_MAJOR bands use trids 1..3
constexpr uint32_t TRID_INPUT_TILES = 5;  // TILE footprint batches

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
    // Split row gather (see the file comment): writer rows of the tile-row sequence, grant lookahead, handshake.
    constexpr bool row_split = get_compile_time_arg_val(45) != 0;
    constexpr uint32_t row_split_period = get_compile_time_arg_val(46);
    constexpr uint32_t row_split_writer_rows = get_compile_time_arg_val(47);
    constexpr uint32_t row_grant_ahead = get_compile_time_arg_val(48);
    constexpr uint32_t row_go_sem_id = get_compile_time_arg_val(49);
    constexpr uint32_t row_done_sem_id = get_compile_time_arg_val(50);
    // Dynamic split (row_dynamic): the reader hands the writer the next rows only while the writer has fewer than
    // row_queue outstanding and has published (row_limit) that it serves them; row_assigned tells the writer how far
    // the assignment got, and the mailbox words carry the row index of each grant.
    // 1: hand-outs stay inside the reader's current K-block; 2: anywhere in the M-block (the writer then also takes
    // rows of later K-blocks, e.g. with one tile-row per K-block).
    constexpr uint32_t row_dynamic_mode = get_compile_time_arg_val(51);
    constexpr bool row_dynamic = row_dynamic_mode != 0;
    constexpr uint32_t row_queue = get_compile_time_arg_val(52);
    constexpr uint32_t row_limit_sem_id = get_compile_time_arg_val(53);
    constexpr uint32_t row_assigned_sem_id = get_compile_time_arg_val(54);
    constexpr uint32_t row_mailbox_sem_id = get_compile_time_arg_val(55);  // row_queue consecutive ids
    constexpr auto input_args = TensorAccessorArgs<56>();

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
    constexpr uint32_t in0_groups = (block_m_tiles + in0_stream_rows - 1) / in0_stream_rows;
    static_assert(in0_groups == 1 || (face_direct && num_k_blocks == 1), "only a whole-K face-direct block streams");
    static_assert(!gather_split || (slab_split && in0_groups >= 2), "the writer gathers odd row groups");
    // The writer's row groups of every earlier K-block, for the cumulative gather_done count.
    constexpr uint32_t writer_groups_per_k_block = gather_split ? in0_groups / 2 : 0;

    const auto input = TensorAccessor(input_args, input_addr, input_page_bytes);
    // TILE: compute pushes exactly slab_tile_rows tile-rows per M-block and the reader pops them all,
    // so the slab always starts at the CB base.
    const uint32_t slab_l1 = tile_input ? get_read_ptr(cb_halo_slab) : get_write_ptr(cb_halo_slab);
    const uint64_t slab_noc = get_noc_addr(slab_l1);
    conv2d_3x3::TaggedReads<face_direct> reads;

    if constexpr (zero_im2col_once) {
        const uint32_t im2col_rm_bytes = im2col_rm_depth_rows * block_k_tiles * get_tile_size(cb_im2col_rm);
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_rm), im2col_rm_bytes);
        reads.noc.write_zeros_l1_barrier();
    }
    if constexpr (padding_only_faces) {
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_tiles), block_m_tiles * block_k_tiles * in0_tile_bytes);
        reads.noc.write_zeros_l1_barrier();
    }

    const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE);
    auto zero_fill = [&](uint32_t dst_l1, uint32_t bytes) {
        while (bytes > 0) {
            const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
            reads.read(zeros_noc, dst_l1, chunk);
            dst_l1 += chunk;
            bytes -= chunk;
        }
    };

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

    using RowGatherT = conv2d_3x3::RowGather<
        output_width,
        input_height,
        input_width,
        stride,
        padding,
        dilation,
        c_in,
        c_in_p,
        block_k_tiles,
        zero_im2col_once>;
    using RowSplitT = conv2d_3x3::RowSplit<row_split_period, row_split ? row_split_writer_rows : 0>;
    static_assert(!row_split || (!face_direct && !tap_major), "the split row gather is the per-pixel row gather");
    static_assert(!row_dynamic || (row_split && row_queue > 0), "the dynamic split is a row split");
    [[maybe_unused]] const uint32_t im2col_rm_base = row_split ? get_write_ptr(cb_im2col_rm) : 0;
    [[maybe_unused]] const uint32_t im2col_rm_slot_bytes = row_split ? block_k_tiles * get_tile_size(cb_im2col_rm) : 0;
    [[maybe_unused]] uint32_t rows_pushed = 0;      // tile-rows of the sequence pushed to cb_im2col_rm
    [[maybe_unused]] uint32_t rows_granted_to = 0;  // every writer row below this one was handed to the writer
    // Dynamic split: next unassigned row, rows given to the writer, and per CB slot the writer grant (index + 1) of
    // the row in it, 0 for a reader row.
    [[maybe_unused]] uint32_t assign_cursor = 0;
    [[maybe_unused]] uint32_t writer_assigned = 0;
    [[maybe_unused]] uint32_t slot_grant[row_dynamic ? im2col_rm_depth_rows : 1];

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
            MaybeDeviceZoneScope("reader_fill");
            conv2d_3x3::TileSlabPlan plan;
            conv2d_3x3::tile_slab_plan<input_width, num_in_px, dilation, c_in_tiles>(footprint, slab_x_base, plan);
            const uint32_t num_segs = plan.num_segs;
            const uint32_t(&seg_first)[3] = plan.seg_first;
            const uint32_t(&seg_count)[3] = plan.seg_count;
            for (int32_t kh = 0; kh < 3; ++kh) {
                slab_ready_pages[kh] = plan.ready_pages[kh];
            }

            // Slab rows past the plan's last row are pushed unread: compute untilizes a fixed count.
            constexpr uint32_t batch_pages = untilize_batch_rows * c_in_tiles;
            uint32_t seg = 0;
            uint32_t seg_row = 0;
            for (uint32_t batch = 0; batch < slab_tile_rows / untilize_batch_rows; ++batch) {
                {
                    // Back-pressure from compute's untilize.
                    MaybeDeviceZoneScope("reader_fill_reserve");
                    cb_reserve_back(cb_input_tiles, batch_pages);
                }
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
            MaybeDeviceZoneScope("reader_fill");
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
                MaybeDeviceZoneScope("reader_fill_fwd_wait");
                Semaphore<>(slab_fwd_done_sem_id).wait_min(m_block_idx + 1);
            }
        }

        // The band's slab rows: compute's untilize (TILE) or this core's own band reads (ROW_MAJOR). Bands are
        // waited in order, each once per M-block; the split row gather may wait a band early, to grant a writer
        // row of the next kernel row.
        int32_t bands_ready = -1;
        auto wait_bands = [&](int32_t last_band) {
            if (last_band <= bands_ready) {
                return;
            }
            MaybeDeviceZoneScope("reader_band_wait");
            for (int32_t band = bands_ready + 1; band <= last_band; ++band) {
                if constexpr (tile_input) {
                    if (slab_ready_pages[band] > 0) {
                        cb_wait_front(cb_halo_slab, slab_ready_pages[band]);
                    }
                } else {
                    reads.wait(band + 1);
                }
            }
            bands_ready = last_band;
        };

        // ---- Split row gather. The reader alone moves cb_im2col_rm's pointers: it reserves a writer row's slot
        // (row s lives at slot s % im2col_rm_depth_rows: every push is one tile-row and the CB holds exactly
        // im2col_rm_depth_rows of them), raises row_go, and pushes rows in sequence order, a writer row once
        // row_done counted it. Rows [rows_pushed, s] are reserved together, so a grant needs them to fit.
        const uint32_t m_seq_end = (m_block_idx + 1) * num_k_blocks * block_m_tiles;
        [[maybe_unused]] auto grant_through = [&](uint32_t limit) {
            for (; rows_granted_to < limit; ++rows_granted_to) {
                const uint32_t w = rows_granted_to;
                if (!RowSplitT::writer_owns(w)) {
                    continue;
                }
                if (w + 1 - rows_pushed > im2col_rm_depth_rows) {
                    return;
                }
                wait_bands(((w / block_m_tiles) % num_k_blocks) * block_k_tiles / tensor_kkh_tiles);
                cb_reserve_back(cb_im2col_rm, (w + 1 - rows_pushed) * block_k_tiles);
                Semaphore<>(row_go_sem_id).up(1);
            }
        };
        // Push rows [rows_pushed, limit) in order (blocking on the writer's rows).
        [[maybe_unused]] auto push_through = [&](uint32_t limit) {
            for (; rows_pushed < limit;) {
                if (RowSplitT::writer_owns(rows_pushed)) {
                    grant_through(rows_pushed + 1);
                    Semaphore<>(row_done_sem_id).wait_min(RowSplitT::writer_rows_before(rows_pushed) + 1);
                }
                cb_push_back(cb_im2col_rm, block_k_tiles);
                ++rows_pushed;
            }
        };

        // Dynamic split: push the oldest row (a writer row once the writer counted it).
        [[maybe_unused]] auto push_one_dyn = [&]() {
            const uint32_t grant = slot_grant[rows_pushed % im2col_rm_depth_rows];
            if (grant > 0) {
                Semaphore<>(row_done_sem_id).wait_min(grant);
            }
            cb_push_back(cb_im2col_rm, block_k_tiles);
            ++rows_pushed;
        };
        // Hand the writer the next rows below `limit` while it has room in its queue, serves them, and they fit.
        [[maybe_unused]] auto top_up_writer = [&](uint32_t limit) {
            while (assign_cursor < limit && assign_cursor + 1 - rows_pushed <= im2col_rm_depth_rows &&
                   writer_assigned - Semaphore<>(row_done_sem_id).value() < row_queue &&
                   assign_cursor < Semaphore<>(row_limit_sem_id).value()) {
                const uint32_t r = assign_cursor;
                if constexpr (row_dynamic_mode == 2) {
                    // A later K-block's row: its band must be in the slab before the writer reads it.
                    wait_bands(((r / block_m_tiles) % num_k_blocks) * block_k_tiles / tensor_kkh_tiles);
                }
                cb_reserve_back(cb_im2col_rm, (r + 1 - rows_pushed) * block_k_tiles);
                slot_grant[r % im2col_rm_depth_rows] = ++writer_assigned;
                Semaphore<>(row_mailbox_sem_id + (writer_assigned - 1) % row_queue).set(r);
                Semaphore<>(row_go_sem_id).up(1);
                ++assign_cursor;
            }
        };

        // ---- gather_im2col_blocks: one K-block = block_m_tiles tile-rows x block_k_cols of kernel row kh.
        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            const uint32_t k_tile0 = k_block_idx * block_k_tiles;
            const int32_t kh = k_tile0 / tensor_kkh_tiles;
            const uint32_t col_lo = (k_tile0 - kh * tensor_kkh_tiles) * TILE_DIM;
            const uint32_t col_hi = col_lo + block_k_cols;
            if (col_lo == 0) {
                wait_bands(kh + static_cast<int32_t>(kh_per_k_block) - 1);
            }

            // Occupancy: includes the per-tile-row cb_reserve_back on compute's tilize / matmul.
            MaybeDeviceZoneScope("reader_gather");
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
                        const uint32_t row_end = (pushed + 1) * in0_stream_rows;
                        const uint32_t rows =
                            (row_end < block_m_tiles ? row_end : block_m_tiles) - pushed * in0_stream_rows;
                        cb_push_back(cb_im2col_tiles, rows * block_k_tiles);
                    }
                };
                for (uint32_t group = 0; group < in0_groups; group += gather_split ? 2 : 1) {
                    reads.begin(TRID_GATHER);
                    FaceGather::tile_rows(
                        reads,
                        tiles_l1,
                        group * in0_stream_rows,
                        (group + 1) * in0_stream_rows < block_m_tiles ? (group + 1) * in0_stream_rows : block_m_tiles,
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
            const uint32_t kb_seq0 = (m_block_idx * num_k_blocks + k_block_idx) * block_m_tiles;

            if constexpr (row_dynamic) {
                const uint32_t kb_end = kb_seq0 + block_m_tiles;
                while (assign_cursor < kb_end) {
                    const uint32_t s = assign_cursor;
                    while (s + 1 - rows_pushed > im2col_rm_depth_rows) {
                        push_one_dyn();
                    }
                    cb_reserve_back(cb_im2col_rm, (s + 1 - rows_pushed) * block_k_tiles);
                    slot_grant[s % im2col_rm_depth_rows] = 0;
                    ++assign_cursor;
                    top_up_writer(row_dynamic_mode == 2 ? m_seq_end : kb_end);
                    Semaphore<>(row_assigned_sem_id).set(assign_cursor);
                    reads.begin(TRID_GATHER);
                    RowGatherT::tile_row(
                        reads,
                        slab_noc,
                        im2col_rm_base + (s % im2col_rm_depth_rows) * im2col_rm_slot_bytes,
                        o_first + (s - kb_seq0) * TILE_DIM,
                        o_end,
                        kh,
                        col_lo,
                        slab_x_base_kh);
                    reads.wait(TRID_GATHER);
                    while (rows_pushed <= s) {
                        push_one_dyn();
                    }
                }
                continue;
            }

            uint32_t o = o_first;
            for (uint32_t tile_row = 0; tile_row < block_m_tiles; ++tile_row) {
                uint32_t rows_l1;
                [[maybe_unused]] const uint32_t s = kb_seq0 + tile_row;
                if constexpr (row_split) {
                    if (RowSplitT::writer_owns(s)) {
                        continue;
                    }
                    // Rows [rows_pushed, s] must fit: push (waiting on the writer) until they do, then hand the
                    // writer its rows up to row_grant_ahead past this one, within the M-block.
                    while (s + 1 - rows_pushed > im2col_rm_depth_rows) {
                        push_through(rows_pushed + 1);
                    }
                    grant_through(s + 1 + row_grant_ahead < m_seq_end ? s + 1 + row_grant_ahead : m_seq_end);
                    cb_reserve_back(cb_im2col_rm, (s + 1 - rows_pushed) * block_k_tiles);
                    rows_l1 = im2col_rm_base + (s % im2col_rm_depth_rows) * im2col_rm_slot_bytes;
                } else {
                    cb_reserve_back(cb_im2col_rm, block_k_tiles);
                    rows_l1 = get_write_ptr(cb_im2col_rm);
                }
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
                    RowGatherT::tile_row(
                        reads, slab_noc, rows_l1, o_first + tile_row * TILE_DIM, o_end, kh, col_lo, slab_x_base_kh);
                }
                // Rows past the block's last valid pixel keep stale bytes: every phase downstream is
                // row-independent and the writer never stores them.
                reads.wait(TRID_GATHER);
                if constexpr (row_split) {
                    push_through(s + 1);
                } else {
                    cb_push_back(cb_im2col_rm, block_k_tiles);
                }
            }
        }
        if constexpr (row_dynamic) {
            // The writer's trailing rows: compute needs them, and the writer must be done reading the slab.
            while (rows_pushed < m_seq_end) {
                push_one_dyn();
            }
        } else if constexpr (row_split) {
            push_through(m_seq_end);
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
            {
                MaybeDeviceZoneScope("reader_combine_wait_go");
                send_go.wait_min(m_block_idx + 1);
            }
            MaybeDeviceZoneScope("reader_combine_send");
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
