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
// N-slice-outer K split (see the writer's file comment): the reader gathers tile-row outer, K-block inner, so compute
// tilizes in0 row-major over all of K, and it sends the +d owner slices on NoC0 from cb_slice_r, which it alone
// consumes.
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
// Split TILE fill (HEIGHT-sharded TILE input, on the weight multicast's pure receivers): the same NoC split for the
// tile-row fill. The reader reads its own and lower shards' tile-rows into each cb_input_tiles batch it reserved; a
// batch that holds higher shards' tile-rows is handed to the writer first (fill_go) and pushed once the writer's rows
// landed (fill_done). Only the reader moves the CB's pointers.
//
// Face-direct im2col (see conv2d_3x3_gather_dataflow.hpp): the reader reserves the K-block's in0 tiles and pushes
// them in row groups of in0_stream_rows tile-rows (the last may be shorter), in order, so compute multiplies each group
// as it lands. With a split gather it raises gather_go once the block is reserved and the slab complete; the writer
// then gathers every odd row group and counts them in gather_done, while the reader gathers the even ones. Only the
// reader touches the CB's pointers; the writer writes into the reserved block, which starts at the CB base.
//
// Split row gather (the per-pixel row gather, more than one tile-row per K-block): the writer gathers some units
// (one tile-row of one K-block) on NoC1 while the reader gathers the rest. At each unit it takes, the reader hands the
// writer the following units while the writer has fewer than row_queue outstanding and has published (row_limit)
// that it serves them; a hand-out is a mailbox word (the unit index) plus row_go, and row_assigned tells the writer
// how far the assignment got. So a writer busy with a weight round gets no units, and an idle one about every other.
// The CB API cannot reserve on behalf of another RISC, hence the index-addressed slots. Hand-outs stay inside the
// reader's current K-block, whose band it has waited for; under the N-slice-outer order every band is waited first.
//
// Strided window (stride 2, see conv2d_3x3_gather_dataflow.hpp WindowGather): a unit is one kernel row's run of input
// pixels, one read per output-row run, which compute tilizes in place at a 2-pixel row stride; the reader alone gathers
// (no split). Every unit's last 16 B carry the L1 address compute tilizes it from. On the N-slice-outer walk
// (window_direct) a full in-image run is not copied at all: its address is the slab run itself, so the slab must
// outlive compute's tilize, and the reader waits for compute to pop every unit before it frees the slab (slab pop,
// slab_dead).
//
// Short last channel slice (K split, C_in not a multiple of the slice width): tap-major copies whole slab pixels, so
// the slice's padded lanes must read as zero. They are zeroed once at kernel start and never read into: on TILE the
// cb_input_tiles slots of the padding tile columns (the ring is whole batches, so each slot keeps its L1 offset),
// on ROW_MAJOR the whole slab, after which every pixel fill reads only its real channels.
//
// Center-tap direct in0 (the descriptor's CENTER_TAP_DIRECT; sharded TILE input, stride 1, padding == dilation,
// W % 32 == 0): output tile-row t's kw = 1 tap of kernel row kh is input tile-row t + (kh*d - p) * W / 32, or a zero
// tile-row outside the image, already in in0's tile format. The reader copies those tiles from the own or a neighbour
// shard (one accessor lookup per in-shard run) instead of gathering the tap, and compute does not tilize it.
//   center_mode 1 (one tap per K-block): K-blocks k % 3 == 1 go whole into cb_center_tiles (compute's In0SourceFn).
//   center_mode 2 (one kernel row per K-block): each tile-row's kw = 1 tiles go straight into their cb_im2col_tiles
//     slots; the tap-major gather leaves that tap's block of the im2col row empty, and compute's tilize reserves and
//     pushes the slots unpacked. The copy (own transaction id) lands before the K-block's last im2col row is pushed.
//     cb_im2col_tiles holds two K-blocks, so K-block k's slots (k % 2) are free once the reader can reserve K-block k's
//     first im2col row: compute then began K-block k - 1, so the matmul of K-block k - 2 is done.
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
#include "conv2d_3x3_nslice_order.hpp"

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t ELEM_BYTES = 2;        // bf16 slab and im2col rows
constexpr uint32_t TRID_GATHER = 4;       // ROW_MAJOR bands use trids 1..3
constexpr uint32_t TRID_INPUT_TILES = 5;  // TILE footprint batches
constexpr uint32_t TRID_CENTER = 6;       // center-tap slots

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
    constexpr uint32_t lane_zero_rows = get_compile_time_arg_val(45);
    constexpr uint32_t cb_zero_src = get_compile_time_arg_val(46);
    // N-slice-outer K split (see conv2d_3x3_nslice_order.hpp): the reader sends the +d owner slices on NoC0.
    constexpr bool nslice_outer = get_compile_time_arg_val(47) != 0;
    constexpr uint32_t cb_slice_r = get_compile_time_arg_val(48);
    constexpr uint32_t combine_credit_sem_id = get_compile_time_arg_val(49);
    constexpr uint32_t combine_arrival_sem_id = get_compile_time_arg_val(50);
    // Split row gather (see the file comment).
    constexpr bool row_split = get_compile_time_arg_val(51) != 0;
    constexpr uint32_t row_queue = get_compile_time_arg_val(52);
    constexpr uint32_t row_go_sem_id = get_compile_time_arg_val(53);
    constexpr uint32_t row_done_sem_id = get_compile_time_arg_val(54);
    constexpr uint32_t row_limit_sem_id = get_compile_time_arg_val(55);
    constexpr uint32_t row_assigned_sem_id = get_compile_time_arg_val(56);
    constexpr uint32_t row_mailbox_sem_id = get_compile_time_arg_val(57);  // row_queue consecutive ids
    // Shared-L1 N-slice walk: cb_combine_slots shares the slab's L1, and slab_dead tells the writer the gather landed.
    constexpr bool alias_slab_combine = get_compile_time_arg_val(58) != 0;
    constexpr uint32_t slab_dead_sem_id = get_compile_time_arg_val(59);
    // Split TILE fill (see the file comment): the writer reads the higher shards' tile-rows. Its two counters reuse the
    // split slab fill's ids: fill_done (writer -> reader) and fill_go (reader -> writer).
    constexpr bool tile_fill_split = get_compile_time_arg_val(60) != 0;
    // Strided window (see the file comment) and its slab-direct units.
    constexpr bool strided_window = get_compile_time_arg_val(61) != 0;
    constexpr bool window_direct = get_compile_time_arg_val(62) != 0;
    // Center-tap direct in0 (see the descriptor's CENTER_TAP_DIRECT): the kw = 1 K-blocks are input tile-rows.
    // 1: one tap per K-block, the kw = 1 K-blocks go to cb_center_tiles. 2: one kernel row per K-block, the kw = 1
    // tiles of every tile-row go straight into their cb_im2col_tiles slots.
    constexpr uint32_t center_mode = get_compile_time_arg_val(63);
    constexpr uint32_t cb_center_tiles = get_compile_time_arg_val(64);
    constexpr bool center_direct = center_mode != 0;
    // Ring slab (the descriptor's ring_capable): cb_halo_slab is a FIFO ring of
    // slab_tile_rows consecutive input tile-rows across M-blocks. Each M-block streams only the rows past the previous
    // block's window and pops only the rows below its own window; every band gathers at one base and wraps.
    constexpr bool ring_slab = get_compile_time_arg_val(65) != 0;
    // Tap-direct N-slice prelude (the descriptor's _tap_direct; TapDirect in conv2d_3x3_gather_dataflow.hpp): per
    // cb_im2col_rm slot, cb_src_table holds one 16 B entry of tap source addresses that compute tilizes from.
    constexpr bool tap_direct = get_compile_time_arg_val(66) != 0;
    constexpr uint32_t cb_src_table = get_compile_time_arg_val(67);
    // TILE fill pipeline (block_fill_pipeline): cb_input_tiles holds input_ring_batches batches, and the reader keeps
    // up to fill_window of them in flight, each under its own transaction id (TRID_FILL_BASE + batch % fill_window).
    constexpr uint32_t input_ring_batches = get_compile_time_arg_val(68);
    constexpr uint32_t fill_window = get_compile_time_arg_val(69);
    constexpr auto input_args = TensorAccessorArgs<70>();
    static_assert(!alias_slab_combine || nslice_outer, "only the N-slice-outer walk aliases the slab");

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t actual_m_tiles = get_arg_val<uint32_t>(2);
    const uint32_t num_m_blocks_this_core = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t read_rotation = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t slice_page_offset = get_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint32_t my_col = get_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t my_shard = get_arg_val<uint32_t>(7);
    constexpr uint32_t row_peer_coords_arg = 8;  // (x, y) of grid columns 0..k_cols-1 of this row
    // Split TILE fill on this core (runtime: the kernel is shared with the cores that keep the reader-only fill).
    constexpr uint32_t split_flag_arg = row_peer_coords_arg + (channel_slice ? 2 * k_cols : 0);
    [[maybe_unused]] const bool split_here = tile_fill_split && get_arg_val<uint32_t>(split_flag_arg) != 0;

    constexpr uint32_t pixel_bytes = c_in_p * ELEM_BYTES;
    constexpr int32_t num_in_px = input_height * input_width;
    constexpr uint32_t block_k_cols = block_k_tiles * TILE_DIM;
    constexpr uint32_t tap_cols = 3 * c_in_p;
    constexpr bool lanes_padded = c_in < c_in_p;
    constexpr bool k_padded = tensor_kkh_tiles * TILE_DIM > tap_cols;
    // One K-block per kernel row: every K-block puts taps, lanes and K padding at the same row offsets.
    constexpr bool one_k_block_per_kh = block_k_tiles == tensor_kkh_tiles;
#ifdef BFP_NOZERO
    // The tap-major gather copies whole slab pixels (lanes included) or zero runs: nothing in im2col needs zeroing.
    constexpr bool zero_im2col_once = !face_direct && !tap_major && one_k_block_per_kh && (lanes_padded || k_padded);
#else
    constexpr bool zero_im2col_once = !face_direct && one_k_block_per_kh && (lanes_padded || k_padded);
#endif
    constexpr uint32_t c_in_tiles = c_in_p / TILE_DIM;
    static_assert(
        (tensor_kkh_tiles % block_k_tiles) == 0 || (face_direct && block_k_tiles % tensor_kkh_tiles == 0),
        "a K-block never straddles a kernel row, unless it is whole kernel rows (face-direct)");
    static_assert(num_k_blocks * block_k_tiles == 3 * tensor_kkh_tiles, "K-blocks tile the kh blocks exactly");
    static_assert((c_in * ELEM_BYTES) % 16 == 0, "per-tap copies move whole 16 B L1 words");
    static_assert(!tile_input || (c_in_p % TILE_DIM) == 0, "TILE slab rows are whole tile columns");
    static_assert(tile_input || input_page_bytes == pixel_bytes, "ROW_MAJOR: one input page is one slab pixel");
    static_assert(
        !tile_input || ring_slab || (slab_tile_rows % untilize_batch_rows) == 0, "slab is whole untilize batches");
    static_assert(
        !ring_slab || (tile_input && tap_major && !face_direct && !channel_slice && !nslice_outer && !strided_window),
        "ring slab: TILE input, the tap-major gather on the K-blocked walk");
    static_assert(!channel_slice || input_shard_rows > 0, "a channel slice is read in owner-shard runs");
    static_assert(
        !slab_split || (!tile_input && !channel_slice && input_shard_rows > 0),
        "the slab fill is split by owner shard of ROW_MAJOR HEIGHT pages");
    static_assert(
        !tile_fill_split || (tile_input && !slab_split && input_shard_rows > 0),
        "the TILE fill is split by owner shard of tile-rows");
    static_assert(fill_window >= 1 && fill_window <= 8, "the windowed fill uses transaction ids 8..15");
    static_assert(fill_window == 1 || (tile_input && !ring_slab), "the windowed fill is the compacted TILE slab's");
    // Tap-major with padded lanes is the short channel slice, whose lanes the reader keeps at zero.
    constexpr bool zero_slab_lanes = tap_major && lanes_padded;
    static_assert(
        !zero_slab_lanes || (channel_slice && (!tile_input || c_in % TILE_DIM == 0)),
        "zeroed slab lanes: a short channel slice, padded by whole tile columns on TILE");
    static_assert(
        !tap_major || (stride == 1 && !k_padded && block_k_cols % c_in_p == 0),
        "tap-major rows are whole taps read in unit-stride pixel runs");
    static_assert(
        !center_direct || (tile_input && tap_major && !face_direct && !row_split && !channel_slice &&
                           block_k_tiles == (center_mode == 1 ? 1 : 3) * (c_in_p / TILE_DIM) && padding == dilation &&
                           input_width % TILE_DIM == 0 && output_width == input_width &&
                           (center_mode == 1 || (block_m_tiles >= 2 && im2col_rm_depth_rows <= block_m_tiles))),
        "center-tap direct: K-blocked walk, whole taps, and the kw = 1 tap is a whole-tile-row vertical shift");

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
    using TapMajorT = conv2d_3x3::TapMajorGather<
        output_width,
        input_height,
        input_width,
        padding,
        dilation,
        c_in_p,
        block_k_tiles,
        center_mode == 2 ? 1 : -1,
        ring_slab ? static_cast<int32_t>(slab_tile_rows * TILE_DIM) : 0>;
    using WindowT = conv2d_3x3::WindowGather<output_width, input_height, input_width, padding, c_in_p>;
    using TapDirectT =
        conv2d_3x3::TapDirect<output_width, input_height, input_width, padding, dilation, c_in_p, block_k_tiles>;
    static_assert(
        !tap_direct || (tap_major && nslice_outer && !center_direct && !ring_slab && !face_direct),
        "tap-direct: the tap-major N-slice-outer gather");
    // The unit address word compute tilizes from: the last 16 B of the unit's cb_im2col_rm pages.
    constexpr uint32_t window_header_offset = block_k_tiles * TILE_DIM * TILE_DIM * ELEM_BYTES - 16;
    static_assert(
        !strided_window ||
            (stride == 2 && dilation == 1 && !tap_major && !face_direct && !row_split && one_k_block_per_kh &&
             c_in_p % TILE_DIM == 0 && WindowT::unit_bytes <= window_header_offset),
        "strided window: stride 2, d 1, whole-tile pixels, one K-block per kernel row, gathered by the reader alone");
    static_assert(!window_direct || (strided_window && nslice_outer), "slab-direct units: the N-slice-outer walk");
    static_assert(!nslice_outer || channel_slice, "the N-slice-outer walk is the K split's");
    // Face-direct under a K split is one whole-K K-block, which already lands in0 as [bm, K].
    constexpr bool tile_row_outer = nslice_outer && !face_direct;
    static_assert(!row_split || (!face_direct && !tap_major && row_queue > 0), "the split gather is the row gather");
    using Units = conv2d_3x3::GatherUnits<tile_row_outer, block_m_tiles, num_k_blocks>;
    // Split row gather: unit s of the push sequence lives in cb_im2col_rm slot s % im2col_rm_depth_rows (every push is
    // one unit and the CB holds exactly that many), so a unit is placed by index, not by the write pointer.
    [[maybe_unused]] const uint32_t im2col_rm_base = row_split ? get_write_ptr(cb_im2col_rm) : 0;
    [[maybe_unused]] const uint32_t im2col_rm_slot_bytes = row_split ? block_k_tiles * get_tile_size(cb_im2col_rm) : 0;
    [[maybe_unused]] uint32_t rows_pushed = 0;      // units pushed to cb_im2col_rm
    [[maybe_unused]] uint32_t assign_cursor = 0;    // next unit nobody gathers yet
    [[maybe_unused]] uint32_t writer_assigned = 0;  // units handed to the writer
    // Per slot: the writer grant (count) of the unit in it, 0 for a reader unit.
    [[maybe_unused]] uint32_t slot_grant[row_split ? im2col_rm_depth_rows : 1];
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
    // The cb_input_tiles ring's base (the windowed fill wraps its batch slots there).
    [[maybe_unused]] const uint32_t input_ring_base = tile_input ? get_write_ptr(cb_input_tiles) : 0;
    // Center-tap slots: cb_im2col_tiles' base (compute produces it; every K-block starts there).
    [[maybe_unused]] const uint32_t im2col_tiles_base = center_mode == 2 ? get_write_ptr(cb_im2col_tiles) : 0;
    // TILE: compute pushes exactly slab_tile_rows tile-rows per M-block and the reader pops them all,
    // so the slab always starts at the CB base.
    const uint32_t slab_l1 = tile_input ? get_read_ptr(cb_halo_slab) : get_write_ptr(cb_halo_slab);
    const uint64_t slab_noc = get_noc_addr(slab_l1);
    [[maybe_unused]] const uint32_t src_table_l1 = tap_direct ? get_write_ptr(cb_src_table) : 0;
    // Tap-direct: units pushed so far; unit u's source entry is slot u % im2col_rm_depth_rows (compute's direct_seq).
    [[maybe_unused]] uint32_t direct_seq = 0;
    conv2d_3x3::TaggedReads<face_direct> reads;

    if constexpr (zero_im2col_once) {
        const uint32_t im2col_rm_bytes = im2col_rm_depth_rows * block_k_tiles * get_tile_size(cb_im2col_rm);
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_rm), im2col_rm_bytes);
        reads.noc.write_zeros_l1_barrier();
    }
#ifdef BFP_LANE_READS
    constexpr bool zero_ring_lanes_once = false;
#else
    constexpr bool zero_ring_lanes_once = true;
#endif
    if constexpr (zero_slab_lanes && (zero_ring_lanes_once || !tile_input)) {
        if constexpr (tile_input) {
            constexpr uint32_t ring_row_bytes = c_in_tiles * input_page_bytes;
            constexpr uint32_t real_row_bytes = (c_in / TILE_DIM) * input_page_bytes;
            for (uint32_t r = 0; r < lane_zero_rows; ++r) {
                reads.noc.async_write_zeros(
                    CircularBuffer(cb_input_tiles),
                    ring_row_bytes - real_row_bytes,
                    {.offset_bytes = r * ring_row_bytes + real_row_bytes});
            }
        } else {
            reads.noc.async_write_zeros(CircularBuffer(cb_halo_slab), lane_zero_rows * pixel_bytes);
        }
        reads.noc.write_zeros_l1_barrier();
    }
    if constexpr (padding_only_faces) {
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_tiles), block_m_tiles * block_k_tiles * in0_tile_bytes);
        reads.noc.write_zeros_l1_barrier();
    }

    // Tap-major zero-fills read a 32-pixel zero block, so an out-of-image run is one read.
    [[maybe_unused]] uint64_t zero_src_noc = 0;
    if constexpr (tap_major) {
        reads.noc.async_write_zeros(CircularBuffer(cb_zero_src), TILE_DIM * pixel_bytes);
        reads.noc.write_zeros_l1_barrier();
        zero_src_noc = get_noc_addr(get_write_ptr(cb_zero_src));
    }
    [[maybe_unused]] const uint32_t zero_src_l1 = tap_major ? get_write_ptr(cb_zero_src) : 0;

    // Input pages [page, end_page) into consecutive L1 page slots from l1. With the split fill, runs of higher
    // shards are the writer's.
    constexpr uint32_t pages_per_row = tile_input ? c_in_tiles : 1;
    // One owner shard: input_shard_rows rows (pixels on ROW_MAJOR, tile-rows on TILE) of pages_per_row pages each.
    constexpr uint32_t owner_shard_pages = input_shard_rows > 0 ? input_shard_rows * pages_per_row : 1;
    const bool skip_higher_shards = slab_split || split_here;
    auto read_pages = [&](uint32_t page, uint32_t end_page, uint32_t l1) {
        while (page < end_page) {
            // Interleaved runs step by the bank count, not by 1; only unit-stride runs are contiguous here.
            const uint32_t run = input.contiguous_page_stride() == 1 ? input.num_contiguous_pages(page, end_page) : 1;
            if (!skip_higher_shards || page / owner_shard_pages <= my_shard) {
                reads.read(input.get_noc_addr(page), l1, run * input_page_bytes);
            }
            page += run;
            l1 += run * input_page_bytes;
        }
    };

    // Input rows [row, end_row) of this core's channels (pixels on ROW_MAJOR, tile-rows on TILE) into
    // consecutive L1 from l1.
    constexpr uint32_t row_bytes = pages_per_row * input_page_bytes;
    // A row's real channels only, when its padded lanes stay zero in L1.
    constexpr uint32_t real_row_bytes = tile_input ? (c_in / TILE_DIM) * input_page_bytes : c_in * ELEM_BYTES;
    auto read_rows = [&](uint32_t row, uint32_t end_row, uint32_t l1) {
        if constexpr (channel_slice) {
            while (row < end_row) {
                const uint32_t shard_end = (row / input_shard_rows + 1) * input_shard_rows;
                const uint32_t run_end = end_row < shard_end ? end_row : shard_end;
                const uint64_t src = input.get_noc_addr(row * row_page_stride + slice_page_offset);
                if (split_here && row / input_shard_rows > my_shard) {
                    // Split fill: a higher shard's rows are the writer's (NoC1).
                } else if constexpr (zero_slab_lanes) {
                    // One owner shard stores the slice's rows back to back, row_bytes apart.
                    for (uint32_t i = 0; i < run_end - row; ++i) {
                        reads.read(src + i * row_bytes, l1 + i * row_bytes, real_row_bytes);
                        if constexpr (tile_input && !zero_ring_lanes_once) {
                            // The row's padding tile columns, from the zero block (tap-major: zeroed at kernel start).
                            reads.read(zero_src_noc, l1 + i * row_bytes + real_row_bytes, row_bytes - real_row_bytes);
                        }
                    }
                } else {
                    reads.read(src, l1, (run_end - row) * row_bytes);
                }
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

    [[maybe_unused]] uint32_t fill_batches_handed = 0;  // split TILE fill: batches handed to the writer
    // Ring slab: the input tile-row at the slab CB's front, its ring slot, and one past the last row pushed.
    [[maybe_unused]] int32_t ring_front_row = 0;
    [[maybe_unused]] uint32_t ring_front_slot = 0;
    [[maybe_unused]] int32_t ring_pushed_hi = 0;
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
        int32_t slab_x_base[3];
        [[maybe_unused]] uint32_t slab_ready_pages[3];  // TILE: slab pages that hold band kh once compute pushed them

        if constexpr (ring_slab) {
            // ---- fill_halo_slab (ring): pop the rows below this block's window (the previous block's gather is done),
            // then stream the window's rows past the last row pushed, in batches; a short last batch is pushed whole
            // and compute pops its unread tail.
            MaybeDeviceZoneScope("reader_fill");
            const conv2d_3x3::RingWindow win = conv2d_3x3::ring_window<input_width, num_in_px, dilation>(footprint);
            // The lowest row this and every later block reads: the ring keeps [keep_lo, ...).
            const int32_t keep_lo = conv2d_3x3::ring_keep_lo<
                output_width,
                input_width,
                stride,
                padding,
                dilation,
                band_last_tap,
                num_in_px,
                num_out_px,
                block_m_tiles>(m_tile_start, actual_m_tiles, m_block_idx, num_m_blocks_this_core);
            if (m_block_idx == 0) {
                ring_front_row = keep_lo;
                ring_pushed_hi = keep_lo;
            } else {
                const int32_t keep = keep_lo < ring_pushed_hi ? keep_lo : ring_pushed_hi;
                const uint32_t dead = static_cast<uint32_t>(keep - ring_front_row);
                if (dead > 0) {
                    cb_wait_front(cb_halo_slab, dead * c_in_tiles);
                    cb_pop_front(cb_halo_slab, dead * c_in_tiles);
                }
                // dead <= the ring's rows <= slab_tile_rows: one wrap at most.
                ring_front_slot += dead;
                if (ring_front_slot >= slab_tile_rows) {
                    ring_front_slot -= slab_tile_rows;
                }
                // Past a gap the ring is empty, and its next row lands at the front slot.
                ring_front_row = keep_lo;
            }
            const int32_t push_lo = ring_pushed_hi > keep_lo ? ring_pushed_hi : keep_lo;
            const uint32_t new_rows = win.hi > push_lo ? static_cast<uint32_t>(win.hi - push_lo) : 0;
            constexpr uint32_t batch_pages = untilize_batch_rows * c_in_tiles;
            for (uint32_t done = 0; done < new_rows; done += untilize_batch_rows) {
                const uint32_t rows = new_rows - done < untilize_batch_rows ? new_rows - done : untilize_batch_rows;
                const uint32_t row0 = static_cast<uint32_t>(push_lo) + done;
                {
                    MaybeDeviceZoneScope("reader_fill_reserve");
                    cb_reserve_back(cb_input_tiles, batch_pages);
                }
                [[maybe_unused]] bool writer_rows = false;
                if constexpr (tile_fill_split) {
                    writer_rows = split_here && row0 + rows > (my_shard + 1) * input_shard_rows;
                    if (writer_rows) {
                        Semaphore<>(slab_free_sem_id).up(1);
                    }
                }
                reads.begin(TRID_INPUT_TILES);
                read_rows(row0, row0 + rows, get_write_ptr(cb_input_tiles));
                reads.wait(TRID_INPUT_TILES);
                if constexpr (tile_fill_split) {
                    if (writer_rows) {
                        MaybeDeviceZoneScope("reader_fill_fwd_wait");
                        Semaphore<>(slab_fwd_done_sem_id).wait_min(++fill_batches_handed);
                    }
                }
                cb_push_back(cb_input_tiles, batch_pages);
            }
            ring_pushed_hi = win.hi > ring_pushed_hi ? win.hi : ring_pushed_hi;
            // Pixel v sits at ring pixel v + base, reduced into the ring by the gather (one wrap at most).
            const int32_t base =
                static_cast<int32_t>(TILE_DIM) * (static_cast<int32_t>(ring_front_slot) - ring_front_row);
            for (int32_t kh = 0; kh < 3; ++kh) {
                slab_x_base[kh] = base;
                slab_ready_pages[kh] =
                    win.band_hi[kh] > 0 ? static_cast<uint32_t>(win.band_hi[kh] - ring_front_row) * c_in_tiles : 0;
            }
        } else if constexpr (tile_input) {
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
            if constexpr (fill_window > 1) {
                // Windowed fill: batches [pushed, batch) are reserved and in flight, batch j at ring slot (write
                // pointer + j - pushed) under transaction id TRID_FILL_BASE + j % fill_window; the oldest is pushed
                // once its reads (and the writer's half) landed.
                constexpr uint32_t num_batches = slab_tile_rows / untilize_batch_rows;
                constexpr uint32_t batch_bytes = batch_pages * input_page_bytes;
                constexpr uint32_t ring_bytes = input_ring_batches * batch_bytes;
                constexpr uint32_t TRID_FILL_BASE = 8;
                // The ring holds the whole slab: reserve it once and hand the writer all of its batches up front, so
                // its higher-shard reads run alongside the reader's from the start.
                constexpr bool whole_slab = input_ring_batches >= num_batches;
                constexpr uint32_t wr_slots = whole_slab ? num_batches : fill_window;
                [[maybe_unused]] bool writer_rows_of[wr_slots];
                if constexpr (whole_slab) {
                    {
                        MaybeDeviceZoneScope("reader_fill_reserve");
                        cb_reserve_back(cb_input_tiles, num_batches * batch_pages);
                    }
                    if constexpr (tile_fill_split) {
                        uint32_t handed = 0;
                        for (uint32_t b = 0; b < num_batches; ++b) {
                            writer_rows_of[b] = split_here && conv2d_3x3::tile_batch_has_rows_from<untilize_batch_rows>(
                                                                  plan, b, (my_shard + 1) * input_shard_rows);
                            handed += writer_rows_of[b] ? 1 : 0;
                        }
                        if (handed > 0) {
                            Semaphore<>(slab_free_sem_id).up(handed);
                        }
                    }
                }
                uint32_t pushed = 0;
                auto retire = [&]() {
                    reads.wait(TRID_FILL_BASE + pushed % fill_window);
                    if constexpr (tile_fill_split) {
                        if (writer_rows_of[pushed % wr_slots]) {
                            MaybeDeviceZoneScope("reader_fill_fwd_wait");
                            Semaphore<>(slab_fwd_done_sem_id).wait_min(++fill_batches_handed);
                        }
                    }
                    cb_push_back(cb_input_tiles, batch_pages);
                    ++pushed;
                };
                for (uint32_t batch = 0; batch < num_batches; ++batch) {
                    if (batch - pushed == fill_window) {
                        retire();
                    }
                    if constexpr (!whole_slab) {
                        {
                            MaybeDeviceZoneScope("reader_fill_reserve");
                            cb_reserve_back(cb_input_tiles, (batch - pushed + 1) * batch_pages);
                        }
                        if constexpr (tile_fill_split) {
                            const bool writer_rows =
                                split_here && conv2d_3x3::tile_batch_has_rows_from<untilize_batch_rows>(
                                                  plan, batch, (my_shard + 1) * input_shard_rows);
                            writer_rows_of[batch % fill_window] = writer_rows;
                            if (writer_rows) {
                                Semaphore<>(slab_free_sem_id).up(1);
                            }
                        }
                    }
                    uint32_t l1 = get_write_ptr(cb_input_tiles) + (batch - pushed) * batch_bytes;
                    if (l1 >= input_ring_base + ring_bytes) {
                        l1 -= ring_bytes;
                    }
                    reads.begin(TRID_FILL_BASE + batch % fill_window);
                    for (uint32_t r = 0; r < untilize_batch_rows && seg < num_segs;) {
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
                }
                while (pushed < num_batches) {
                    retire();
                }
            }
            for (uint32_t batch = 0; fill_window == 1 && batch < slab_tile_rows / untilize_batch_rows; ++batch) {
                {
                    // Back-pressure from compute's untilize.
                    MaybeDeviceZoneScope("reader_fill_reserve");
                    cb_reserve_back(cb_input_tiles, batch_pages);
                }
                // Split fill: hand the reserved batch to the writer first when it holds higher shards' tile-rows.
                [[maybe_unused]] bool writer_rows = false;
                if constexpr (tile_fill_split) {
                    writer_rows = split_here && conv2d_3x3::tile_batch_has_rows_from<untilize_batch_rows>(
                                                    plan, batch, (my_shard + 1) * input_shard_rows);
                    if (writer_rows) {
                        Semaphore<>(slab_free_sem_id).up(1);
                    }
                }
                uint32_t l1 = get_write_ptr(cb_input_tiles);
#ifdef BFP_BATCH_ZONES
                // Diagnostic: the batch's read time by owner direction (hi: a higher shard's row, lo: a lower one's).
                const bool bz_hi = conv2d_3x3::tile_batch_has_rows_from<untilize_batch_rows>(
                    plan, batch, (my_shard + 1) * input_shard_rows);
                const bool bz_lo = !bz_hi && seg < num_segs && seg_first[seg] + seg_row < my_shard * input_shard_rows;
#endif
                auto batch_reads = [&]() {
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
                };
#ifdef BFP_BATCH_ZONES
                if (bz_hi) {
                    MaybeDeviceZoneScope("bfp_rd_hi");
                    batch_reads();
                } else if (bz_lo) {
                    MaybeDeviceZoneScope("bfp_rd_lo");
                    batch_reads();
                } else {
                    MaybeDeviceZoneScope("bfp_rd_own");
                    batch_reads();
                }
#else
                batch_reads();
#endif
                if constexpr (tile_fill_split) {
                    if (writer_rows) {
                        MaybeDeviceZoneScope("reader_fill_fwd_wait");
                        Semaphore<>(slab_fwd_done_sem_id).wait_min(++fill_batches_handed);
                    }
                }
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

        // ---- gather_im2col_blocks: one K-block = block_m_tiles tile-rows x block_k_cols of kernel row kh.
        // Tile-row tile_row of K-block k_block_idx into rows_l1.
        [[maybe_unused]] auto gather_into = [&](uint32_t k_block_idx, uint32_t tile_row, uint32_t rows_l1) {
            const uint32_t k_tile0 = k_block_idx * block_k_tiles;
            const int32_t kh = k_tile0 / tensor_kkh_tiles;
            const uint32_t col_lo = (k_tile0 - kh * tensor_kkh_tiles) * TILE_DIM;
            const uint32_t o = o_first + tile_row * TILE_DIM;
            reads.begin(TRID_GATHER);
            if constexpr (tap_major) {
                TapMajorT::tile_row(reads, slab_noc, zero_src_noc, rows_l1, o, o_end, kh, col_lo, slab_x_base[kh]);
            } else if constexpr (strided_window) {
                uint32_t src = 0;
                if constexpr (window_direct) {
                    src = WindowT::direct_source(slab_l1, o, o_end, kh, slab_x_base[kh]);
                }
                if (src == 0) {
                    WindowT::tile_row(reads, slab_noc, rows_l1, o, o_end, kh, slab_x_base[kh]);
                    src = rows_l1;
                }
                *reinterpret_cast<volatile uint32_t*>(rows_l1 + window_header_offset) = src;
            } else {
                RowGatherT::tile_row(reads, slab_noc, rows_l1, o, o_end, kh, col_lo, slab_x_base[kh]);
            }
            reads.wait(TRID_GATHER);
        };
        [[maybe_unused]] auto gather_unit = [&](uint32_t k_block_idx, uint32_t tile_row) {
            cb_reserve_back(cb_im2col_rm, block_k_tiles);
            gather_into(k_block_idx, tile_row, get_write_ptr(cb_im2col_rm));
            cb_push_back(cb_im2col_rm, block_k_tiles);
        };
        // Center-tap slots (center_mode 2): the kw = 1 tiles of kernel row kh of every tile-row of the M-block,
        // straight into their cb_im2col_tiles slots: tile-row r's at l1_first + r * l1_row_stride. Issued under
        // TRID_CENTER and waited only before the K-block's last im2col row is pushed, which compute tilizes before its
        // matmul reads the slots. A row's source is c_in_tiles consecutive L1 pages of one owner shard, so one accessor
        // lookup per in-shard run.
        [[maybe_unused]] auto issue_center_rows = [&](int32_t kh, uint32_t l1_first, uint32_t l1_row_stride) {
            constexpr uint32_t row_tile_bytes = c_in_tiles * input_page_bytes;
            // Compiled on every path; it only runs with input_width % 32 == 0 (center_direct's static_assert).
            constexpr int32_t tile_rows_per_image_row = input_width >= TILE_DIM ? input_width / TILE_DIM : 1;
            const int32_t shift_rows = (kh * dilation - padding) * tile_rows_per_image_row;
            const uint32_t t_first = m_tile_start + block_tiles_done;
            uint32_t l1 = l1_first;
            reads.begin(TRID_CENTER);
            for (uint32_t r = 0; r < block_m_tiles;) {
                const uint32_t t = t_first + r;
                const int32_t h_in = static_cast<int32_t>(t / tile_rows_per_image_row) + kh * dilation - padding;
                if (r >= block_valid_tiles || h_in < 0 || h_in >= input_height) {
                    reads.read(zero_src_noc, l1, row_tile_bytes);
                    l1 += l1_row_stride;
                    ++r;
                    continue;
                }
                // Rows below stay in the image up to its last row.
                const uint32_t rows_left_in_image =
                    static_cast<uint32_t>(input_height - h_in) * tile_rows_per_image_row - t % tile_rows_per_image_row;
                uint32_t run = block_valid_tiles - r;
                run = run < rows_left_in_image ? run : rows_left_in_image;
                const uint32_t page = (t + shift_rows) * c_in_tiles;
                const uint32_t contig_pages =
                    input.contiguous_page_stride() == 1 ? input.num_contiguous_pages(page, page + run * c_in_tiles) : 1;
                const uint32_t contig_rows = contig_pages / (c_in_tiles > 0 ? c_in_tiles : 1);
                if (contig_rows == 0) {
                    // A row split across pages that are not contiguous: one read per page.
                    for (uint32_t c = 0; c < c_in_tiles; ++c) {
                        reads.read(input.get_noc_addr(page + c), l1 + c * input_page_bytes, input_page_bytes);
                    }
                    l1 += l1_row_stride;
                    ++r;
                    continue;
                }
                uint64_t src = input.get_noc_addr(page);
                for (uint32_t i = 0; i < contig_rows; ++i) {
                    reads.read(src, l1, row_tile_bytes);
                    src += row_tile_bytes;
                    l1 += l1_row_stride;
                }
                r += contig_rows;
            }
        };
        // ---- Split row gather. The reader alone moves cb_im2col_rm's pointers and pushes units in sequence order,
        // a writer unit once row_done counted it.
        const uint32_t m_seq_end = (m_block_idx + 1) * Units::per_block;
        [[maybe_unused]] auto push_one = [&]() {
            const uint32_t grant = slot_grant[rows_pushed % im2col_rm_depth_rows];
            if (grant > 0) {
                Semaphore<>(row_done_sem_id).wait_min(grant);
            }
            cb_push_back(cb_im2col_rm, block_k_tiles);
            ++rows_pushed;
        };
        // Hand the writer the next units below `limit` while it has fewer than row_queue outstanding, has published
        // (row_limit) that it serves them, and they fit the CB.
        [[maybe_unused]] auto top_up_writer = [&](uint32_t limit) {
            while (assign_cursor < limit && assign_cursor + 1 - rows_pushed <= im2col_rm_depth_rows &&
                   writer_assigned - Semaphore<>(row_done_sem_id).value() < row_queue &&
                   assign_cursor < Semaphore<>(row_limit_sem_id).value()) {
                const uint32_t r = assign_cursor;
                cb_reserve_back(cb_im2col_rm, (r + 1 - rows_pushed) * block_k_tiles);
                slot_grant[r % im2col_rm_depth_rows] = ++writer_assigned;
                Semaphore<>(row_mailbox_sem_id + (writer_assigned - 1) % row_queue).set(r);
                Semaphore<>(row_go_sem_id).up(1);
                ++assign_cursor;
            }
        };
        // The reader takes the next unassigned unit, first handing the writer the following ones below `limit`, then
        // gathers it and pushes every unit up to it.
        [[maybe_unused]] auto take_unit = [&](uint32_t limit) {
            const uint32_t s = assign_cursor;
            while (s + 1 - rows_pushed > im2col_rm_depth_rows) {
                push_one();
            }
            cb_reserve_back(cb_im2col_rm, (s + 1 - rows_pushed) * block_k_tiles);
            slot_grant[s % im2col_rm_depth_rows] = 0;
            ++assign_cursor;
            top_up_writer(limit);
            // The writer's serve loop ends once every unit of its K-block (or M-block) is assigned.
            Semaphore<>(row_assigned_sem_id).set(assign_cursor);
            gather_into(
                Units::k_block(s),
                Units::tile_row(s),
                im2col_rm_base + (s % im2col_rm_depth_rows) * im2col_rm_slot_bytes);
            while (rows_pushed <= s) {
                push_one();
            }
        };
        if constexpr (tile_row_outer) {
            // N-slice-outer K split: tile-row outer, K-block inner, so compute's tilize lays in0 out row-major over
            // all of K ([bm, K], the one-K-block layout of the slice matmuls).
            {
                MaybeDeviceZoneScope("reader_band_wait");
                for (int32_t band = 0; band < 3; ++band) {
                    if constexpr (tile_input) {
                        if (slab_ready_pages[band] > 0) {
                            cb_wait_front(cb_halo_slab, slab_ready_pages[band]);
                        }
                    } else {
                        reads.wait(band + 1);
                    }
                }
            }
            MaybeDeviceZoneScope("reader_gather");
            if constexpr (row_split) {
                // Every band is in: the writer may take any unit of the M-block.
                while (assign_cursor < m_seq_end) {
                    take_unit(m_seq_end);
                }
            } else if constexpr (tap_direct) {
                for (uint32_t tile_row = 0; tile_row < block_m_tiles; ++tile_row) {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                        cb_reserve_back(cb_im2col_rm, block_k_tiles);
                        const uint32_t k_tile0 = k_block_idx * block_k_tiles;
                        const int32_t kh = k_tile0 / tensor_kkh_tiles;
                        const uint32_t col_lo = (k_tile0 - kh * tensor_kkh_tiles) * TILE_DIM;
                        volatile tt_l1_ptr uint32_t* srcs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            src_table_l1 + (direct_seq % im2col_rm_depth_rows) * 16);
                        reads.begin(TRID_GATHER);
                        const uint32_t gathered = TapDirectT::tile_row(
                            reads,
                            slab_l1,
                            slab_noc,
                            zero_src_l1,
                            zero_src_noc,
                            get_write_ptr(cb_im2col_rm),
                            o_first + tile_row * TILE_DIM,
                            o_end,
                            kh,
                            col_lo,
                            slab_x_base[kh],
                            srcs);
                        if (gathered > 0) {
                            reads.wait(TRID_GATHER);
                        }
                        cb_push_back(cb_im2col_rm, block_k_tiles);
                        ++direct_seq;
                    }
                }
            } else {
                for (uint32_t tile_row = 0; tile_row < block_m_tiles; ++tile_row) {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                        gather_unit(k_block_idx, tile_row);
                    }
                }
            }
        }
        for (uint32_t k_block_idx = 0; k_block_idx < (tile_row_outer ? 0 : num_k_blocks); ++k_block_idx) {
            const uint32_t k_tile0 = k_block_idx * block_k_tiles;
            const int32_t kh = k_tile0 / tensor_kkh_tiles;
            const uint32_t col_lo = (k_tile0 - kh * tensor_kkh_tiles) * TILE_DIM;
            if constexpr (center_mode == 1) {
                if (k_block_idx % 3 == 1) {
                    // ---- center tap (kh, 1): output tile-row t reads input tile-row t + (kh*d - p) * W / 32, or zeros
                    // above / below the image (and past the core's last tile-row). Copied as whole tile-rows; the
                    // reader never touches the slab for it.
                    MaybeDeviceZoneScope("reader_center");
                    constexpr uint32_t row_tile_pages = c_in_tiles;
                    constexpr uint32_t row_tile_bytes = row_tile_pages * input_page_bytes;
                    constexpr int32_t tile_rows_per_image_row = input_width >= TILE_DIM ? input_width / TILE_DIM : 1;
                    const int32_t shift_rows = (kh * dilation - padding) * tile_rows_per_image_row;
                    cb_reserve_back(cb_center_tiles, block_m_tiles * block_k_tiles);
                    uint32_t l1 = get_write_ptr(cb_center_tiles);
                    const uint32_t t_first = m_tile_start + block_tiles_done;
                    reads.begin(TRID_GATHER);
                    for (uint32_t r = 0; r < block_m_tiles;) {
                        const uint32_t t = t_first + r;
                        const int32_t h_in =
                            static_cast<int32_t>(t / tile_rows_per_image_row) + kh * dilation - padding;
                        if (r >= block_valid_tiles || h_in < 0 || h_in >= input_height) {
                            reads.read(zero_src_noc, l1, row_tile_bytes);
                            l1 += row_tile_bytes;
                            ++r;
                            continue;
                        }
                        // Tile-rows of this image row and below stay inside the image up to its last row.
                        const uint32_t rows_left_in_image =
                            static_cast<uint32_t>(input_height - h_in) * tile_rows_per_image_row -
                            t % tile_rows_per_image_row;
                        uint32_t run = block_valid_tiles - r;
                        run = run < rows_left_in_image ? run : rows_left_in_image;
                        uint32_t page = (t + shift_rows) * row_tile_pages;
                        const uint32_t end_page = page + run * row_tile_pages;
                        while (page < end_page) {
                            const uint32_t n =
                                input.contiguous_page_stride() == 1 ? input.num_contiguous_pages(page, end_page) : 1;
                            reads.read(input.get_noc_addr(page), l1, n * input_page_bytes);
                            page += n;
                            l1 += n * input_page_bytes;
                        }
                        r += run;
                    }
                    reads.wait(TRID_GATHER);
                    cb_push_back(cb_center_tiles, block_m_tiles * block_k_tiles);
                    continue;
                }
            }
            if (col_lo == 0) {
                // The band's slab rows: compute's untilize (TILE) or this core's own band reads (ROW_MAJOR).
                MaybeDeviceZoneScope("reader_band_wait");
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
            if constexpr (row_split) {
                // Only this K-block's band is known to be in: hand-outs stay inside it.
                const uint32_t kb_end = (m_block_idx * num_k_blocks + k_block_idx + 1) * block_m_tiles;
                while (assign_cursor < kb_end) {
                    take_unit(kb_end);
                }
            } else if constexpr (center_mode == 2) {
                // Reserving this K-block's first im2col row implies compute began the previous K-block (the ring is at
                // most bm rows), so the matmul of K-block k - 2, the last reader of slot k % 2, is done.
                const uint32_t k_block_seq = m_block_idx * num_k_blocks + k_block_idx;
                for (uint32_t tile_row = 0; tile_row < block_m_tiles; ++tile_row) {
                    cb_reserve_back(cb_im2col_rm, block_k_tiles);
                    if (tile_row == 0) {
                        issue_center_rows(
                            kh,
                            im2col_tiles_base +
                                ((k_block_seq % 2) * block_m_tiles * block_k_tiles + c_in_tiles) * input_page_bytes,
                            block_k_tiles * input_page_bytes);
                    }
                    gather_into(k_block_idx, tile_row, get_write_ptr(cb_im2col_rm));
                    if (tile_row + 1 == block_m_tiles) {
                        reads.wait(TRID_CENTER);
                    }
                    cb_push_back(cb_im2col_rm, block_k_tiles);
                }
            } else {
                for (uint32_t tile_row = 0; tile_row < block_m_tiles; ++tile_row) {
                    gather_unit(k_block_idx, tile_row);
                }
            }
        }
        if constexpr (row_split) {
            // The writer's trailing units: compute needs them, and the slab must not be refilled under them.
            while (rows_pushed < m_seq_end) {
                push_one();
            }
        }

        if constexpr (window_direct || tap_direct) {
            // Compute tilizes slab runs in place: the slab is dead only once compute popped every unit.
            cb_reserve_back(cb_im2col_rm, im2col_rm_depth_rows * block_k_tiles);
        }
        if constexpr (tile_input && !ring_slab) {
            constexpr uint32_t slab_pages = slab_tile_rows * c_in_tiles;
            cb_wait_front(cb_halo_slab, slab_pages);
            cb_pop_front(cb_halo_slab, slab_pages);
        }
        if constexpr (alias_slab_combine) {
            // Every gather read of the slab (and zero block) has landed and compute's untilize is complete: the
            // writer may now let the row's peers write their slices into this L1.
            Semaphore<>(slab_dead_sem_id).up(1);
        }

        if constexpr (nslice_outer) {
            // NoC0 half of the slice sends: the +d owner slices, from cb_slice_r, in the shared order.
            constexpr uint32_t num_owning_cols = (tensor_n_tiles + out_n_tiles - 1) / out_n_tiles;
            constexpr uint32_t slice_tiles = block_m_tiles * out_n_tiles;
            Semaphore<> credit(combine_credit_sem_id);
            Semaphore<> arrival(combine_arrival_sem_id);
            Noc noc;
            const uint32_t tile_bytes = get_tile_size(cb_slice_r);
            const uint32_t combine_base = get_write_ptr(cb_combine_slots);
            bool credited = false;
            for (uint32_t j = 0; j < k_cols; ++j) {
                const uint32_t y = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
                if (y >= num_owning_cols || !conv2d_3x3::nslice_reader_sends<k_cols>(j)) {
                    continue;
                }
                {
                    MaybeDeviceZoneScope("reader_combine_wait_go");
                    cb_wait_front(cb_slice_r, slice_tiles);
                    if (!credited) {
                        credit.wait_min((m_block_idx + 1) * num_owning_cols);
                        credited = true;
                    }
                }
                {
                    MaybeDeviceZoneScope("reader_combine_send");
                    conv2d_3x3::send_slice<block_m_tiles, out_n_tiles, tensor_n_tiles>(
                        get_read_ptr(cb_slice_r), combine_base, tile_bytes, my_col, y, row_peer_coords_arg);
                    noc_async_write_barrier();
                }
                arrival.up(
                    noc,
                    get_arg_val<uint32_t>(row_peer_coords_arg + 2 * y),
                    get_arg_val<uint32_t>(row_peer_coords_arg + 2 * y + 1),
                    1);
                cb_pop_front(cb_slice_r, slice_tiles);
            }
        } else if constexpr (combine_send) {
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
    if constexpr (nslice_outer) {
        noc_async_atomic_barrier();
    }
}
