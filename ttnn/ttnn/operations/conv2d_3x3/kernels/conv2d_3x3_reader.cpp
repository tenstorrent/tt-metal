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
// N-slice-outer K split (see conv2d_3x3_nslice_order.hpp): the reader gathers tile-row outer, K-block inner, so
// compute tilizes in0 row-major over all of K, and it runs the ring reduce-scatter's forwarding: every partial
// compute packs moves one column on (NoC0) into column my_col + 1's slots.
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
// Windowed TILE fill (compacted slab): the reader keeps up to fill_window cb_input_tiles batches in flight, each under
// its own transaction id, and pushes them in order as they land, so the untilize is fed at the NoC's rate rather than
// one read round trip per batch. Where the ring holds the whole slab (the descriptor puts cb_input_tiles in
// cb_im2col_tiles' L1 when every core runs one M-block: in0 is written only after the untilize consumed every batch),
// it reserves the whole slab once and never waits on compute.
//
// kw-fold fill (the descriptor's _kwf_fill_args): the host plans the slab as reads of (owner core, offset, slab row,
// rows) pieces in runtime args, pushed in groups; on a split core the writer reads the slab's suffix (the higher
// shards' rows and the last own rows). No on-device plan and no TensorAccessor lookup: per page run they cost ~0.1-0.2
// us of RISC time on a control-bound fill (res2 @ LoFi 10.11 -> 9.85 us unzoned with the host plan alone).
//
// Split TILE fill (sharded TILE input, on the weight multicast's pure receivers; a per-core runtime flag): the same NoC
// split for the tile-row fill. The reader reads its own and lower shards' tile-rows into each cb_input_tiles batch it
// reserved; a batch that holds higher shards' tile-rows is handed to the writer (fill_go) and pushed once the writer's
// rows landed (fill_done). With the whole slab reserved, every such batch is handed over up front, so the writer's
// higher-shard reads run alongside the reader's. Only the reader moves the CB's pointers. Under a K split the core
// reads its channel slice of each row; the BLOCK split runs only on cores reading enough from higher shards.
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
// the slice's padded lanes must read as zero. On TILE each row's padding tile columns are read from the zero block
// (cb_zero_src) with the row's real tiles; on ROW_MAJOR the whole slab is zeroed once at kernel start, after which
// every pixel fill reads only its real channels. Zeroing the TILE ring's lanes at kernel start instead delayed the
// short column's fill (the op's critical column) by the zero writes.
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
// Tap-direct on the K-blocked walk (TapDirectFifo in conv2d_3x3_gather_dataflow.hpp): every unit (tile-row x K-block)
// is one 16 B entry of cb_src_table, a FIFO one K-block deep, holding the L1 source compute tilizes each tap from: a
// slab run, the zero block, or a block of cb_im2col_rm the reader gathered (image edges, a run across the ring slab's
// end, a tile-row straddling two output rows), so the reader copies only those. With d = 1 and one kernel row per
// K-block (no center slots) a unit of three slab runs is flagged as one strided window. The slab must outlive compute's
// tilize: on TILE input compute itself is the slab's next writer (its untilize of the next M-block follows its tilize
// of this one), on ROW_MAJOR the reader waits for compute to pop every entry before it refills the slab.
//
// N-slice center taps (the descriptor's NSLICE_CENTER_TAPS; tap-direct N-slice prelude, the kw = 1 tap a whole-tile-row
// shift, the whole slab in the input ring): the center taps of the three kernel rows are input tile-rows of the core's
// channel slice, which the fill already landed in the input ring. Right after the fill the reader copies them (a local
// copy, one read per run of rows inside one slab segment; zero tile-rows outside the image or past the block's valid
// rows) into cb_center_tiles ([3, bm, c_in_tiles], kernel-row major), so the tap-direct units carry taps 0 and 2 only.
// The copy lands before the first unit is pushed: compute's tilize then overwrites the ring (it shares in0's L1).
//
// Padded slab (the descriptor's PADDED_SLAB, on the N-slice center taps; conv2d_3x3_padded_slab.hpp): compute
// untilizes every image row of the slab into a row slot behind d zero pixels, so every tap block is one slab run and
// compute computes and tilizes the sources itself. This reader then publishes nothing and gathers nothing: it fills,
// zeroes the slots' gaps from the zero block with the center copy, pushes the center taps (compute tilizes only after
// that push: its packs overwrite the input ring the copy reads, and it reads the gaps) and waits for compute's one-page
// token on cb_src_table before it pops the slab.
//
// Slab-direct round 0 on NoC0 (the descriptor's SD_R0_NOC0: slab-direct with a fixed-sender weight multicast and no
// weight servers): on the channel's sender this reader multicasts the first weight round, which its writer read from
// DRAM at kernel entry and bias-split, to the channel's rectangles on NoC0, polled from the fill loops (r0_try). The
// writers' NoC1 multicast of it ran beside, and slowed, the forward fill of the cores whose slab lies mostly in higher
// shards (stem.conv1's last grid row: the 108-core split drifts past the 110 input shards), and those cores bound the
// op. Its own ready flag per receiver, so the channel's later rounds keep their handshake. No ack: round 0's slot is
// fresh at kernel start, and a program's go reaches a core only after every core finished the previous program and this
// one's semaphores and runtime args were written (dispatch's wait_for_workers), so the multicast may land before a
// receiver starts. Measured (Blackhole p150b, stem.conv1 512x1024x8 -> 64 s2, LoFi, bias + ReLU, unzoned, 3 alternating
// pairs): 10.27 -> 9.83 us; the same round with no multicast at all (a diagnostic, stale weights) 9.80 us. The
// receivers' kernel-start acks (up to 107 NoC1 atomics into the sender, ~10 ns apart) delayed the sender's own DRAM
// read of the round past its writer's first landing poll: 64x128x8 -> 64 s2 3.34 -> 2.98 us without them, stem.conv1
// flat (its critical core waits on its slab, not the round).
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
#include "conv2d_3x3_slab_direct.hpp"
#include "conv2d_3x3_slab_stream.hpp"

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
    // N-slice-outer K split (see conv2d_3x3_nslice_order.hpp): the reader forwards the ring's partials on NoC0.
    constexpr bool nslice_outer = get_compile_time_arg_val(47) != 0;
    constexpr uint32_t cb_ring_token = get_compile_time_arg_val(48);
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
    // Tap-direct (the descriptor's _tap_direct). N-slice prelude (TapDirectPlanned / TapDirect in
    // conv2d_3x3_gather_dataflow.hpp): per cb_im2col_rm slot, cb_src_table holds one 16 B entry of tap source addresses
    // that compute tilizes from. K-blocked walk: see TapDirectFifo (args 72, 73).
    constexpr bool tap_direct = get_compile_time_arg_val(66) != 0;
    constexpr uint32_t cb_src_table = get_compile_time_arg_val(67);
    // N-slice ring: tile-rows per forwarded row group (compute's slice subblock height; the last group of a slice may
    // be shorter, see conv2d_3x3::ring_group_row0).
    constexpr uint32_t ring_group_rows = get_compile_time_arg_val(68);
    // Tap-direct units per reserve / fence / push (at most half the cb_im2col_rm ring).
    constexpr uint32_t tap_direct_group = get_compile_time_arg_val(69);
    // Windowed TILE fill (see the file comment): cb_input_tiles holds input_ring_batches batches, and the reader keeps
    // up to fill_window of them in flight, each under its own transaction id (TRID_FILL_BASE + batch % fill_window).
    constexpr uint32_t input_ring_batches = get_compile_time_arg_val(70);
    constexpr uint32_t fill_window = get_compile_time_arg_val(71);
    // Tap-direct on the K-blocked walk (TapDirectFifo): cb_src_table is a FIFO of this many 16 B unit entries, and
    // cb_im2col_rm holds only the gathered tap blocks; window units (d = 1, one kernel row per K-block, no center
    // slots).
    constexpr uint32_t tap_direct_fifo_units = get_compile_time_arg_val(72);
    constexpr bool tap_direct_windows = get_compile_time_arg_val(73) != 0;
    // N-slice center taps (see the file comment).
    constexpr bool nslice_center = get_compile_time_arg_val(74) != 0;
    // Padded slab (the descriptor's PADDED_SLAB, conv2d_3x3_padded_slab.hpp): compute untilizes the slab into
    // zero-gapped row slots and tilizes the prelude from sources it computes; this reader zeroes the gaps once, fills,
    // copies the center taps and waits for compute's one-page "slab dead" token on cb_src_table.
#ifdef CONV2D_3X3_PADDED_SLAB
    constexpr bool padded_slab = true;
    constexpr uint32_t pad_slots_max = PAD_SLAB_SLOTS_MAX;
#else
    constexpr bool padded_slab = false;
    constexpr uint32_t pad_slots_max = 0;
#endif
    // Slab-direct (the descriptor's SLAB_DIRECT, conv2d_3x3_slab_direct.hpp): the fill writes the padded slab and
    // pushes it to compute, which unpacks in0 straight from it; no gather. Streamed (SD_STREAM,
    // conv2d_3x3_slab_stream.hpp): pushed as its landed prefix grows.
#ifdef CONV2D_3X3_SLAB_DIRECT
    constexpr bool slab_direct = true;
#else
    constexpr bool slab_direct = false;
#endif
    // Work-balanced M split (the descriptor's balanced_block_ranges): the N-slice prelude (center copy, gather) and the
    // ring forwarding run over the M-block's valid tile-rows only, and the TILE fill streams a host-given slab row
    // count per M-block (runtime args from slab_rows_arg), so a grid row's prelude and walk scale with its own
    // tile-rows.
    constexpr bool rt_valid = get_compile_time_arg_val(75) != 0;
    static_assert(!rt_valid || nslice_outer, "valid-row sizing is the N-slice walk's");
    constexpr auto input_args = TensorAccessorArgs<76>();
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
    // rt_valid: M-block m's slab rows (a whole number of untilize batches) at slab_rows_arg + m.
    constexpr uint32_t slab_rows_arg = split_flag_arg + 1;
#ifdef SD_R0_NOC0
    // Slab-direct round 0 on NoC0 (see the file comment). Args at SD_R0_ARG: 1 on the weight multicast's sender (0
    // elsewhere), then the round's bytes, the rectangles, and per rectangle its NoC0 multicast start x, y, end x, y and
    // destination count (the sender excluded). r0_try: go (this core's writer set SD_R0_LOCAL_SEM to 1: the round
    // landed and its bias is split), then the round and, linked behind it, SD_R0_READY_SEM = 1 to every rectangle;
    // SD_R0_LOCAL_SEM = 2 once the source was read.
    const bool r0_sender = get_arg_val<uint32_t>(SD_R0_ARG) != 0;
    uint32_t r0_state = r0_sender ? 0 : 2;
    auto r0_try = [&]() {
        if (r0_state == 2) {
            return;
        }
        volatile tt_l1_ptr uint32_t* r0_local =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SD_R0_LOCAL_SEM));
        if (r0_state == 0) {
            invalidate_l1_cache();
            if (*r0_local != 1) {
                return;
            }
            MaybeDeviceZoneScope("reader_r0_mcast");
            const uint32_t src = get_write_ptr(SD_R0_CB_WEIGHTS);
            const uint32_t bytes = get_arg_val<uint32_t>(SD_R0_ARG + 1);
            const uint32_t ready = get_semaphore(SD_R0_READY_SEM);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready) = 1;
            for (uint32_t r = 0, a = SD_R0_ARG + 3; r < get_arg_val<uint32_t>(SD_R0_ARG + 2); ++r, a += 5) {
                const uint32_t cnt = get_arg_val<uint32_t>(a + 4);
                if (cnt == 0) {
                    continue;
                }
                const uint32_t x0 = get_arg_val<uint32_t>(a);
                const uint32_t y0 = get_arg_val<uint32_t>(a + 1);
                const uint32_t x1 = get_arg_val<uint32_t>(a + 2);
                const uint32_t y1 = get_arg_val<uint32_t>(a + 3);
                noc_async_write_multicast(src, get_noc_multicast_addr(x0, y0, x1, y1, src), bytes, cnt, true);
                noc_semaphore_set_multicast(ready, get_noc_multicast_addr(x0, y0, x1, y1, ready), cnt, false);
            }
            r0_state = 1;
        }
        if (r0_state == 1 && ncrisc_noc_nonposted_writes_sent(noc_index)) {
            *r0_local = 2;
            r0_state = 2;
        }
    };
#else
    auto r0_try = [&]() {};
#endif
#ifdef CONV2D_3X3_HOST_FILL
    // Host-planned fill (descriptor _host_fill_args): its words follow the slab rows.
    [[maybe_unused]] uint32_t host_fill_arg = slab_rows_arg + (rt_valid ? get_arg_val<uint32_t>(3) : 0);
#endif

    constexpr uint32_t pixel_bytes = c_in_p * ELEM_BYTES;
    constexpr int32_t num_in_px = input_height * input_width;
    constexpr uint32_t block_k_cols = block_k_tiles * TILE_DIM;
    constexpr uint32_t tap_cols = 3 * c_in_p;
    constexpr bool lanes_padded = c_in < c_in_p;
    constexpr bool k_padded = tensor_kkh_tiles * TILE_DIM > tap_cols;
    // One K-block per kernel row: every K-block puts taps, lanes and K padding at the same row offsets.
    constexpr bool one_k_block_per_kh = block_k_tiles == tensor_kkh_tiles;
    // Stride-2 taps from DEST (the descriptor's S2_TAPS, conv2d_3x3_s2_taps.hpp): compute builds every tap from the
    // TILE slab this reader fills; there is no gather, no RM slab handshake and no unit ring.
#ifdef CONV2D_3X3_S2_TAPS
    constexpr bool s2_taps = true;
#else
    constexpr bool s2_taps = false;
#endif
    // The tap-major gather copies whole slab pixels (lanes included) or zero runs: nothing of its im2col needs zeroing.
    constexpr bool zero_im2col_once =
        !face_direct && !tap_major && one_k_block_per_kh && (lanes_padded || k_padded) && !s2_taps;
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
    static_assert(!tile_input || fill_window <= input_ring_batches, "the window never reserves past the ring");
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
    using WindowT = conv2d_3x3::WindowGather<output_width, input_height, input_width, padding, c_in_p>;
    // N-slice center taps: the tap-direct units leave tap 1 to cb_center_tiles.
    constexpr int32_t unit_skip_kw = nslice_center ? 1 : -1;
    using TapDirectT = conv2d_3x3::
        TapDirect<output_width, input_height, input_width, padding, dilation, c_in_p, block_k_tiles, unit_skip_kw>;
    using TapPlannedT = conv2d_3x3::TapDirectPlanned<
        output_width,
        input_height,
        input_width,
        padding,
        dilation,
        c_in_p,
        block_k_tiles,
        unit_skip_kw>;
    static_assert(
        !nslice_center ||
            (tap_direct && tile_input && channel_slice && padding == dilation && input_width % TILE_DIM == 0 &&
             output_width == input_width && block_k_tiles == 3 * (c_in_p / TILE_DIM) && fill_window >= 1 &&
             input_ring_batches * untilize_batch_rows >= slab_tile_rows),
        "N-slice center taps: tap-direct units of one kernel row, the kw = 1 tap a whole-tile-row shift, and the whole "
        "slab in the input ring");
    // Tap-direct on the K-blocked walk (TapDirectFifo); the N-slice prelude keeps the slot-indexed table.
    constexpr bool kb_tap_direct = tap_direct && !nslice_outer;
#ifndef CONV2D_3X3_KW_FOLD
    // Every tap-major gather publishes tap sources (the descriptor's _tap_direct); kw-fold has no gather.
    static_assert(!tap_major || tap_direct, "tap-major units are tap-direct");
#endif
    constexpr uint32_t fifo_taps = c_in_p >= TILE_DIM ? block_k_tiles * TILE_DIM / c_in_p : 1;
    using TapFifoT = conv2d_3x3::TapDirectFifo<
        output_width,
        input_height,
        input_width,
        padding,
        dilation,
        c_in_p,
        block_k_tiles,
        center_mode == 2 ? 1 : -1,
        ring_slab ? static_cast<int32_t>(slab_tile_rows * TILE_DIM) : 0,
        im2col_rm_depth_rows * fifo_taps,
        (tap_direct_fifo_units > 0 ? tap_direct_fifo_units : 1),
        cb_im2col_rm,
        cb_src_table,
        TRID_GATHER,
        tap_direct_windows>;
    static_assert(
        !tap_direct || (tap_direct_group >= 1 &&
                        (kb_tap_direct || tap_direct_group == 1 || 2 * tap_direct_group <= im2col_rm_depth_rows) &&
                        (kb_tap_direct || !TapPlannedT::zero_once || im2col_rm_depth_rows * 3 <= 32)),
        "tap-direct groups: half the ring at most; the zero-once mask holds 3 bits per slot");
    static_assert(
        !tap_direct || (tap_major && !face_direct && !row_split && !strided_window &&
                        (!nslice_outer || (!center_direct && !ring_slab))),
        "tap-direct: the tap-major gather (N-slice prelude or K-blocked walk)");
    static_assert(
        !kb_tap_direct || (tap_direct_fifo_units > 0 && im2col_rm_depth_rows >= 2 &&
                           (center_mode != 2 || tap_direct_fifo_units <= block_m_tiles) &&
                           (center_mode != 1 || block_k_tiles == c_in_p / TILE_DIM)),
        "K-blocked tap-direct: a unit's blocks never fill the block ring; center slots: at most one K-block of "
        "entries");
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
    [[maybe_unused]] uint32_t zeroed_blocks = 0;  // tap-direct zero-once: slot blocks holding their zero rows
    [[maybe_unused]] TapFifoT fifo;               // K-blocked tap-direct state
    if constexpr (kb_tap_direct) {
        fifo.init(get_write_ptr(cb_im2col_rm), src_table_l1);
    }
    conv2d_3x3::TaggedReads<face_direct> reads;

    if constexpr (zero_im2col_once) {
        const uint32_t im2col_rm_bytes = im2col_rm_depth_rows * block_k_tiles * get_tile_size(cb_im2col_rm);
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_rm), im2col_rm_bytes);
        reads.noc.write_zeros_l1_barrier();
    }
    if constexpr (zero_slab_lanes && !tile_input) {
        reads.noc.async_write_zeros(CircularBuffer(cb_halo_slab), lane_zero_rows * pixel_bytes);
        reads.noc.write_zeros_l1_barrier();
    }
    if constexpr (padding_only_faces && !slab_direct) {
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_tiles), block_m_tiles * block_k_tiles * in0_tile_bytes);
        reads.noc.write_zeros_l1_barrier();
    }
#ifdef CONV2D_3X3_SLAB_DIRECT
    using SdGeom = conv2d_3x3::slab_direct::Geom<
        input_height,
        input_width,
        output_width,
        stride,
        padding,
        dilation,
        SD_GAP,
        SD_PITCH,
        pixel_bytes,
        SD_HALF_BYTES>;
#ifdef SD_STREAM
    // Pieces: the union of the three bands, split at image rows (at most one per slot, plus the band splits).
    conv2d_3x3::slab_stream::Stream<SD_SLOTS + 4, SD_STREAM_NTRID_R, 3, SD_STREAM_CHUNK_PX, input_width, SD_GAP>
        sd_stream;
#endif
    {
        // The zero block (cb_im2col_tiles page 0) and every slot's leading zero zone, once: the fill never writes
        // them. Compute reads them only after the first slab push.
        reads.noc.async_write_zeros(CircularBuffer(cb_im2col_tiles), 2 * in0_tile_bytes);
        reads.noc.write_zeros_l1_barrier();
#ifdef SD_FUSED_BIAS
        {
            // The ones block (page 1): bf16 1.0 in K columns 0 and 1 of every 16-element row (fused bias K-step).
            // Zeroed above with the zero block (the CB's two pages); one word per 32 B row, for the rows the two runs
            // read (32 + SD_HALF_BYTES / 32).
            volatile uint32_t* ones =
                reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_im2col_tiles) + in0_tile_bytes);
            for (uint32_t i = 0; i < 8 * (32 + SD_HALF_BYTES / 32); i += 8) {
                ones[i] = 0x3F803F80u;
            }
        }
#endif
        // The gaps: NoC reads of the zero region (each gap is one read). No gap pixel is an image pixel, so the fill
        // (the writer's forward half included) never races them; they land before the first slab push.
        static_assert(SD_ZONE == SD_GAP, "the zero zone of a slot is exactly its gap");
        reads.begin(2);
        for (uint32_t r = 0; r <= SD_SLOTS; ++r) {
            reads.read(reads.zeros_noc, slab_l1 + r * SD_PITCH * pixel_bytes, SD_ZONE * pixel_bytes);
        }
    }
#endif

    // Tap-major zero-fills read a 32-pixel zero block, so an out-of-image run is one read. kw-fold has no gather and no
    // center taps, so nothing reads it there, and its write barrier (~0.27 us) sat in front of the slab fill
    // (center_head.1 unzoned: 9.89 us with the zeroing, 9.47 us without).
    [[maybe_unused]] uint64_t zero_src_noc = 0;
#ifdef CONV2D_3X3_KW_FOLD
    constexpr bool zero_block = false;
#else
    constexpr bool zero_block = tap_major;
#endif
    if constexpr (zero_block) {
        reads.noc.async_write_zeros(CircularBuffer(cb_zero_src), TILE_DIM * pixel_bytes);
        reads.noc.write_zeros_l1_barrier();
        zero_src_noc = get_noc_addr(get_write_ptr(cb_zero_src));
    }
    [[maybe_unused]] const uint32_t zero_src_l1 = tap_major ? get_write_ptr(cb_zero_src) : 0;
    static_assert(
        !padded_slab ||
            (nslice_center && tile_input && padding == dilation && input_width % static_cast<int32_t>(TILE_DIM) == 0),
        "padded slab: N-slice center taps, stride 1, p == d, whole-tile image rows");

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
                        if constexpr (tile_input) {
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
    // kw-fold: the host-planned fill's runtime args (descriptor _kwf_fill_args), per M-block in order.
    [[maybe_unused]] uint32_t kwf_fill_arg = split_flag_arg + 1;
    // Ring slab: the input tile-row at the slab CB's front, its ring slot, and one past the last row pushed.
    [[maybe_unused]] int32_t ring_front_row = 0;
    [[maybe_unused]] uint32_t ring_front_slot = 0;
    // N-slice ring: row groups forwarded to column my_col + 1 so far, over all M-blocks (the absolute arrival count
    // each forward writes there), and the four 16 B source words of those count writes: the first ring token pages
    // (no kernel writes a token page's bytes; the token CB only counts).
    [[maybe_unused]] uint32_t ring_sent = 0;
    [[maybe_unused]] const uint32_t ring_count_words = nslice_outer ? get_read_ptr(cb_ring_token) : 0;
    [[maybe_unused]] int32_t ring_pushed_hi = 0;
    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        const uint32_t block_tiles_done = m_block_idx * block_m_tiles;
        const uint32_t block_valid_tiles =
            (actual_m_tiles - block_tiles_done) < block_m_tiles ? (actual_m_tiles - block_tiles_done) : block_m_tiles;
        const uint32_t o_first = (m_tile_start + block_tiles_done) * TILE_DIM;
        const uint32_t o_end_tiles = o_first + block_valid_tiles * TILE_DIM;
        const uint32_t o_end = o_end_tiles < num_out_px ? o_end_tiles : num_out_px;
        // rt_valid: the slab rows this block untilizes, the tile-rows its prelude gathers (the split row gather hands
        // out a CT unit sequence, so it keeps every row) and the tile-rows its ring walk multiplies and forwards.
        [[maybe_unused]] const uint32_t slab_rows_m =
            rt_valid ? get_arg_val<uint32_t>(slab_rows_arg + m_block_idx) : slab_tile_rows;
        [[maybe_unused]] const uint32_t prelude_rows = (rt_valid && !row_split) ? block_valid_tiles : block_m_tiles;
        [[maybe_unused]] const uint32_t walk_rows = rt_valid ? block_valid_tiles : block_m_tiles;

        const conv2d_3x3::SlabFootprint footprint =
            conv2d_3x3::block_footprint<output_width, input_width, stride, padding, dilation, band_last_tap>(
                o_first, o_end);
        int32_t slab_x_base[3];
        [[maybe_unused]] uint32_t slab_ready_pages[3];  // TILE: slab pages that hold band kh once compute pushed them

        // N-slice center taps: the slab plan (where the ring holds each input tile-row) and cb_center_tiles' block.
        [[maybe_unused]] conv2d_3x3::TileSlabPlan center_plan{};
        [[maybe_unused]] uint32_t center_l1 = 0;
        if constexpr (nslice_center) {
            cb_reserve_back(cb_center_tiles, 3 * block_m_tiles * c_in_tiles);
            center_l1 = get_write_ptr(cb_center_tiles);
        }

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
#ifdef CONV2D_3X3_KW_FOLD
            // kw-fold: the host planned the fill (descriptor _kwf_fill_args: only the slab rows compute reads, as
            // reads from each owner shard's core). Every group of this RISC is issued at once (up to fill_window in
            // flight, one transaction id each) and pushed in slab order as it lands; on a split core the last groups
            // are the writer's (a suffix of the slab: the higher shards' rows and the last own rows), handed over once
            // per block and each pushed on its slab_fwd_done. On-device planning and a TensorAccessor lookup per page
            // run had made this fill control-bound (res2: ~2.2 us with every read stubbed). Nothing reads
            // slab_x_base / slab_ready_pages.
            {
                const uint32_t num_groups = get_arg_val<uint32_t>(kwf_fill_arg++);
                const uint32_t writer_groups = get_arg_val<uint32_t>(kwf_fill_arg++);
                const uint32_t reader_groups = num_groups - writer_groups;
                {
                    MaybeDeviceZoneScope("reader_fill_reserve");
                    cb_reserve_back(cb_input_tiles, slab_rows_m * c_in_tiles);
                }
                if (writer_groups > 0) {
                    Semaphore<>(slab_free_sem_id).up(1);
                }
                const uint32_t slab_l1 = get_write_ptr(cb_input_tiles);
                constexpr uint32_t TRID_FILL_BASE = 8;
                uint32_t group_rows[KWF_FILL_MAX_GROUPS];
                uint32_t pushed = 0;
                uint32_t row_lo = 0;
                auto retire = [&]() {
                    if (pushed < reader_groups) {
                        reads.wait(TRID_FILL_BASE + pushed % fill_window);
                    } else {
                        MaybeDeviceZoneScope("reader_fill_fwd_wait");
                        Semaphore<>(slab_fwd_done_sem_id).wait_min(++fill_batches_handed);
                    }
                    cb_push_back(cb_input_tiles, group_rows[pushed] * c_in_tiles);
                    ++pushed;
                };
                for (uint32_t grp = 0; grp < num_groups; ++grp) {
                    group_rows[grp] = get_arg_val<uint32_t>(kwf_fill_arg++);
                    if (grp < reader_groups) {
                        if (grp - pushed == fill_window) {
                            retire();
                        }
                        const uint32_t pieces = get_arg_val<uint32_t>(kwf_fill_arg++);
                        reads.begin(TRID_FILL_BASE + grp % fill_window);
                        const uint32_t l1 = slab_l1 + row_lo * row_bytes;
                        for (uint32_t i = 0; i < pieces; ++i) {
                            const uint32_t xy = get_arg_val<uint32_t>(kwf_fill_arg);
                            const uint32_t offset = get_arg_val<uint32_t>(kwf_fill_arg + 1);
                            const uint32_t dst = get_arg_val<uint32_t>(kwf_fill_arg + 2);
                            kwf_fill_arg += 3;
                            reads.read(
                                get_noc_addr(xy >> 16, xy & 0xFFFF, input_addr + offset),
                                l1 + (dst >> 16) * row_bytes,
                                (dst & 0xFFFF) * row_bytes);
                        }
                    }
                    row_lo += group_rows[grp];
                }
                while (pushed < num_groups) {
                    retire();
                }
            }
#elif defined(CONV2D_3X3_HOST_FILL)
            // Host-planned fill (the descriptor's BLOCK_HOST_FILL, _host_fill_args; one M-block per core, the whole
            // slab in the ring). Per batch: (writer flag << 31 | gate << 24 on batch 0 | this RISC's pieces), then
            // (owner x << 16 | y, offset in the owner's shard, slab row << 16 | rows) per piece. The reads go out
            // first; the device plan (center taps, band pages) is computed under them. The first `gate` batches (the
            // first kernel row's band, which the prelude reads first: BLOCK_FILL_GATE) go out alone, the rest once
            // they landed. A split core's writer issues its pieces from kernel start (its gated part once its first
            // band landed) and publishes how many batches it issued in slab_fwd_done; this reader sees a writer batch
            // land on NoC1's transaction-id counter, so the writer never serves the fill in its waits. Raw reads
            // instead of a TensorAccessor lookup per run: the accessor has no precomputed-run form, and the
            // device-side plan cost the writer ~0.4 us before its first read and ~0.27 us per 4-row batch (res4.0,
            // identical on every core: control-bound). Raw transaction-id polls of the writer's NoC (capability): no
            // helper waits for another RISC's reads.
            {
                constexpr uint32_t batch_pages = untilize_batch_rows * c_in_tiles;
                constexpr uint32_t max_batches = slab_tile_rows / untilize_batch_rows;
                constexpr uint32_t TRID_FILL_BASE = 8;
                const uint32_t num_batches = slab_rows_m / untilize_batch_rows;
                ASSERT(num_batches <= max_batches && m_block_idx == 0);
                {
                    MaybeDeviceZoneScope("reader_fill_reserve");
                    cb_reserve_back(cb_input_tiles, num_batches * batch_pages);
                }
                const uint32_t hf_slab = get_write_ptr(cb_input_tiles);
                bool hf_writer_of[max_batches];
                uint32_t pushed = 0;
                auto retire = [&]() {
                    reads.wait(TRID_FILL_BASE + pushed % fill_window);
                    if constexpr (tile_fill_split) {
                        if (hf_writer_of[pushed]) {
                            MaybeDeviceZoneScope("reader_fill_fwd_wait");
                            // Writer batch j: issued (slab_fwd_done > j), then landed (no request outstanding on its
                            // transaction id, TRID_FILL_BASE + j % window on the writer's NoC).
                            Semaphore<>(slab_fwd_done_sem_id).wait_min(fill_batches_handed + 1);
                            while (!ncrisc_noc_read_with_transaction_id_flushed(
                                1 - noc_index, TRID_FILL_BASE + fill_batches_handed % HOST_FILL_WRITER_WINDOW)) {
                            }
                            ++fill_batches_handed;
                        }
                    }
                    cb_push_back(cb_input_tiles, batch_pages);
                    ++pushed;
                };
                uint32_t batch = 0;
                auto issue_through = [&](uint32_t end) {
                    for (; batch < end; ++batch) {
                        if (batch - pushed == fill_window) {
                            retire();
                        }
                        const uint32_t word = get_arg_val<uint32_t>(host_fill_arg++);
                        hf_writer_of[batch] = (word >> 31) != 0;
                        const uint32_t pieces = word & 0x00FFFFFFu;
                        reads.begin(TRID_FILL_BASE + batch % fill_window);
                        for (uint32_t i = 0; i < pieces; ++i, host_fill_arg += 3) {
                            const uint32_t xy = get_arg_val<uint32_t>(host_fill_arg);
                            const uint32_t dst = get_arg_val<uint32_t>(host_fill_arg + 2);
                            const uint64_t src = get_noc_addr(
                                xy >> 16, xy & 0xFFFF, input_addr + get_arg_val<uint32_t>(host_fill_arg + 1));
                            const uint32_t l1 = hf_slab + (dst >> 16) * row_bytes;
                            if constexpr (zero_slab_lanes) {
                                // Short last slice: each row's real tiles, its padding tiles from the zero block.
                                for (uint32_t r = 0; r < (dst & 0xFFFF); ++r) {
                                    reads.read(src + r * row_bytes, l1 + r * row_bytes, real_row_bytes);
                                    reads.read(
                                        zero_src_noc, l1 + r * row_bytes + real_row_bytes, row_bytes - real_row_bytes);
                                }
                            } else {
                                reads.read(src, l1, (dst & 0xFFFF) * row_bytes);
                            }
                        }
                    }
                };
                const uint32_t gate_word = (get_arg_val<uint32_t>(host_fill_arg) >> 24) & 0x3Fu;
                const uint32_t gate = gate_word != 0 && gate_word < num_batches ? gate_word : num_batches;
                const uint32_t first = num_batches < fill_window ? num_batches : fill_window;
                issue_through(first < gate ? first : gate);
                conv2d_3x3::TileSlabPlan plan;
                conv2d_3x3::tile_slab_plan<input_width, num_in_px, dilation, c_in_tiles>(footprint, slab_x_base, plan);
                if constexpr (nslice_center) {
                    center_plan = plan;
                }
                for (int32_t kh = 0; kh < 3; ++kh) {
                    slab_ready_pages[kh] = plan.ready_pages[kh];
                }
                while (pushed < gate && gate < num_batches) {
                    retire();
                }
                issue_through(num_batches);
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
                while (pushed < num_batches) {
                    retire();
                }
            }
#else
            conv2d_3x3::TileSlabPlan plan;
            conv2d_3x3::tile_slab_plan<input_width, num_in_px, dilation, c_in_tiles>(footprint, slab_x_base, plan);
            if constexpr (nslice_center) {
                center_plan = plan;
            }
            // Split fill: the slab rows the writer reads (the higher-index shards' input tile-rows).
            [[maybe_unused]] const conv2d_3x3::SlabSuffix writer_sfx =
                tile_fill_split ? conv2d_3x3::slab_rows_from(plan, (my_shard + 1) * input_shard_rows)
                                : conv2d_3x3::SlabSuffix{0, 0};
            for (int32_t kh = 0; kh < 3; ++kh) {
                slab_ready_pages[kh] = plan.ready_pages[kh];
            }
            const uint32_t num_segs = plan.num_segs;
            const uint32_t(&seg_first)[3] = plan.seg_first;
            const uint32_t(&seg_count)[3] = plan.seg_count;

            // Slab rows past the plan's last row are pushed unread: compute untilizes a fixed count.
            constexpr uint32_t batch_pages = untilize_batch_rows * c_in_tiles;
            uint32_t seg = 0;
            uint32_t seg_row = 0;
            {
                // Windowed fill: batches [pushed, batch) are reserved and in flight, batch j at ring slot (write
                // pointer + j - pushed) under transaction id TRID_FILL_BASE + j % fill_window; the oldest is pushed
                // once its reads (and the writer's half) landed.
                constexpr uint32_t max_batches = slab_tile_rows / untilize_batch_rows;
                const uint32_t num_batches = slab_rows_m / untilize_batch_rows;
                ASSERT(num_batches <= max_batches);
                constexpr uint32_t batch_bytes = batch_pages * input_page_bytes;
                constexpr uint32_t ring_bytes = input_ring_batches * batch_bytes;
                constexpr uint32_t TRID_FILL_BASE = 8;
                // The ring holds the whole slab: reserve it once and hand the writer all of its batches up front, so
                // its higher-shard reads run alongside the reader's from the start.
                constexpr bool whole_slab = input_ring_batches >= max_batches;
                constexpr uint32_t wr_slots = whole_slab ? max_batches : fill_window;
                [[maybe_unused]] bool writer_rows_of[wr_slots];
                if constexpr (whole_slab) {
                    {
                        MaybeDeviceZoneScope("reader_fill_reserve");
                        cb_reserve_back(cb_input_tiles, num_batches * batch_pages);
                    }
                    if constexpr (tile_fill_split) {
                        uint32_t handed = 0;
                        for (uint32_t b = 0; b < num_batches; ++b) {
                            writer_rows_of[b] =
                                split_here && conv2d_3x3::batch_has_rows_from<untilize_batch_rows>(writer_sfx, b);
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
                                split_here && conv2d_3x3::batch_has_rows_from<untilize_batch_rows>(writer_sfx, batch);
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
                }
                while (pushed < num_batches) {
                    retire();
                }
            }
#endif
        } else {
            // ---- fill_halo_slab (ROW_MAJOR): band kh tagged trid kh + 1; overlapping bands read only their
            // new tail.
            MaybeDeviceZoneScope("reader_fill");
#ifdef CONV2D_3X3_SLAB_DIRECT
            {
                // Slab-direct: compute is the slab's reader, so it is free once compute popped the previous block.
                {
                    MaybeDeviceZoneScope("reader_fill_reserve");
                    // Polled: compute frees this slab only after the previous block's matmul, which needs round 0
                    // (and the writer's round 1 waits for SD_R0_LOCAL_SEM 2), so round 0's multicast must keep moving.
                    while (!cb_pages_reservable_at_back(cb_halo_slab, SD_SLAB_PX)) {
                        r0_try();
                    }
                    cb_reserve_back(cb_halo_slab, SD_SLAB_PX);
                }
                if constexpr (slab_split) {
                    if (m_block_idx > 0) {
                        Semaphore<>(slab_free_sem_id).up(1);
                    }
                }
#ifdef SD_STREAM
                // Streamed (conv2d_3x3_slab_stream.hpp): this RISC's part (flat pixels below B: own and lower shards)
                // in chunks, the slab CB pushed as the combined landed count grows (the writer's count from
                // slab_fwd_done = block * (SD_SLAB_PX + 1) + count, once this part is done).
                {
                    const uint32_t B = slab_split ? (my_shard + 1) * input_shard_rows : 0xFFFFFFFFu;
                    uint32_t s_b = SD_SLAB_PX;
                    sd_stream.reset(0);
                    SdGeom::fill_pieces<band_last_tap>(o_first, o_end, [&](uint32_t lo, uint32_t hi, uint32_t off) {
                        const uint32_t px = off / pixel_bytes;
                        if (s_b == SD_SLAB_PX && hi > B) {
                            s_b = lo >= B ? px : px + (B - lo);
                        }
                        sd_stream.add(lo, hi < B ? hi : B, px);
                    });
                    sd_stream.region_end = s_b;
                    const uint32_t w_base = m_block_idx * (SD_SLAB_PX + 1);
                    uint32_t pushed = 0;
                    bool gaps_landed = false;
                    auto advance = [&]() {
                        r0_try();
                        sd_stream.poll(reads.noc.get_noc_id());
                        uint32_t target = sd_stream.landed;
                        if (sd_stream.done()) {
                            target = s_b;
                            if constexpr (slab_split) {
                                const uint32_t v = Semaphore<>(slab_fwd_done_sem_id).value();
                                const uint32_t w = v > w_base ? v - w_base : 0;
                                target = w > target ? w : target;
                            } else {
                                target = SD_SLAB_PX;
                            }
                        }
                        if (target > pushed) {
                            if (!gaps_landed) {
                                reads.wait(2);  // the slot gaps (kernel start)
                                gaps_landed = true;
                            }
                            cb_push_back(cb_halo_slab, target - pushed);
                            pushed = target;
                        }
                    };
                    auto read_chunk = [&](uint32_t trid, uint32_t lo, uint32_t hi, uint32_t px) {
                        reads.begin(trid);
                        read_rows(lo, hi, slab_l1 + px * pixel_bytes);
                    };
                    while (!sd_stream.issued()) {
                        r0_try();
                        if (!sd_stream.issue_one(read_chunk)) {
                            advance();
                        }
                    }
                    dataflow_kernel_lib::set_read_trid(reads.noc, 0);
                    {
                        MaybeDeviceZoneScope("reader_fill_fwd_wait");
                        while (pushed < SD_SLAB_PX) {
                            advance();
                        }
                    }
                }
                continue;
#else
                reads.begin(1);
                SdGeom::fill_pieces<band_last_tap>(
                    o_first, o_end, [&](uint32_t lo, uint32_t hi, uint32_t off) { read_rows(lo, hi, slab_l1 + off); });
                while (!ncrisc_noc_read_with_transaction_id_flushed(reads.noc.get_noc_id(), 1)) {
                    r0_try();
                }
                reads.wait(1);
                reads.wait(2);  // the slot gaps (kernel start)
                if constexpr (slab_split) {
                    MaybeDeviceZoneScope("reader_fill_fwd_wait");
                    while (Semaphore<>(slab_fwd_done_sem_id).value() < m_block_idx + 1) {
                        r0_try();
                    }
                }
                cb_push_back(cb_halo_slab, SD_SLAB_PX);
                continue;
#endif
            }
#endif
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

        if constexpr (nslice_center) {
            // ---- N-slice center taps: output tile-row t's center tap of kernel row kh is input tile-row
            // t + (kh * d - p) * W / 32, which the fill landed in the ring (slab order: the plan's segments back to
            // back from the ring base; one read per run inside one segment), or a zero tile-row outside the image or
            // past the block's valid rows. cb_center_tiles row r of kernel row kh at (kh * bm + r) * c_in_tiles pages.
            MaybeDeviceZoneScope("reader_center");
            // Compiled on every path (W < 32 must not divide by zero); it only runs with W % 32 == 0.
            constexpr int32_t tile_rows_per_image_row = input_width >= TILE_DIM ? input_width / TILE_DIM : 1;
            const uint32_t t_first = m_tile_start + block_tiles_done;
            reads.begin(TRID_CENTER);
#ifdef CONV2D_3X3_TAP_SHIFT
            // Tap shift: no row-major slab (compute builds the taps from the input ring, which now sits in the slab's
            // L1): zeroing the gaps would overwrite the ring.
            constexpr bool zero_gaps = false;
#else
            constexpr bool zero_gaps = padded_slab;
#endif
            if constexpr (zero_gaps) {
                // Padded slab: the gap in front of every row slot and after the last one (d pixels each, at fixed
                // offsets), from the zero block, on the center copy's transaction id. The untilize never writes the
                // gaps and compute's tilize reads them only after the center push below, which waits for these reads
                // too (compute must not tilize ahead of that push: zeroing them with the first fill batch instead
                // measured +2.9%, median of 3). One M-block per core: peers write the shared combine slots only after
                // this core's slab is dead.
                constexpr uint32_t slot_bytes = (input_width + dilation) * pixel_bytes;
                constexpr uint32_t gap_bytes = dilation * pixel_bytes;
                constexpr uint32_t zero_block_bytes = TILE_DIM * pixel_bytes;
                for (uint32_t r = 0; r <= pad_slots_max; ++r) {
                    for (uint32_t off = 0; off < gap_bytes; off += zero_block_bytes) {
                        const uint32_t n = gap_bytes - off < zero_block_bytes ? gap_bytes - off : zero_block_bytes;
                        reads.read(zero_src_noc, slab_l1 + r * slot_bytes + off, n);
                    }
                }
            }
            // rt_valid: rows past the block's valid ones are never multiplied, so they are not written.
            const uint32_t center_rows = rt_valid ? block_valid_tiles : block_m_tiles;
            for (int32_t kh = 0; kh < 3; ++kh) {
                const int32_t shift_rows = (kh * dilation - padding) * tile_rows_per_image_row;
                uint32_t l1 = center_l1 + kh * block_m_tiles * row_bytes;
                for (uint32_t r = 0; r < center_rows;) {
                    const uint32_t t = t_first + r;
                    const int32_t h_in = static_cast<int32_t>(t / tile_rows_per_image_row) + kh * dilation - padding;
                    if (r >= block_valid_tiles || h_in < 0 || h_in >= input_height) {
                        reads.read(zero_src_noc, l1, row_bytes);
                        l1 += row_bytes;
                        ++r;
                        continue;
                    }
                    // Tile-rows of this image row and below stay inside the image up to its last row.
                    const uint32_t rows_left = static_cast<uint32_t>(input_height - h_in) * tile_rows_per_image_row -
                                               t % tile_rows_per_image_row;
                    uint32_t run = block_valid_tiles - r;
                    run = run < rows_left ? run : rows_left;
                    const uint32_t row = t + shift_rows;
                    // Band kh's in-image rows hold every center row of kernel row kh: find the run's segment.
                    uint32_t seg_lo = 0;
                    uint32_t slab_row = 0;
                    uint32_t seg_left = 0;
                    for (uint32_t sg = 0; sg < center_plan.num_segs; ++sg) {
                        const uint32_t first = center_plan.seg_first[sg];
                        if (row >= first && row < first + center_plan.seg_count[sg]) {
                            slab_row = seg_lo + (row - first);
                            seg_left = first + center_plan.seg_count[sg] - row;
                            break;
                        }
                        seg_lo += center_plan.seg_count[sg];
                    }
                    ASSERT(seg_left > 0);
                    run = run < seg_left ? run : seg_left;
                    reads.read(get_noc_addr(input_ring_base + slab_row * row_bytes), l1, run * row_bytes);
                    l1 += run * row_bytes;
                    r += run;
                }
            }
        }

#ifdef CONV2D_3X3_KW_FOLD
        // kw-fold (conv2d_3x3_kw_fold.hpp): no gather, and no accumulator seed (compute packs Q_1 as an overwrite).
        if (m_block_idx == 0) {
            // kw-fold's final pass adds the bias as srcB rows: per output column tile n, the fp32 row [b_0 .. b_31]
            // (KWF_CB_BIAS_ROWS page n: srcB rows 0 and 1, each broadcast to its faces' rows), decoded from the bias
            // tiles the writer reads into cb_bias after its first M-block's weights. Done after the first block's fill,
            // not at kernel end: a later block's fill can wait for the first block's slab, which compute frees only
            // after the final pass that adds this bias.
            // Row 0 of a tile is row 0 of faces 0 and 1: bf16 at datum c % 16 (+ 256 for face 1); bf8_b
            // (KWF_BIAS_BFP8): 64 exponent bytes, one per face row (face f row 0 at byte 16 f), then sign-magnitude
            // bytes m, value (-1)^s (m & 127) 2^(e - 133), from byte 64 (+ 256 for face 1).
            MaybeDeviceZoneScope("reader_kwf_bias_rows");
            cb_wait_front(KWF_CB_BIAS, KWF_N);
            invalidate_l1_cache();
            cb_reserve_back(KWF_CB_BIAS_ROWS, KWF_N);
            const uint32_t raw = get_read_ptr(KWF_CB_BIAS);
            const uint32_t tile_bytes = get_tile_size(KWF_CB_BIAS);
            uint32_t* rows = reinterpret_cast<uint32_t*>(get_write_ptr(KWF_CB_BIAS_ROWS));
            for (uint32_t n = 0; n < KWF_N; ++n, rows += 32) {
                const uint32_t t = raw + n * tile_bytes;
                for (uint32_t c = 0; c < 32; ++c) {
#if KWF_BIAS_BFP8
                    const uint32_t e = reinterpret_cast<const uint8_t*>(t)[16 * (c / 16)];
                    const uint32_t m = reinterpret_cast<const uint8_t*>(t)[64 + 256 * (c / 16) + c % 16];
                    const uint32_t mag = m & 0x7f;
                    uint32_t v = 0;
                    if (mag != 0) {
                        // Normalize the 7-bit magnitude to bit 6 (at most 6 shifts; no clz libcall).
                        uint32_t mm = mag;
                        uint32_t sh = 0;
                        while (!(mm & 0x40)) {
                            mm <<= 1;
                            ++sh;
                        }
                        v = ((m & 0x80) << 24) | ((e - sh) << 23) | ((mm & 0x3f) << 17);
                    }
#else
                    const uint32_t v =
                        static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(t)[256 * (c / 16) + c % 16]) << 16;
#endif
                    rows[c] = v;
                }
            }
            cb_push_back(KWF_CB_BIAS_ROWS, KWF_N);
            cb_pop_front(KWF_CB_BIAS, KWF_N);
        }
        continue;
#endif
        // ---- gather_im2col_blocks: one K-block = block_m_tiles tile-rows x block_k_cols of kernel row kh.
        // Tile-row tile_row of K-block k_block_idx into rows_l1.
        [[maybe_unused]] auto gather_into = [&](uint32_t k_block_idx, uint32_t tile_row, uint32_t rows_l1) {
            const uint32_t k_tile0 = k_block_idx * block_k_tiles;
            const int32_t kh = k_tile0 / tensor_kkh_tiles;
            const uint32_t col_lo = (k_tile0 - kh * tensor_kkh_tiles) * TILE_DIM;
            const uint32_t o = o_first + tile_row * TILE_DIM;
            reads.begin(TRID_GATHER);
            if constexpr (strided_window) {
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
        if constexpr (tile_row_outer && !s2_taps) {
            // N-slice-outer K split: tile-row outer, K-block inner, so compute's tilize lays in0 out row-major over
            // all of K ([bm, K], the one-K-block layout of the slice matmuls).
            // Padded slab: compute tilizes from the slab it untilized, with no reader handshake.
            if constexpr (!padded_slab) {
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
            } else if constexpr (padded_slab) {
                // Padded slab: no gather (compute computes every tap source). The center copy reads the input ring,
                // which compute's tilize overwrites: compute tilizes only once this push says it landed.
                MaybeDeviceZoneScope("reader_center_wait");
                reads.wait(TRID_CENTER);
                cb_push_back(cb_center_tiles, 3 * block_m_tiles * c_in_tiles);
            } else if constexpr (tap_direct) {
                if constexpr (nslice_center) {
                    // The center copy reads the input ring, which compute's tilize overwrites once a unit is pushed.
                    MaybeDeviceZoneScope("reader_center_wait");
                    reads.wait(TRID_CENTER);
                    cb_push_back(cb_center_tiles, 3 * block_m_tiles * c_in_tiles);
                }
                // Each tile-row's taps are classified once (TapDirectPlanned::plan; a tile-row that is not one
                // output-row run takes TapDirect::tile_row), and units are reserved, fenced and pushed in groups of
                // tap_direct_group: one trid wait and one push per group. A group never crosses the ring end (a CB
                // push must land exactly on it), so its slots are consecutive. The reader's per-unit work, not the
                // read latency, paced the prelude, so the group is fenced in place.
                const uint32_t units_per_block = prelude_rows * num_k_blocks;
                const uint32_t unit_bytes = block_k_tiles * get_tile_size(cb_im2col_rm);
                uint32_t g_left = 0;       // units of the current group still to issue
                uint32_t g_size = 0;       // units in the current group
                uint32_t g_base = 0;       // slot address of the group's first unit
                uint32_t g_reads = 0;      // taps the current group gathered
                uint32_t fixed_state = 0;  // the fill's reads reprogrammed the read command buffer
                uint32_t u = 0;
                TapPlannedT::Row plan;
                for (uint32_t tile_row = 0; tile_row < prelude_rows; ++tile_row) {
                    const uint32_t o = o_first + tile_row * TILE_DIM;
                    const bool planned = TapPlannedT::plan(plan, o, o_end);
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx, ++u, ++direct_seq) {
                        const uint32_t slot = direct_seq % im2col_rm_depth_rows;
                        if (g_left == 0) {
                            const uint32_t to_ring_end = im2col_rm_depth_rows - slot;
                            g_size = units_per_block - u < tap_direct_group ? units_per_block - u : tap_direct_group;
                            g_size = g_size < to_ring_end ? g_size : to_ring_end;
                            cb_reserve_back(cb_im2col_rm, g_size * block_k_tiles);
                            g_base = get_write_ptr(cb_im2col_rm);
                            g_left = g_size;
                            g_reads = 0;
                            reads.begin(TRID_GATHER);
                        }
                        const uint32_t slot_l1 = g_base + (g_size - g_left) * unit_bytes;
                        const uint32_t k_tile0 = k_block_idx * block_k_tiles;
                        const int32_t kh = k_tile0 / tensor_kkh_tiles;
                        const uint32_t col_lo = (k_tile0 - kh * tensor_kkh_tiles) * TILE_DIM;
                        volatile tt_l1_ptr uint32_t* srcs =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_table_l1 + slot * 16);
                        if (planned) {
                            g_reads += TapPlannedT::unit(
                                reads,
                                plan,
                                slab_l1,
                                slab_noc,
                                zero_src_l1,
                                zero_src_noc,
                                slot_l1,
                                kh,
                                col_lo / c_in_p,
                                slab_x_base[kh],
                                srcs,
                                zeroed_blocks,
                                slot,
                                fixed_state);
                        } else {
                            fixed_state = 0;
                            g_reads += TapDirectT::tile_row(
                                reads,
                                slab_l1,
                                slab_noc,
                                zero_src_l1,
                                zero_src_noc,
                                slot_l1,
                                o,
                                o_end,
                                kh,
                                col_lo,
                                slab_x_base[kh],
                                srcs);
                        }
                        if (--g_left == 0) {
                            if (g_reads > 0) {
                                reads.wait(TRID_GATHER);
                            }
                            cb_push_back(cb_im2col_rm, g_size * block_k_tiles);
                        }
                    }
                }
            } else {
                for (uint32_t tile_row = 0; tile_row < prelude_rows; ++tile_row) {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                        gather_unit(k_block_idx, tile_row);
                    }
                }
            }
        }
        for (uint32_t k_block_idx = 0; k_block_idx < (tile_row_outer || s2_taps ? 0 : num_k_blocks); ++k_block_idx) {
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
            } else if constexpr (kb_tap_direct) {
                // Tap-direct K-block (TapDirectFifo): one entry per unit, reserved and pushed tap_direct_group at a
                // time. Center-tap slots: the center tiles are issued once the K-block's first group is reserved (at
                // most one K-block of entries is outstanding, so compute began the previous K-block: the matmul of
                // K-block k - 2, the last reader of slot k % 2, is done) and land before its last entry is pushed.
                const uint32_t kw_first = col_lo / c_in_p;
                [[maybe_unused]] const uint32_t k_block_seq = m_block_idx * num_k_blocks + k_block_idx;
                fifo.fixed_state = 0;
                reads.begin(TRID_GATHER);
                for (uint32_t tile_row = 0; tile_row < block_m_tiles; ++tile_row) {
                    fifo.reserve(tap_direct_group, block_m_tiles - tile_row);
                    if constexpr (center_mode == 2) {
                        if (tile_row == 0) {
                            issue_center_rows(
                                kh,
                                im2col_tiles_base +
                                    ((k_block_seq % 2) * block_m_tiles * block_k_tiles + c_in_tiles) * input_page_bytes,
                                block_k_tiles * input_page_bytes);
                            fifo.fixed_state = 0;
                            reads.begin(TRID_GATHER);
                        }
                        if (tile_row + 1 == block_m_tiles) {
                            MaybeDeviceZoneScope("reader_center_wait");
                            reads.wait(TRID_CENTER);
                        }
                    }
                    fifo.unit(
                        reads,
                        slab_l1,
                        slab_noc,
                        zero_src_l1,
                        zero_src_noc,
                        o_first + tile_row * TILE_DIM,
                        o_end,
                        kh,
                        kw_first,
                        slab_x_base[kh]);
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

        if constexpr (padded_slab) {
            // Compute tilized the prelude straight from the slab: its token says the slab is dead.
            MaybeDeviceZoneScope("reader_slab_dead_wait");
            cb_wait_front(cb_src_table, 1);
            cb_pop_front(cb_src_table, 1);
        } else if constexpr ((window_direct && !s2_taps) || (tap_direct && nslice_outer)) {
            // Compute tilizes slab runs in place: the slab is dead only once compute popped every unit.
            cb_reserve_back(cb_im2col_rm, im2col_rm_depth_rows * block_k_tiles);
        } else if constexpr (kb_tap_direct && !tile_input) {
            // ROW_MAJOR: this reader refills the slab for the next M-block itself, so it waits until compute popped
            // every entry pointing into it. TILE needs no wait: compute's untilize of the next block, the slab's only
            // writer, runs after its own tilize of this one (same unpacker, in order).
            if (m_block_idx + 1 < num_m_blocks_this_core) {
                cb_reserve_back(cb_src_table, tap_direct_fifo_units);
            }
        }
        if constexpr (tile_input && !ring_slab && !s2_taps) {
            const uint32_t slab_pages = slab_rows_m * c_in_tiles;
            cb_wait_front(cb_halo_slab, slab_pages);
            cb_pop_front(cb_halo_slab, slab_pages);
        }
        if constexpr (alias_slab_combine) {
            // Every gather read of the slab (and zero block) has landed and compute's untilize is complete: the
            // writer may now let the row's peers write their slices into this L1.
            Semaphore<>(slab_dead_sem_id).up(1);
        }

        if constexpr (nslice_outer) {
            // Ring reduce-scatter forwarding (conv2d_3x3_nslice_order.hpp). Compute packs step j's partial into slot
            // k_cols - 1 - j one row group at a time, each onto the incoming partial it waited for itself (the arrival
            // count, see the compute kernel), and pushes a ring token after each. The reader writes each group one
            // column on (NoC0, +x) into slot k_cols - 2 - j of column my_col + 1 (that core's step j + 1 partial) and
            // pops the token. Right behind the group it writes that core's arrival count (the absolute number of
            // groups sent there): same command buffer, same static VC, same destination, so the count lands after the
            // data with no wait for the data's ack, and the arrival is not delayed by this reader's next wait.
            // The previous protocol freed the packer's next group from this reader once its incoming partial landed,
            // and signalled each arrival after the write's ack: two RISC hand-overs and a NoC round trip per hop.
            // Measured (Blackhole p150b, unzoned, medians of 3): a 5-owner 11-column ring (6 non-owning columns) at
            // LoFi 13905 -> 11036 ns, a 2-slice ring 5339 -> 5151 ns. res4.x: the everything-stubbed skeleton 5990 ->
            // 4964 ns (one run), weights stubbed 10653 -> 10436 ns, real 11532 -> 11595 ns (flat: the walk waits on the
            // weight stream, and with it stubbed on the matmul).
            MaybeDeviceZoneScope("reader_combine_send");
            constexpr uint32_t num_owning_cols = (tensor_n_tiles + out_n_tiles - 1) / out_n_tiles;
            constexpr uint32_t slot_tiles = block_m_tiles * out_n_tiles;
            // rt_valid: only the block's valid tile-rows are multiplied and forwarded (every core of the grid row has
            // the same M range, so the row's columns agree on the groups).
            const uint32_t groups = (walk_rows + ring_group_rows - 1) / ring_group_rows;
            // Transaction ids (one per count word, in rotation): a count word is rewritten only once the writes issued
            // under its id have left L1.
            constexpr uint32_t ring_trids = 4;
            constexpr uint32_t ring_trid_base = 12;
            Semaphore<> credit(combine_credit_sem_id);
            const uint32_t slot_bytes = slot_tiles * get_tile_size(cb_combine_slots);
            const uint32_t row_bytes = slot_bytes / block_m_tiles;
            const uint32_t combine_base = get_write_ptr(cb_combine_slots);
            const uint32_t next_col = my_col + 1 == k_cols ? 0 : my_col + 1;
            const uint32_t nx = get_arg_val<uint32_t>(row_peer_coords_arg + 2 * next_col);
            const uint32_t ny = get_arg_val<uint32_t>(row_peer_coords_arg + 2 * next_col + 1);
            const uint64_t next_arrival = get_noc_addr(nx, ny, get_semaphore(combine_arrival_sem_id));
            bool credited = false;
            const uint32_t first = conv2d_3x3::ring_next_real<k_cols, num_owning_cols>(my_col, 0);
            for (uint32_t j = first; j < k_cols;
                 j = conv2d_3x3::ring_next_real<k_cols, num_owning_cols>(my_col, j + 1)) {
                for (uint32_t g = 0; g < groups; ++g) {
                    {
                        MaybeDeviceZoneScope("reader_ring_wait_step");
                        cb_wait_front(cb_ring_token, 1);
                    }
                    MaybeDeviceZoneScope("reader_ring_send");
                    if (j + 1 < k_cols) {
                        if (!credited) {
                            // Column my_col + 1's slots are free (its slab is dead, its previous M-block combined).
                            credit.wait_min(m_block_idx + 1);
                            credited = true;
                        }
                        const uint32_t w = ring_sent % ring_trids;
                        const uint32_t trid = ring_trid_base + w;
                        while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, trid)) {
                        }
                        const uint32_t owner = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
                        const uint32_t r0 = rt_valid
                                                ? conv2d_3x3::ring_group_row0_rt<ring_group_rows>(walk_rows, owner, g)
                                                : conv2d_3x3::ring_group_row0<block_m_tiles, ring_group_rows>(owner, g);
                        const uint32_t rows = walk_rows - r0 < ring_group_rows ? walk_rows - r0 : ring_group_rows;
                        const uint32_t group_bytes = rows * row_bytes;
                        const uint32_t src = combine_base + (k_cols - 1 - j) * slot_bytes + r0 * row_bytes;
                        const uint32_t dst = combine_base + (k_cols - 2 - j) * slot_bytes + r0 * row_bytes;
                        for (uint32_t off = 0; off < group_bytes; off += NOC_MAX_BURST_SIZE) {
                            const uint32_t bytes =
                                group_bytes - off < NOC_MAX_BURST_SIZE ? group_bytes - off : NOC_MAX_BURST_SIZE;
                            noc_async_write_one_packet_with_trid(
                                src + off, get_noc_addr(nx, ny, dst + off), bytes, trid);
                        }
                        ++ring_sent;
                        const uint32_t word = ring_count_words + w * 16;
                        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = ring_sent;
                        // The store reaches L1 before the NoC reads it (Blackhole stores can land out of order).
                        asm volatile("fence" ::: "memory");
                        noc_async_write_one_packet_with_trid(word, next_arrival, 4, trid);
                    }
                    cb_pop_front(cb_ring_token, 1);
                }
            }
            for (uint32_t t = 0; t < ring_trids; ++t) {
                while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, ring_trid_base + t)) {
                }
            }
            // The firmware requires every NoC packet tag cleared at kernel end.
            noc_async_write_set_trid(0);
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
#ifdef SD_R0_NOC0
    while (r0_state != 2) {
        r0_try();
    }
    noc_async_write_barrier();
#endif
    if constexpr (nslice_outer) {
        noc_async_atomic_barrier();
    }
}
