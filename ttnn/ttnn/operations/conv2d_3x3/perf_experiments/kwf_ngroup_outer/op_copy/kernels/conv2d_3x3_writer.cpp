// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 writer (BRISC / NoC1): prefetch_bias once, then per M-block stream_weight_blocks and
// store_block. Weight K-blocks of the next M-block are prefetched between output row groups, but only
// into slots that are already free, so the writer never blocks on cb_weights while compute waits on
// the output CB.
//
// Sharded output: cb_output_tiles is backed on the output shard, so compute packs in place and
// store_block is empty. The writer only streams bias and weights; compute never waits on the output
// CB there, so blocking weight reserves cannot deadlock. A K split on more grid rows than the output has shard rows
// (remote_output) packs into a local CB of one [bm, out_n] block instead, which the writer drains after the M-block's
// weights and NoC-writes tile by tile to the output shard that holds each tile-row (a TensorAccessor over the sharded
// output); compute never waits on it within a block.
//
// Weight multicast (regime R1, the `weight_mcast` channel): every K-block is one round. The sender
// reads it from DRAM into its own cb_weights slot and multicasts it to the same slot on every
// receiver; a receiver acks only after reserving that slot, so the slot is free. All participants
// push the same quanta in the same order, so their cb_weights write pointers agree every round.
// A core outside the channel (one with fewer M-blocks, or an absent channel) reads its own weights.
// Under a K split the channel has one group per grid column, whose sender reads that column's K-slice.
//
// Weight servers (HEIGHT-sharded K-blocked walk, when the device has a spare core per channel rectangle): the rounds
// come from a server core outside the work grid instead (conv2d_3x3_weight_server_{reader,writer}.cpp: DRAM reads on
// NoC0, multicast on NoC1, one server per rectangle), and every participant is a pure receiver. A participant acks its
// server once at kernel start (it runs and its semaphores are initialized) and then once per round that reuses a
// cb_weights slot (round >= weight depth: compute freed the slot); the server sets one flag per slot after the round's
// data, and the receiver clears it. The bias reads are issued at kernel start and completed after the first round:
// compute needs the bias only after its walk, and every core reading the same bias pages at once took ~3 us.
//
// K split (BLOCK_SHARDED, k_cols > 1; regime R4): every core of a grid row computes fp32 partials
// [bm, all C_out] over its own channel slice, and grid column y owns output N-slice y. Per M-block the
// writer runs one row-local reduce-scatter round:
//   1. a core that owns output columns reserves its whole cb_combine_slots and gives every row peer
//      (itself included) one credit: the slots are free;
//   2. once compute pushed its partials and every owning column gave its credit, the core writes N-slice
//      y of its partials into slot `my_col` of column y, flushes, and bumps column y's arrival count;
//   3. after k_cols arrivals the owning core pushes the slots to compute, which sums them.
// With combine_split_nocs the scatter is shared with the reader: every core receives and sends k_cols
// slices per round, and one NoC interface carrying all of them in and out is what bounds the round. The
// writer sends to its own and lower columns on NoC1 (which runs toward lower x/y), the reader to the
// higher columns on NoC0. Two core-local counters hand the round over: send_go (writer -> reader:
// partials pushed, credits in) and send_done (reader -> writer: its slices written and acked). The
// writer alone pops the partials and bumps the arrival counts, after both halves landed.
// Split slab fill (HEIGHT-sharded ROW_MAJOR, see the reader's file comment): before each M-block's weights the
// writer reads the slab runs owned by higher-index input shards on NoC1, whose responses take the short way back,
// and raises slab_fwd_done; from the second block on it first waits for the reader's slab_free.
//
// Split TILE fill (sharded TILE input, pure weight-multicast receivers; a per-core runtime flag): the reader reserves
// each cb_input_tiles batch and, when it holds tile-rows of higher-index input shards, hands it over (fill_go); the
// writer reads those rows on NoC1 into the batch's ring slot and counts it (fill_done), and the reader pushes the batch
// after both halves. On the compacted slab the writer keeps up to fill_writer_window handed-over batches in flight (own
// transaction id each) and counts them in order as they land. The writer serves hand-overs wherever it waits (weight
// reserve and receive, bias read, row-gather serving, and on the N-slice walk its weight-read, slab_dead, partials,
// credit and arrival waits), because compute untilizes a block before it multiplies it. Under a K split the writer
// reads the core's channel slice of each row, a short slice's padding tiles from the reader's zero block.
//
// Split face-direct gather: after issuing each K-block's weights the writer waits for the reader's gather_go (the
// in0 block is reserved and the slab complete), then writes every odd row group (in0_stream_rows tile-rows) on
// NoC1, raising gather_done after each; the reader pushes the groups in order.
//
// N-slice-outer K split (the default under a K split when the resident in0 fits L1): a ring reduce-scatter
// (conv2d_3x3_nslice_order.hpp). Compute multiplies one owner N-slice at a time over the whole K and accumulates it
// onto the partial column my_col - 1 sent; the reader forwards each partial to column my_col + 1. The writer streams
// each slice's weights (K x out_n tiles, one multicast round per slice down the column), up to the weight CB's depth
// ahead; the column's weight-reading core issues round r + 1's DRAM reads (own transaction id) before it multicasts
// round r. It also gives column my_col - 1, the only core that writes this core's slots, one credit per M-block once
// the slots are free.
//
// Shared-L1 N-slice walk (whenever every core runs one M-block): cb_combine_slots shares cb_halo_slab's L1, dead once
// the core's gather landed. The credit therefore waits for the reader's slab_dead, so no peer writes a slot before
// then; the first weight rounds and the split row gather's serving still go out before that wait.
//
// Every push of cb_combine_slots is its whole capacity, so its write pointer never leaves the CB base and
// the same base address is right on every core. The counters only grow, so round j waits for
// (j + 1) x the per-round count.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "conv2d_3x3_combine_dataflow.hpp"
#include "conv2d_3x3_slab_dataflow.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"
#include "conv2d_3x3_nslice_order.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(8);
    constexpr bool has_bias = get_compile_time_arg_val(9) != 0;
    constexpr bool output_sharded = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t out_n_tiles = get_compile_time_arg_val(11);
    constexpr bool k_split = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t k_cols = get_compile_time_arg_val(13);
    constexpr uint32_t cb_partials = get_compile_time_arg_val(14);
    constexpr uint32_t cb_combine_slots = get_compile_time_arg_val(15);
    constexpr uint32_t combine_credit_sem_id = get_compile_time_arg_val(16);
    constexpr uint32_t combine_arrival_sem_id = get_compile_time_arg_val(17);
    constexpr bool combine_split_nocs = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t combine_send_go_sem_id = get_compile_time_arg_val(19);
    constexpr uint32_t combine_send_done_sem_id = get_compile_time_arg_val(20);
    constexpr bool slab_split = get_compile_time_arg_val(21) != 0;
    constexpr uint32_t cb_halo_slab = get_compile_time_arg_val(22);
    constexpr int32_t input_height = get_compile_time_arg_val(23);
    constexpr int32_t input_width = get_compile_time_arg_val(24);
    constexpr uint32_t output_width = get_compile_time_arg_val(25);
    constexpr uint32_t num_out_px = get_compile_time_arg_val(26);
    constexpr int32_t padding = get_compile_time_arg_val(27);
    constexpr int32_t stride = get_compile_time_arg_val(28);
    constexpr int32_t dilation = get_compile_time_arg_val(29);
    constexpr int32_t band_stride_px = get_compile_time_arg_val(30);
    constexpr int32_t band_last_tap = get_compile_time_arg_val(31);
    constexpr uint32_t input_page_bytes = get_compile_time_arg_val(32);
    constexpr uint32_t input_shard_rows = get_compile_time_arg_val(33);
    constexpr uint32_t slab_fwd_done_sem_id = get_compile_time_arg_val(34);
    constexpr uint32_t slab_free_sem_id = get_compile_time_arg_val(35);
    constexpr bool gather_split = get_compile_time_arg_val(36) != 0;
    constexpr uint32_t in0_stream_rows = get_compile_time_arg_val(37);
    constexpr uint32_t gather_go_sem_id = get_compile_time_arg_val(38);
    constexpr uint32_t gather_done_sem_id = get_compile_time_arg_val(39);
    constexpr uint32_t cb_im2col_tiles = get_compile_time_arg_val(40);
    constexpr uint32_t c_in_p = get_compile_time_arg_val(41);
    constexpr uint32_t tensor_kkh_tiles = get_compile_time_arg_val(42);
    constexpr bool nslice_outer = get_compile_time_arg_val(43) != 0;
    constexpr uint32_t nslice_weight_depth = get_compile_time_arg_val(44);
    // Split row gather (see the reader's file comment).
    constexpr bool row_split = get_compile_time_arg_val(45) != 0;
    constexpr uint32_t row_queue = get_compile_time_arg_val(46);
    constexpr uint32_t row_go_sem_id = get_compile_time_arg_val(47);
    constexpr uint32_t row_done_sem_id = get_compile_time_arg_val(48);
    constexpr uint32_t row_limit_sem_id = get_compile_time_arg_val(49);
    constexpr uint32_t row_assigned_sem_id = get_compile_time_arg_val(50);
    constexpr uint32_t row_mailbox_sem_id = get_compile_time_arg_val(51);
    constexpr uint32_t cb_im2col_rm = get_compile_time_arg_val(52);
    constexpr uint32_t im2col_rm_depth_rows = get_compile_time_arg_val(53);
    constexpr uint32_t c_in = get_compile_time_arg_val(54);  // this core's channel slice
    constexpr bool tile_input = get_compile_time_arg_val(55) != 0;
    constexpr bool zero_im2col_once = get_compile_time_arg_val(56) != 0;
    // Shared-L1 N-slice walk: cb_combine_slots shares the slab's L1, so the credits wait for the reader's slab_dead.
    constexpr bool alias_slab_combine = get_compile_time_arg_val(57) != 0;
    constexpr uint32_t slab_dead_sem_id = get_compile_time_arg_val(58);
    // Split TILE fill (see the file comment).
    constexpr bool tile_fill_split = get_compile_time_arg_val(59) != 0;
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(60);
    constexpr uint32_t slab_tile_rows = get_compile_time_arg_val(61);
    constexpr uint32_t untilize_batch_rows = get_compile_time_arg_val(62);
    constexpr uint32_t input_tile_depth_batches = get_compile_time_arg_val(63);
    // Ring slab (see the reader): every M-block streams the input tile-rows [max(previous hi, lo), hi) of its window.
    constexpr bool ring_slab = get_compile_time_arg_val(64) != 0;
    // Split TILE fill under a K split: the core's channel slice of input tile-row r is c_in_p / 32 pages from page
    // r * fill_row_page_stride + my_col * c_in_p / 32; with fill_zero_lanes only the slice's real tiles are read and
    // its padding tile columns are read from the reader's zero block.
    constexpr bool fill_channel_slice = get_compile_time_arg_val(65) != 0;
    constexpr uint32_t fill_row_page_stride = get_compile_time_arg_val(66);
    constexpr bool fill_zero_lanes = get_compile_time_arg_val(67) != 0;
    // The reader's zero block (tap-major, zeroed before the reader hands anything over): a short slice's padding tiles.
    constexpr uint32_t cb_zero_src = get_compile_time_arg_val(68);
    // Higher-shard batches the writer keeps in flight (own transaction id each).
    constexpr uint32_t fill_writer_window = get_compile_time_arg_val(69);
    static_assert(fill_writer_window >= 1 && fill_writer_window <= 8, "writer fill transaction ids 8..15");
    static_assert(fill_writer_window == 1 || !ring_slab, "the ring slab's writer fill is one batch at a time");
    // Weight servers (see the file comment): on, the server's ack semaphore, the first of weight_depth flag semaphores.
    constexpr bool weight_servers = get_compile_time_arg_val(70) != 0;
    constexpr uint32_t server_ack_sem_id = get_compile_time_arg_val(71);
    constexpr uint32_t server_flag_sem_id = get_compile_time_arg_val(72);
    // K split on more grid rows than the output has shard rows (see the file comment).
    constexpr bool remote_output = get_compile_time_arg_val(73) != 0;
    static_assert(!remote_output || (k_split && output_sharded));
    constexpr auto weight_args = TensorAccessorArgs<74>();
    [[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto output_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto weight_mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("weight_mcast_ct_offset"),
        get_named_compile_time_arg_val("weight_mcast_rt_offset")>();

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t output_addr = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t m_tile_start = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t actual_m_tiles = get_arg_val<uint32_t>(4);
    const uint32_t num_m_blocks_this_core = get_arg_val<uint32_t>(5);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(6);
    const uint32_t weight_k_tile_start = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t out_n_tile_start = get_arg_val<uint32_t>(8);
    const uint32_t out_valid_tiles = get_arg_val<uint32_t>(9);
    [[maybe_unused]] const uint32_t my_col = get_arg_val<uint32_t>(10);
    [[maybe_unused]] const uint32_t input_addr = get_arg_val<uint32_t>(11);
    [[maybe_unused]] const uint32_t my_shard = get_arg_val<uint32_t>(12);
    constexpr uint32_t row_peer_coords_arg = 13;  // (x, y) of grid columns 0..k_cols-1 of this row
    // Split TILE fill on this core (runtime: the kernel is shared with the cores that keep the reader-only fill).
    constexpr uint32_t split_flag_arg = row_peer_coords_arg + (k_split ? 2 * k_cols : 0);
    [[maybe_unused]] const bool split_here = tile_fill_split && get_arg_val<uint32_t>(split_flag_arg) != 0;
    // Weight servers: 1 on a receiver, then its server's (x, y).
    constexpr uint32_t server_arg = split_flag_arg + 1;
    [[maybe_unused]] const bool server_receiver = weight_servers && get_arg_val<uint32_t>(server_arg) != 0;

    constexpr uint32_t weight_k_block_tiles = block_k_tiles * block_n_tiles;
    constexpr uint32_t row_group_tiles = out_subblock_h * block_n_tiles;
    constexpr uint32_t row_groups_per_block = block_m_tiles / out_subblock_h;
    const uint32_t weight_tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, weight_tile_bytes);

    // ---- prefetch_bias: the core's output columns; stays fronted in cb_bias for the whole kernel. Slots
    // past the last C_out tile (a short last N-slice) only feed output padding columns. With the split slab fill
    // it follows the first block's slab reads, which the reader waits on before it can gather.
    auto prefetch_bias = [&]() {
        if constexpr (has_bias) {
            MaybeDeviceZoneScope("writer_bias");
            const uint32_t bias_tile_bytes = get_tile_size(cb_bias);
            const auto bias = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);
            cb_reserve_back(cb_bias, out_n_tiles);
            uint32_t l1 = get_write_ptr(cb_bias);
            for (uint32_t n = 0; n < out_valid_tiles; ++n, l1 += bias_tile_bytes) {
                noc_async_read_page(out_n_tile_start + n, bias, l1);
            }
            noc_async_read_barrier();
            cb_push_back(cb_bias, out_n_tiles);
        }
    };
#ifdef CONV2D_3X3_KW_FOLD
    // kw-fold adds the bias in its final pass: read it late (below), so every core reading the same bias pages does
    // not hold the weight stream behind it.
    constexpr bool kw_fold = true;
    // Sub-rounds before the first final pass (kw_fold.hpp): one n-group's three columns, or all of the M-block's.
    constexpr uint32_t kwf_first_pass_rounds = KWF_NG_OUTER ? 3 : num_k_blocks;
#else
    constexpr bool kw_fold = false;
    [[maybe_unused]] constexpr uint32_t kwf_first_pass_rounds = 0;
#endif
    // With the polling split TILE fill the bias reads are issued later, serving the first block's hand-overs.
    if constexpr (!slab_split && !tile_fill_split && !weight_servers && !kw_fold) {
        prefetch_bias();
    }

    // ---- Split TILE fill, writer half: the higher shards' input tile-rows of every cb_input_tiles batch the reader
    // hands over (fill_go), on NoC1, into the batch's ring slot; fill_done counts the filled batches. Every reader
    // reserve and push is one batch and the CB holds input_tile_depth_batches of them, so batch g of the kernel sits
    // at slot g % depth. Served wherever the writer would otherwise wait: compute untilizes a block before it
    // multiplies it, so the weight round two past the block's first waits on this block's fill.
    constexpr uint32_t fill_c_in_tiles = c_in_p / 32;
    constexpr uint32_t fill_batches_per_block = tile_fill_split ? slab_tile_rows / untilize_batch_rows : 1;
    constexpr uint32_t fill_row_bytes = fill_c_in_tiles * input_page_bytes;
    constexpr uint32_t fill_batch_bytes = untilize_batch_rows * fill_row_bytes;
    [[maybe_unused]] const uint32_t fill_ring_base = tile_fill_split ? get_write_ptr(cb_input_tiles) : 0;
    [[maybe_unused]] const uint32_t fill_hi_row = (my_shard + 1) * input_shard_rows;
    [[maybe_unused]] uint32_t fill_m = 0;  // cursor: block and batch of the next batch to look at
    [[maybe_unused]] uint32_t fill_b = 0;
    [[maybe_unused]] uint32_t fill_plan_m = 0xFFFFFFFF;
    [[maybe_unused]] uint32_t fill_served = 0;
    [[maybe_unused]] conv2d_3x3::TileSlabPlan fill_plan;
    // Ring slab: the cursor block's first streamed row, its row end and batch count, the kernel batches before it, and
    // the previous block's window end.
    [[maybe_unused]] uint32_t ring_push_lo = 0;
    [[maybe_unused]] uint32_t ring_hi = 0;
    [[maybe_unused]] uint32_t ring_batches = 0;
    [[maybe_unused]] uint32_t ring_batch_base = 0;
    [[maybe_unused]] int32_t ring_prev_hi = -1;
    conv2d_3x3::TaggedReads<false> fill_reads;
    constexpr uint32_t TRID_FILL = 6;
    [[maybe_unused]] const uint64_t fill_zero_noc =
        fill_zero_lanes ? get_noc_addr(get_write_ptr(cb_zero_src), fill_reads.noc.get_noc_id()) : 0;
    // Moves the cursor to the next batch holding higher shards' rows; false past the last block.
    [[maybe_unused]] auto fill_next = [&]() -> bool {
        if constexpr (ring_slab) {
            while (fill_m < num_m_blocks_this_core) {
                if (fill_plan_m != fill_m) {
                    const uint32_t block_tiles_done = fill_m * block_m_tiles;
                    const uint32_t block_valid_tiles = actual_m_tiles - block_tiles_done < block_m_tiles
                                                           ? actual_m_tiles - block_tiles_done
                                                           : block_m_tiles;
                    const uint32_t first = (m_tile_start + block_tiles_done) * 32;
                    const uint32_t end_tiles = first + block_valid_tiles * 32;
                    const conv2d_3x3::RingWindow win =
                        conv2d_3x3::ring_window<input_width, input_height * input_width, dilation>(
                            conv2d_3x3::
                                block_footprint<output_width, input_width, stride, padding, dilation, band_last_tap>(
                                    first, end_tiles < num_out_px ? end_tiles : num_out_px));
                    const int32_t keep_lo = conv2d_3x3::ring_keep_lo<
                        output_width,
                        input_width,
                        stride,
                        padding,
                        dilation,
                        band_last_tap,
                        input_height * input_width,
                        num_out_px,
                        block_m_tiles>(m_tile_start, actual_m_tiles, fill_m, num_m_blocks_this_core);
                    ring_batch_base += ring_batches;
                    const int32_t push_lo = ring_prev_hi > keep_lo ? ring_prev_hi : keep_lo;
                    ring_push_lo = static_cast<uint32_t>(push_lo);
                    ring_hi = static_cast<uint32_t>(win.hi > push_lo ? win.hi : push_lo);
                    ring_batches = (ring_hi - ring_push_lo + untilize_batch_rows - 1) / untilize_batch_rows;
                    ring_prev_hi = win.hi > ring_prev_hi ? win.hi : ring_prev_hi;
                    fill_plan_m = fill_m;
                }
                if (fill_b < ring_batches) {
                    const uint32_t row_end = ring_push_lo + (fill_b + 1) * untilize_batch_rows;
                    if ((row_end < ring_hi ? row_end : ring_hi) > fill_hi_row) {
                        return true;
                    }
                    ++fill_b;
                }
                if (fill_b >= ring_batches) {
                    fill_b = 0;
                    ++fill_m;
                }
            }
            return false;
        }
        while (fill_m < num_m_blocks_this_core) {
            if (fill_plan_m != fill_m) {
                const uint32_t block_tiles_done = fill_m * block_m_tiles;
                const uint32_t block_valid_tiles = actual_m_tiles - block_tiles_done < block_m_tiles
                                                       ? actual_m_tiles - block_tiles_done
                                                       : block_m_tiles;
                const uint32_t first = (m_tile_start + block_tiles_done) * 32;
                const uint32_t end_tiles = first + block_valid_tiles * 32;
                int32_t unused_x_base[3];
                conv2d_3x3::tile_slab_plan<input_width, input_height * input_width, dilation, c_in_p / 32>(
                    conv2d_3x3::block_footprint<output_width, input_width, stride, padding, dilation, band_last_tap>(
                        first, end_tiles < num_out_px ? end_tiles : num_out_px),
                    unused_x_base,
                    fill_plan);
                fill_plan_m = fill_m;
            }
            if (conv2d_3x3::tile_batch_has_rows_from<untilize_batch_rows>(fill_plan, fill_b, fill_hi_row)) {
                return true;
            }
            if (++fill_b == fill_batches_per_block) {
                fill_b = 0;
                ++fill_m;
            }
        }
        return false;
    };
    // Compacted slab: the cursor batch's higher-shard rows, issued under trid; advances the cursor.
    [[maybe_unused]] auto fill_issue = [&](uint32_t trid) {
        MaybeDeviceZoneScope("writer_fill");
        const auto input = TensorAccessor(input_args, input_addr, input_page_bytes);
        const uint32_t slot =
            fill_ring_base + ((fill_m * fill_batches_per_block + fill_b) % input_tile_depth_batches) * fill_batch_bytes;
        fill_reads.begin(trid);
        conv2d_3x3::tile_batch_rows<untilize_batch_rows>(
            fill_plan, fill_b, [&](uint32_t row0, uint32_t rows, uint32_t batch_row) {
                const uint32_t lo = row0 > fill_hi_row ? row0 : fill_hi_row;
                if (lo >= row0 + rows) {
                    return;
                }
                uint32_t l1 = slot + (batch_row + lo - row0) * fill_row_bytes;
                if constexpr (fill_channel_slice) {
                    // One owner shard stores the slice's rows back to back, fill_row_bytes apart.
                    constexpr uint32_t shard_rows = input_shard_rows > 0 ? input_shard_rows : 1;
                    constexpr uint32_t real_row_bytes = (c_in / 32) * input_page_bytes;
                    const uint32_t slice_page_offset = my_col * fill_c_in_tiles;
                    for (uint32_t row = lo, end_row = row0 + rows; row < end_row;) {
                        const uint32_t shard_end = (row / shard_rows + 1) * shard_rows;
                        const uint32_t run_end = end_row < shard_end ? end_row : shard_end;
                        const uint64_t src = input.get_noc_addr(row * fill_row_page_stride + slice_page_offset);
                        if constexpr (fill_zero_lanes) {
                            for (uint32_t i = 0; i < run_end - row; ++i) {
                                fill_reads.read(src + i * fill_row_bytes, l1 + i * fill_row_bytes, real_row_bytes);
                                fill_reads.read(
                                    fill_zero_noc,
                                    l1 + i * fill_row_bytes + real_row_bytes,
                                    fill_row_bytes - real_row_bytes);
                            }
                        } else {
                            fill_reads.read(src, l1, (run_end - row) * fill_row_bytes);
                        }
                        l1 += (run_end - row) * fill_row_bytes;
                        row = run_end;
                    }
                    return;
                }
                for (uint32_t page = lo * fill_c_in_tiles, end_page = (row0 + rows) * fill_c_in_tiles;
                     page < end_page;) {
                    const uint32_t run = input.num_contiguous_pages(page, end_page);
                    fill_reads.read(input.get_noc_addr(page), l1, run * input_page_bytes);
                    page += run;
                    l1 += run * input_page_bytes;
                }
            });
        dataflow_kernel_lib::set_read_trid(fill_reads.noc, 0);
        if (++fill_b == fill_batches_per_block) {
            fill_b = 0;
            ++fill_m;
        }
    };
    // Ring slab: one handed-over batch, read and counted.
    [[maybe_unused]] auto fill_one = [&]() {
        MaybeDeviceZoneScope("writer_fill");
        if constexpr (ring_slab) {
            const auto input = TensorAccessor(input_args, input_addr, input_page_bytes);
            const uint32_t slot =
                fill_ring_base + ((ring_batch_base + fill_b) % input_tile_depth_batches) * fill_batch_bytes;
            const uint32_t row0 = ring_push_lo + fill_b * untilize_batch_rows;
            const uint32_t row_end_b = row0 + untilize_batch_rows;
            const uint32_t row_end = row_end_b < ring_hi ? row_end_b : ring_hi;
            const uint32_t lo = row0 > fill_hi_row ? row0 : fill_hi_row;
            fill_reads.begin(TRID_FILL);
            uint32_t l1 = slot + (lo - row0) * fill_row_bytes;
            for (uint32_t page = lo * fill_c_in_tiles, end_page = row_end * fill_c_in_tiles; page < end_page;) {
                const uint32_t run = input.num_contiguous_pages(page, end_page);
                fill_reads.read(input.get_noc_addr(page), l1, run * input_page_bytes);
                page += run;
                l1 += run * input_page_bytes;
            }
            fill_reads.wait(TRID_FILL);
            dataflow_kernel_lib::set_read_trid(fill_reads.noc, 0);
            Semaphore<>(slab_fwd_done_sem_id).up(1);
            ++fill_served;
            if (++fill_b >= ring_batches) {
                fill_b = 0;
                ++fill_m;
            }
        }
    };
    // Windowed writer fill: batches [fill_served, fill_issued) are in flight, batch j under trid
    // TRID_WFILL_BASE + j % fill_writer_window.
    constexpr uint32_t TRID_WFILL_BASE = 8;
    [[maybe_unused]] uint32_t fill_issued = 0;
    // Every batch the reader has handed over so far (windowed: retire the landed ones, issue up to the window).
    [[maybe_unused]] auto serve_fill = [&]() {
        if constexpr (tile_fill_split) {
            if constexpr (!ring_slab) {
                while (fill_served < fill_issued &&
                       ncrisc_noc_read_with_transaction_id_flushed(
                           fill_reads.noc.get_noc_id(), TRID_WFILL_BASE + fill_served % fill_writer_window)) {
                    Semaphore<>(slab_fwd_done_sem_id).up(1);
                    ++fill_served;
                }
                while (fill_issued - fill_served < fill_writer_window &&
                       Semaphore<>(slab_free_sem_id).value() > fill_issued) {
                    fill_next();
                    fill_issue(TRID_WFILL_BASE + fill_issued % fill_writer_window);
                    ++fill_issued;
                }
            } else {
                while (Semaphore<>(slab_free_sem_id).value() > fill_served) {
                    fill_next();
                    fill_one();
                }
            }
        }
    };
    // Blocking waits of the K-split walk, serving the reader's fill hand-overs while they wait (split TILE fill).
    [[maybe_unused]] auto reserve_serving = [&](uint32_t cb, uint32_t pages) {
        if constexpr (tile_fill_split) {
            while (!cb_pages_reservable_at_back(cb, pages)) {
                serve_fill();
            }
        }
        cb_reserve_back(cb, pages);
    };
    [[maybe_unused]] auto wait_front_serving = [&](uint32_t cb, uint32_t pages) {
        if constexpr (tile_fill_split) {
            while (!cb_pages_available_at_front(cb, pages)) {
                serve_fill();
            }
        }
        cb_wait_front(cb, pages);
    };
    [[maybe_unused]] auto sem_wait_serving = [&](uint32_t sem_id, uint32_t count) {
        if constexpr (tile_fill_split) {
            while (Semaphore<>(sem_id).value() < count) {
                serve_fill();
            }
        }
        Semaphore<>(sem_id).wait_min(count);
    };
    // The rest of the kernel's batches, waiting for each hand-over.
    [[maybe_unused]] auto drain_fill = [&]() {
        if constexpr (tile_fill_split) {
            if (!split_here) {
                return;
            }
            if constexpr (!ring_slab) {
                while (true) {
                    serve_fill();
                    if (fill_served == fill_issued && !fill_next()) {
                        break;
                    }
                }
            } else {
                while (fill_next()) {
                    Semaphore<>(slab_free_sem_id).wait_min(fill_served + 1);
                    fill_one();
                }
            }
        }
    };

    // ---- stream_weight_blocks: K-block tiles k-major, n-fastest (matmul in1 layout).
    const uint32_t total_k_blocks = num_m_blocks_this_core * num_k_blocks;
    uint32_t k_blocks_issued = 0;
    auto read_weight_k_block = [&](uint32_t l1) {
#ifdef CONV2D_3X3_KW_FOLD
        // kw-fold: column round kw holds W'[kh Ct + c, kw N + n] = prepared tile ((kh * 3 + kw) Ct + c, n),
        // (kh, c)-major, n-fastest (block_k_tiles = Ct, block_n_tiles = 3 N: the same round size). Rounds go kw = 1,
        // 0, 2 (kw_fold.hpp KW_ROUND_ORDER: Q_1 is the overwrite the other two accumulate onto).
        // Sub-rounds (kw_fold.hpp): one column kw and one n-group of sbw = block_n_tiles / 3 output column tiles,
        // (kh, c, n) order; KWF_NG_OUTER: sub-round s = (n-group s / 3, kw = order[s % 3]), else (kw = order[s / NG],
        // n-group s % NG).
        constexpr uint32_t kw_round_order[3] = {1, 0, 2};
        constexpr uint32_t kwf_sbw = block_n_tiles / 3;
        constexpr uint32_t kwf_ng = tensor_n_tiles / kwf_sbw;
        const uint32_t s = k_blocks_issued % num_k_blocks;
        const uint32_t kw = kw_round_order[KWF_NG_OUTER ? s % 3 : s / kwf_ng];
        const uint32_t n0 = (KWF_NG_OUTER ? s / 3 : s % kwf_ng) * kwf_sbw;
        for (uint32_t kh = 0; kh < 3; ++kh) {
            for (uint32_t c = 0; c < block_k_tiles; ++c) {
                const uint32_t row_page = ((kh * 3 + kw) * block_k_tiles + c) * tensor_n_tiles + n0;
                for (uint32_t n = 0; n < kwf_sbw; ++n, l1 += weight_tile_bytes) {
                    noc_async_read_page(row_page + n, weights, l1);
                }
            }
        }
#else
        const uint32_t k_tile0 = (k_blocks_issued % num_k_blocks) * block_k_tiles;
        for (uint32_t k = 0; k < block_k_tiles; ++k) {
            const uint32_t row_page = (weight_k_tile_start + k_tile0 + k) * tensor_n_tiles + n_tile_start;
            for (uint32_t n = 0; n < block_n_tiles; ++n, l1 += weight_tile_bytes) {
                noc_async_read_page(row_page + n, weights, l1);
            }
        }
#endif
        noc_async_read_barrier();
    };

    Noc noc;
    auto weight_sender = weight_mcast.optional_sender(noc);
    auto weight_receiver = weight_mcast.optional_receiver(noc);
    // Raw weight-channel receive, bypassing ReceiverPipe::receive (which blocks on the data-ready signal and has no
    // poll / idle-callback form): the same protocol for a fixed-sender hardware multicast (ack the sender, then wait
    // for the round's data-ready signal), serving the reader's fill hand-overs while it waits, so one slow core's
    // weight round does not hold this core's untilize. Measured on sem res2.fuse_conv.1 @ LoFi: 138.6 us with the
    // blocking receive vs 129.8 us polled (split TILE fill on both).
    [[maybe_unused]] auto receive_serving = [&](auto& wm, uint32_t round) {
        using WM = std::decay_t<decltype(wm)>;
        // Dependent on WM, so a kernel without the split never instantiates the sender-coordinate reads.
        if constexpr (WM::active && tile_fill_split) {
            static_assert(WM::transfer_mode == dataflow_kernel_lib::TransferMode::Multicast && !WM::rotating);
            constexpr uint32_t ct_base = get_named_compile_time_arg_val("weight_mcast_ct_offset");
            Semaphore<> data_ready(
                dataflow_kernel_lib::detail::
                    positional_mcast_semaphore<ct_base, dataflow_kernel_lib::mcast_wire::DATA_READY>());
            if constexpr (WM::pre_handshake) {
                Semaphore<>(dataflow_kernel_lib::detail::
                                positional_mcast_semaphore<ct_base, dataflow_kernel_lib::mcast_wire::CONSUMER_READY>())
                    .up(noc, wm.sender_x(), wm.sender_y(), 1);
            }
            if constexpr (WM::signal == dataflow_kernel_lib::DataReadySignal::Counter) {
                while (data_ready.value() < round + 1) {
                    serve_fill();
                }
            } else {
                while (data_ready.value() != VALID) {
                    serve_fill();
                }
                data_ready.set(INVALID);
            }
        }
    };
    // prefetch_bias with the reads tagged and their completion polled while serving hand-overs: the bias is one DRAM
    // page per output tile column that every core reads at kernel start, so it can take microseconds.
    [[maybe_unused]] auto prefetch_bias_serving = [&]() {
        if constexpr (has_bias) {
            MaybeDeviceZoneScope("writer_bias");
            constexpr uint32_t TRID_BIAS = 7;
            const uint32_t bias_tile_bytes = get_tile_size(cb_bias);
            const auto bias = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);
            cb_reserve_back(cb_bias, out_n_tiles);
            uint32_t l1 = get_write_ptr(cb_bias);
            conv2d_3x3::TaggedReads<false> bias_reads;
            bias_reads.begin(TRID_BIAS);
            for (uint32_t n = 0; n < out_valid_tiles; ++n, l1 += bias_tile_bytes) {
                bias_reads.read(bias.get_noc_addr(out_n_tile_start + n), l1, bias_tile_bytes);
            }
            dataflow_kernel_lib::set_read_trid(bias_reads.noc, 0);
            while (!ncrisc_noc_read_with_transaction_id_flushed(bias_reads.noc.get_noc_id(), TRID_BIAS)) {
                serve_fill();
            }
            bias_reads.wait(TRID_BIAS);
            cb_push_back(cb_bias, out_n_tiles);
        }
    };
    if constexpr (!slab_split && tile_fill_split && !weight_servers && !kw_fold) {
        prefetch_bias_serving();
    }
#ifdef CONV2D_3X3_KW_FOLD
    // kw-fold adds the bias in its final pass (srcB rows the reader decodes from cb_bias, conv2d_3x3_kw_fold.hpp).
    // With weight servers it takes the late reads below (issued after round 0, completed after the block's last round).
    // Without, it is read after the first M-block's weight rounds, so the cores reading the same bias pages do not hold
    // the weight stream behind it (stem.conv2 / conv3, in-grid sender: 40.6 -> 37.6 / 60.0 -> 53.3 us). A blocking
    // read there on server receivers put ~2.8 us of bias (~100 cores on the same DRAM pages) between the last round
    // and the final pass: res2 13.9 -> 14.8, center_head.1 12.1 -> 12.6 us.
    [[maybe_unused]] auto kwf_bias_read = [&]() {
        if constexpr (has_bias) {
            if constexpr (tile_fill_split) {
                prefetch_bias_serving();
            } else {
                prefetch_bias();
            }
        }
    };
#endif
    // Weight servers: the bias reads go out now and complete after the first weight round (see the file comment).
    // kw-fold needs the bias only in its final pass, and its first matmul waits for round 0: its reads go out once
    // round 0 landed (~100 cores reading the same bias pages slowed round 0 by ~1 us when issued at kernel start) and
    // are pushed as soon as they land, polled inside the round waits; the first M-block's last round blocks on them.
    constexpr uint32_t TRID_BIAS_LATE = 7;
    conv2d_3x3::TaggedReads<false> late_bias_reads;
    [[maybe_unused]] bool late_bias_pending = false;
    [[maybe_unused]] auto issue_late_bias = [&]() {
        if constexpr (weight_servers && has_bias && !slab_split) {
            const uint32_t bias_tile_bytes = get_tile_size(cb_bias);
            const auto bias = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);
            cb_reserve_back(cb_bias, out_n_tiles);
            uint32_t l1 = get_write_ptr(cb_bias);
            late_bias_reads.begin(TRID_BIAS_LATE);
            for (uint32_t n = 0; n < out_valid_tiles; ++n, l1 += bias_tile_bytes) {
                late_bias_reads.read(bias.get_noc_addr(out_n_tile_start + n), l1, bias_tile_bytes);
            }
            dataflow_kernel_lib::set_read_trid(late_bias_reads.noc, 0);
            late_bias_pending = true;
        }
    };
    if constexpr (!kw_fold) {
        issue_late_bias();
    }
    // kw-fold: push the late bias as soon as it landed, from inside the weight waits (the reader decodes it into the
    // final pass's fp32 rows at ~1.3 us per tile, so it must not wait for the block's last round).
    [[maybe_unused]] auto try_finish_late_bias = [&]() {
        if (late_bias_pending &&
            ncrisc_noc_read_with_transaction_id_flushed(late_bias_reads.noc.get_noc_id(), TRID_BIAS_LATE)) {
            late_bias_reads.wait(TRID_BIAS_LATE);
            cb_push_back(cb_bias, out_n_tiles);
            late_bias_pending = false;
        }
    };
    [[maybe_unused]] auto finish_late_bias = [&]() {
        if (late_bias_pending) {
            MaybeDeviceZoneScope("writer_bias");
            while (!ncrisc_noc_read_with_transaction_id_flushed(late_bias_reads.noc.get_noc_id(), TRID_BIAS_LATE)) {
                serve_fill();
            }
            late_bias_reads.wait(TRID_BIAS_LATE);
            cb_push_back(cb_bias, out_n_tiles);
            late_bias_pending = false;
        }
    };
    // Weight servers: the receiver's ack to its server (one per round that reuses a slot, plus one at kernel start).
    [[maybe_unused]] uint64_t server_ack_noc = 0;
    if (server_receiver) {
        const uint32_t server_x = get_arg_val<uint32_t>(server_arg + 1);
        const uint32_t server_y = get_arg_val<uint32_t>(server_arg + 2);
        server_ack_noc = get_noc_addr(server_x, server_y, get_semaphore(server_ack_sem_id));
        noc_semaphore_inc(server_ack_noc, 1);
    }
    auto issue_weight_k_block = [&]() {
        {
            // Back-pressure from compute's matmul.
            MaybeDeviceZoneScope("writer_weight_reserve");
            if constexpr (tile_fill_split) {
                // Compute may need this core's fill before it frees the slot.
                while (!cb_pages_reservable_at_back(cb_weights, weight_k_block_tiles)) {
                    serve_fill();
                }
            }
            cb_reserve_back(cb_weights, weight_k_block_tiles);
        }
        const uint32_t l1 = get_write_ptr(cb_weights);
        // DRAM read (+ multicast) on a sender or own-stream core; the multicast wait on a receiver.
        MaybeDeviceZoneScope("writer_weight_fetch");
        if (server_receiver) {
            // Weight server round (see the file comment); serves the fill hand-overs while it waits.
            if (k_blocks_issued >= nslice_weight_depth) {
                noc_semaphore_inc(server_ack_noc, 1);
            }
            volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(server_flag_sem_id + k_blocks_issued % nslice_weight_depth));
            while (true) {
                invalidate_l1_cache();
                if (*flag == VALID) {
                    break;
                }
                serve_fill();
                if constexpr (kw_fold) {
                    try_finish_late_bias();
                }
            }
            *flag = INVALID;
        } else if (weight_sender && weight_mcast.should_send(k_blocks_issued)) {
            read_weight_k_block(l1);
            weight_sender->send(l1, l1, weight_k_block_tiles * weight_tile_bytes);
        } else if (weight_receiver) {
            if constexpr (tile_fill_split) {
                receive_serving(weight_mcast, k_blocks_issued);
            } else {
                weight_receiver->receive(k_blocks_issued);
            }
        } else {
            read_weight_k_block(l1);
        }
        cb_push_back(cb_weights, weight_k_block_tiles);
        ++k_blocks_issued;
        if constexpr (weight_servers) {
            if constexpr (kw_fold) {
                if (k_blocks_issued == 1) {
                    issue_late_bias();
                }
                if (k_blocks_issued == kwf_first_pass_rounds) {
                    finish_late_bias();
                } else if (k_blocks_issued > 1) {
                    try_finish_late_bias();
                }
            } else {
                finish_late_bias();
            }
        }
    };

    // ---- R4 row reduce-scatter (see the file comment).
    constexpr uint32_t partials_tiles = block_m_tiles * block_n_tiles;
    constexpr uint32_t slot_tiles = block_m_tiles * out_n_tiles;
    constexpr uint32_t num_owning_cols = (tensor_n_tiles + out_n_tiles - 1) / out_n_tiles;
    Semaphore<> combine_credit(combine_credit_sem_id);
    Semaphore<> combine_arrival(combine_arrival_sem_id);
    Semaphore<> combine_send_go(combine_send_go_sem_id);
    Semaphore<> combine_send_done(combine_send_done_sem_id);
    [[maybe_unused]] const uint32_t combine_base = get_write_ptr(cb_combine_slots);
    [[maybe_unused]] const uint32_t partials_base = get_read_ptr(cb_partials);
    // Columns this RISC sends to: all, or its own and the lower ones when the reader takes the rest.
    [[maybe_unused]] const uint32_t writer_cols_end =
        combine_split_nocs ? (my_col + 1 < num_owning_cols ? my_col + 1 : num_owning_cols) : num_owning_cols;
    auto peer_x = [](uint32_t col) { return get_arg_val<uint32_t>(row_peer_coords_arg + 2 * col); };
    auto peer_y = [](uint32_t col) { return get_arg_val<uint32_t>(row_peer_coords_arg + 2 * col + 1); };
    [[maybe_unused]] auto combine_round = [&](uint32_t m_block_idx) {
        if (out_valid_tiles > 0) {
            cb_reserve_back(cb_combine_slots, k_cols * slot_tiles);
            for (uint32_t col = 0; col < k_cols; ++col) {
                combine_credit.up(noc, peer_x(col), peer_y(col), 1);
            }
        }
        {
            MaybeDeviceZoneScope("writer_combine_wait_partials");
            cb_wait_front(cb_partials, partials_tiles);
        }
        {
            // Every owning column's slots are free.
            MaybeDeviceZoneScope("writer_combine_wait_credit");
            combine_credit.wait_min((m_block_idx + 1) * num_owning_cols);
        }
        if constexpr (combine_split_nocs) {
            combine_send_go.up(1);
        }
        {
            // This RISC's half of the scatter, then the reader's half (send_done).
            MaybeDeviceZoneScope("writer_combine_send");
            conv2d_3x3::scatter_partial_slices<block_m_tiles, block_n_tiles, out_n_tiles, tensor_n_tiles>(
                partials_base,
                combine_base,
                get_tile_size(cb_partials),
                my_col,
                0,
                writer_cols_end,
                row_peer_coords_arg);
            noc_async_write_barrier();
            if constexpr (combine_split_nocs) {
                combine_send_done.wait_min(m_block_idx + 1);
            }
        }
        for (uint32_t col = 0; col < num_owning_cols; ++col) {
            combine_arrival.up(noc, peer_x(col), peer_y(col), 1);
        }
        cb_pop_front(cb_partials, partials_tiles);
        if (out_valid_tiles > 0) {
            {
                // The row's slowest sender.
                MaybeDeviceZoneScope("writer_combine_wait_arrivals");
                combine_arrival.wait_min((m_block_idx + 1) * k_cols);
            }
            cb_push_back(cb_combine_slots, k_cols * slot_tiles);
        }
    };

    // ---- N-slice-outer K split (see the file comment).
    static_assert(!nslice_outer || (k_split && output_sharded));
    constexpr uint32_t nslice_k_tiles = block_k_tiles * num_k_blocks;
    constexpr uint32_t slice_weight_tiles = nslice_k_tiles * out_n_tiles;
    // Rounds issued before the split row gather's serving and the credit.
    constexpr uint32_t nslice_weight_lookahead = 2;
    static_assert(!nslice_outer || nslice_weight_depth == 1 || nslice_weight_depth >= nslice_weight_lookahead + 2);
    // Every participant pushes one slice per round from the CB base, so slot round % depth is the same L1 address on
    // every core of the column.
    const uint32_t weights_base = get_write_ptr(cb_weights);
    conv2d_3x3::TaggedReads<false> weight_reads;
    constexpr uint32_t TRID_WEIGHTS = 2;  // 2 and 3 alternate; the face gather uses 1
    uint32_t prefetched_round = 0xFFFFFFFF;
    [[maybe_unused]] auto start_weight_reads = [&](uint32_t round, uint32_t y) {
        weight_reads.begin(TRID_WEIGHTS + (round & 1));
        uint32_t l1 = weights_base + (round % nslice_weight_depth) * slice_weight_tiles * weight_tile_bytes;
        const uint32_t n0 = y * out_n_tiles;
        const uint32_t valid = tensor_n_tiles - n0 < out_n_tiles ? tensor_n_tiles - n0 : out_n_tiles;
        for (uint32_t k = 0; k < nslice_k_tiles; ++k) {
            const uint32_t row_page = (weight_k_tile_start + k) * tensor_n_tiles + n0;
            for (uint32_t n = 0; n < valid; ++n) {
                weight_reads.read(weights.get_noc_addr(row_page + n), l1 + n * weight_tile_bytes, weight_tile_bytes);
            }
            l1 += out_n_tiles * weight_tile_bytes;
        }
        dataflow_kernel_lib::set_read_trid(weight_reads.noc, 0);
    };
    // Weights of slice y; next_y is the M-block's next slice, or k_cols after the last.
    [[maybe_unused]] auto issue_weight_slice = [&](uint32_t y, uint32_t next_y) {
        const uint32_t round = k_blocks_issued;
        MaybeDeviceZoneScope("writer_weight_fetch");
        if (!weight_receiver || (weight_sender && weight_mcast.should_send(round))) {
            if (prefetched_round != round) {
                reserve_serving(cb_weights, slice_weight_tiles);
                start_weight_reads(round, y);
            }
            if (nslice_weight_depth > 1 && next_y < k_cols) {
                reserve_serving(cb_weights, 2 * slice_weight_tiles);
                start_weight_reads(round + 1, next_y);
                prefetched_round = round + 1;
            }
            if constexpr (tile_fill_split) {
                while (!ncrisc_noc_read_with_transaction_id_flushed(
                    weight_reads.noc.get_noc_id(), TRID_WEIGHTS + (round & 1))) {
                    serve_fill();
                }
            }
            weight_reads.wait(TRID_WEIGHTS + (round & 1));
            if (weight_sender && weight_mcast.should_send(round)) {
                const uint32_t l1 =
                    weights_base + (round % nslice_weight_depth) * slice_weight_tiles * weight_tile_bytes;
                weight_sender->send(l1, l1, slice_weight_tiles * weight_tile_bytes);
            }
        } else {
            reserve_serving(cb_weights, slice_weight_tiles);
            if constexpr (tile_fill_split) {
                receive_serving(weight_mcast, round);
            } else {
                weight_receiver->receive(round);
            }
        }
        cb_push_back(cb_weights, slice_weight_tiles);
        ++k_blocks_issued;
    };
    // before_sends runs once the first weight rounds are out, before the credit (the split row gather's serving).
    [[maybe_unused]] auto nslice_round = [&](uint32_t m_block_idx, auto&& before_sends) {
        // One credit per M-block to column my_col - 1, once this core's slots are free: its gather landed (shared-L1
        // walk) and compute popped the previous M-block's slots (it pushes and pops the whole CB once per M-block,
        // so its acked count is then m_block_idx whole CBs).
        auto give_credit = [&]() {
            MaybeDeviceZoneScope("writer_ring_credit");
            const uintptr_t acked_ptr = (uintptr_t)get_cb_tiles_acked_ptr(cb_combine_slots);
            const uint16_t acked_target = static_cast<uint16_t>(m_block_idx * k_cols * slot_tiles);
            while (static_cast<uint16_t>(reg_read(acked_ptr)) != acked_target) {
                serve_fill();
                invalidate_l1_cache();
            }
            const uint32_t prev_col = my_col == 0 ? k_cols - 1 : my_col - 1;
            combine_credit.up(noc, peer_x(prev_col), peer_y(prev_col), 1);
        };
        uint32_t order[k_cols];
        uint32_t num_slices = 0;
        for (uint32_t j = 0; j < k_cols; ++j) {
            const uint32_t y = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
            if (y < num_owning_cols) {
                order[num_slices++] = y;
            }
        }
        uint32_t weights_issued = 0;
        auto issue_weights_through = [&](uint32_t end) {
            for (; weights_issued < end && weights_issued < num_slices; ++weights_issued) {
                issue_weight_slice(
                    order[weights_issued], weights_issued + 1 < num_slices ? order[weights_issued + 1] : k_cols);
            }
        };
        issue_weights_through(1 + nslice_weight_lookahead);
        before_sends();
        if constexpr (alias_slab_combine) {
            // The combine slots are the slab's L1: column my_col - 1 may write them only once this core's gather
            // landed.
            MaybeDeviceZoneScope("writer_slab_dead_wait");
            sem_wait_serving(slab_dead_sem_id, m_block_idx + 1);
        }
        give_credit();
        issue_weights_through(num_slices);
    };

    // ---- fill_halo_slab, higher-shard half, and the trailing tile-rows of the face-direct gather (see the file
    // comment).
    const uint32_t slab_l1 = get_write_ptr(cb_halo_slab);
    int32_t slab_x_base[3];
    uint32_t o_first = 0;
    uint32_t o_end = 0;
    [[maybe_unused]] auto fill_forward_slab = [&](uint32_t m_block_idx) {
        static_assert(!slab_split || input_shard_rows > 0);
        MaybeDeviceZoneScope("writer_slab_fwd");
        // Compiled on every path; only the split fill runs it.
        constexpr uint32_t shard_rows = input_shard_rows > 0 ? input_shard_rows : 1;
        if (m_block_idx > 0) {
            Semaphore<>(slab_free_sem_id).wait_min(m_block_idx);
        }
        const auto input = TensorAccessor(input_args, input_addr, input_page_bytes);
        const uint32_t block_tiles_done = m_block_idx * block_m_tiles;
        const uint32_t block_valid_tiles =
            actual_m_tiles - block_tiles_done < block_m_tiles ? actual_m_tiles - block_tiles_done : block_m_tiles;
        o_first = (m_tile_start + block_tiles_done) * 32;
        const uint32_t o_end_tiles = o_first + block_valid_tiles * 32;
        o_end = o_end_tiles < num_out_px ? o_end_tiles : num_out_px;
        conv2d_3x3::slab_bands<input_width, input_height * input_width, dilation, band_stride_px>(
            conv2d_3x3::block_footprint<output_width, input_width, stride, padding, dilation, band_last_tap>(
                o_first, o_end),
            slab_x_base,
            [&](int32_t, int32_t x_lo, int32_t x_hi, int32_t v_minus_x) {
                if (x_lo >= x_hi) {
                    return;
                }
                uint32_t l1 = slab_l1 + x_lo * input_page_bytes;
                for (uint32_t page = x_lo + v_minus_x, end_page = x_hi + v_minus_x; page < end_page;) {
                    const uint32_t run = input.num_contiguous_pages(page, end_page);
                    if (page / shard_rows > my_shard) {
                        noc_async_read(input.get_noc_addr(page), l1, run * input_page_bytes);
                    }
                    page += run;
                    l1 += run * input_page_bytes;
                }
            });
        noc_async_read_barrier();
        Semaphore<>(slab_fwd_done_sem_id).up(1);
    };

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
    conv2d_3x3::TaggedReads<true> gather_reads;
    constexpr uint32_t TRID_GATHER = 1;
    // Every in0 K-block starts at the CB base: its capacity is one K-block.
    const uint32_t in0_l1 = get_write_ptr(cb_im2col_tiles);
    [[maybe_unused]] auto gather_odd_groups = [&](uint32_t k_block_seq) {
        {
            MaybeDeviceZoneScope("writer_gather_wait_go");
            Semaphore<>(gather_go_sem_id).wait_min(k_block_seq + 1);
        }
        MaybeDeviceZoneScope("writer_gather");
        const int32_t kh = (k_block_seq % num_k_blocks) * block_k_tiles / tensor_kkh_tiles;
        constexpr uint32_t in0_groups = (block_m_tiles + in0_stream_rows - 1) / in0_stream_rows;
        for (uint32_t group = 1; group < in0_groups; group += 2) {
            gather_reads.begin(TRID_GATHER);
            FaceGather::tile_rows(
                gather_reads,
                in0_l1,
                group * in0_stream_rows,
                (group + 1) * in0_stream_rows < block_m_tiles ? (group + 1) * in0_stream_rows : block_m_tiles,
                o_first,
                o_end,
                kh,
                slab_l1,
                slab_x_base);
            gather_reads.wait(TRID_GATHER);
            Semaphore<>(gather_done_sem_id).up(1);
        }
        dataflow_kernel_lib::set_read_trid(gather_reads.noc, 0);
    };

    // ---- Split row gather, writer half: the units the reader hands over, into the cb_im2col_rm slots it reserved
    // for them, on NoC1.
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
    using Units = conv2d_3x3::GatherUnits<nslice_outer, block_m_tiles, num_k_blocks>;  // row_split excludes face-direct
    conv2d_3x3::TaggedReads<false> row_reads;
    [[maybe_unused]] const uint64_t row_slab_noc = row_split ? get_noc_addr(slab_l1, row_reads.noc.get_noc_id()) : 0;
    [[maybe_unused]] const uint32_t im2col_rm_base = row_split ? get_write_ptr(cb_im2col_rm) : 0;
    [[maybe_unused]] const uint32_t im2col_rm_slot_bytes = row_split ? block_k_tiles * get_tile_size(cb_im2col_rm) : 0;
    [[maybe_unused]] uint32_t grants_served = 0;
    // The M-block's output pixels and slab bands, placed exactly as the reader places them.
    [[maybe_unused]] auto plan_row_block = [&](uint32_t m_block_idx) {
        const uint32_t block_tiles_done = m_block_idx * block_m_tiles;
        const uint32_t block_valid_tiles =
            actual_m_tiles - block_tiles_done < block_m_tiles ? actual_m_tiles - block_tiles_done : block_m_tiles;
        o_first = (m_tile_start + block_tiles_done) * 32;
        const uint32_t o_end_tiles = o_first + block_valid_tiles * 32;
        o_end = o_end_tiles < num_out_px ? o_end_tiles : num_out_px;
        const auto footprint =
            conv2d_3x3::block_footprint<output_width, input_width, stride, padding, dilation, band_last_tap>(
                o_first, o_end);
        if constexpr (tile_input) {
            conv2d_3x3::TileSlabPlan plan;
            conv2d_3x3::tile_slab_plan<input_width, input_height * input_width, dilation, c_in_p / 32>(
                footprint, slab_x_base, plan);
        } else {
            conv2d_3x3::slab_bands<input_width, input_height * input_width, dilation, band_stride_px>(
                footprint, slab_x_base, [](int32_t, int32_t, int32_t, int32_t) {});
        }
    };
    // Serve hand-outs until every unit below end_unit is assigned. The reader posts a grant before it advances
    // row_assigned past it, so one more look at row_go after seeing row_assigned catches the last one.
    [[maybe_unused]] auto serve_until = [&](uint32_t end_unit) {
        MaybeDeviceZoneScope("writer_row_serve");
        while (true) {
            if (Semaphore<>(row_go_sem_id).value() > grants_served) {
                const uint32_t s = Semaphore<>(row_mailbox_sem_id + grants_served % row_queue).value();
                const uint32_t k_tile0 = Units::k_block(s) * block_k_tiles;
                const int32_t kh = k_tile0 / tensor_kkh_tiles;
                row_reads.begin(1);
                RowGatherT::tile_row(
                    row_reads,
                    row_slab_noc,
                    im2col_rm_base + (s % im2col_rm_depth_rows) * im2col_rm_slot_bytes,
                    o_first + Units::tile_row(s) * 32,
                    o_end,
                    kh,
                    (k_tile0 - kh * tensor_kkh_tiles) * 32,
                    slab_x_base[kh]);
                row_reads.wait(1);
                Semaphore<>(row_done_sem_id).up(1);
                ++grants_served;
                continue;
            }
            if (Semaphore<>(row_assigned_sem_id).value() >= end_unit) {
                if (Semaphore<>(row_go_sem_id).value() > grants_served) {
                    continue;
                }
                break;
            }
            serve_fill();
        }
        dataflow_kernel_lib::set_read_trid(row_reads.noc, 0);
    };

    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        if constexpr (slab_split) {
            fill_forward_slab(m_block_idx);
            if (m_block_idx == 0) {
                prefetch_bias();
            }
        }
        const uint32_t m_seq_end = (m_block_idx + 1) * Units::per_block;
        if constexpr (nslice_outer) {
            nslice_round(m_block_idx, [&]() {
                if constexpr (row_split) {
                    // The reader waits every band before its first hand-out: serve the whole M-block.
                    plan_row_block(m_block_idx);
                    Semaphore<>(row_limit_sem_id).set(m_seq_end);
                    serve_until(m_seq_end);
                }
            });
        } else if constexpr (row_split) {
            // Per K-block: its weights, then the units of it the reader hands over (row_limit: units whose weights
            // are out), so the weight round of K-block k + 1 lands while compute multiplies K-block k.
            plan_row_block(m_block_idx);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                const uint32_t k_block_seq = m_block_idx * num_k_blocks + k_block_idx;
                while (k_blocks_issued <= k_block_seq) {
                    issue_weight_k_block();
                }
                const uint32_t ready = k_blocks_issued * block_m_tiles;
                Semaphore<>(row_limit_sem_id).set(ready < m_seq_end ? ready : m_seq_end);
                serve_until((k_block_seq + 1) * block_m_tiles);
            }
            if constexpr (k_split) {
                combine_round(m_block_idx);
            }
        } else {
#if defined(CONV2D_3X3_KW_FOLD) && !KWF_KBLOCKED
            // kw-fold whole-K: the weights stay resident across this core's M-blocks: one stream.
            while (k_blocks_issued < num_k_blocks) {
#else
            while (k_blocks_issued < (m_block_idx + 1) * num_k_blocks) {
#endif
                issue_weight_k_block();
                if constexpr (gather_split) {
                    gather_odd_groups(k_blocks_issued - 1);
                }
#ifdef CONV2D_3X3_KW_FOLD
                // The first final pass needs the bias: read it once that pass's sub-rounds are out (all of the first
                // M-block's unless KWF_NG_OUTER), before the writer can block on a slot only a later pass frees.
                if constexpr (!(weight_servers && !slab_split)) {
                    if (k_blocks_issued == kwf_first_pass_rounds) {
                        kwf_bias_read();
                    }
                }
#endif
            }
            if constexpr (k_split) {
                combine_round(m_block_idx);
            }
        }
        if constexpr (remote_output) {
            // ---- store_block (remote output shard): compute pushes the owner's [bm, out_n] block (per bias row
            // group, or whole); each valid tile-row goes to the output shard that holds it, as soon as it is packed.
            if (out_valid_tiles > 0) {
                MaybeDeviceZoneScope("writer_store");
                const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
                const auto output = TensorAccessor(output_args, output_addr, out_tile_bytes);
                const uint32_t block_tile0 = m_block_idx * block_m_tiles;
                const uint32_t l1_base = get_read_ptr(cb_output_tiles);
                for (uint32_t r = 0; r < block_m_tiles; ++r) {
                    const uint32_t local_tile_row = block_tile0 + r;
                    if (local_tile_row >= actual_m_tiles) {
                        break;
                    }
                    wait_front_serving(cb_output_tiles, (r + 1) * out_n_tiles);
                    const uint32_t row_page = (m_tile_start + local_tile_row) * tensor_n_tiles + out_n_tile_start;
                    const uint32_t l1 = l1_base + r * out_n_tiles * out_tile_bytes;
                    for (uint32_t n = 0; n < out_valid_tiles; ++n) {
                        noc_async_write_page(row_page + n, output, l1 + n * out_tile_bytes);
                    }
                }
                // The ragged tail's rows are packed too (nominal counts); the CB is popped whole.
                wait_front_serving(cb_output_tiles, block_m_tiles * out_n_tiles);
                noc_async_writes_flushed();
                cb_pop_front(cb_output_tiles, block_m_tiles * out_n_tiles);
            }
        }
        if constexpr (!output_sharded) {
            // ---- store_block: valid output tile-rows only; the ragged tail keeps nominal CB counts.
            MaybeDeviceZoneScope("writer_store");
            const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
            const auto output = TensorAccessor(output_args, output_addr, out_tile_bytes);
            const uint32_t block_tile0 = m_block_idx * block_m_tiles;
            for (uint32_t row_group = 0; row_group < row_groups_per_block; ++row_group) {
                cb_wait_front(cb_output_tiles, row_group_tiles);
                uint32_t l1 = get_read_ptr(cb_output_tiles);
                for (uint32_t r = 0; r < out_subblock_h; ++r) {
                    const uint32_t local_tile_row = block_tile0 + row_group * out_subblock_h + r;
                    if (local_tile_row < actual_m_tiles) {
                        const uint32_t row_page = (m_tile_start + local_tile_row) * tensor_n_tiles + n_tile_start;
                        for (uint32_t n = 0; n < block_n_tiles; ++n) {
                            noc_async_write_page(row_page + n, output, l1 + n * out_tile_bytes);
                        }
                    }
                    l1 += block_n_tiles * out_tile_bytes;
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_output_tiles, row_group_tiles);

                while (k_blocks_issued < total_k_blocks &&
                       cb_pages_reservable_at_back(cb_weights, weight_k_block_tiles)) {
                    issue_weight_k_block();
                }
            }
        }
    }
    drain_fill();
    if constexpr (!output_sharded || remote_output) {
        noc_async_write_barrier();
    }
    if constexpr (k_split || weight_servers) {
        noc_async_atomic_barrier();
    }
}
