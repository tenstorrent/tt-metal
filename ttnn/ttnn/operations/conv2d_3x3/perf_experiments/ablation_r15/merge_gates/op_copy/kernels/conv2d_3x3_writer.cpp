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
// server once per round that reuses a cb_weights slot (no kernel-start ack: a program's go follows every core's
// configuration, see conv2d_3x3_weight_server_writer.cpp); the server sets one flag per slot after the round's data,
// and the receiver clears it. The reuse
// ack of round k + 1 goes out while round k is awaited, as soon as compute freed round k + 1's slot (acking only once
// round k had landed held the server's round 2 behind every receiver's round 1). The server also multicasts the bias
// into cb_bias and sets the bias flag (the semaphore after the slot flags): the receiver reserves cb_bias at kernel
// start and pushes it once the flag is set, polled inside the round waits. Under kw-fold (KWF_ROUND0_STREAM) the server
// also sets an early flag per landed chunk of round 0 (the semaphores after the bias flag), and the receiver pushes
// round 0 chunk by chunk, so compute starts on its first kernel rows. Under kw-fold a rectangle may have two servers
// (one multicasting on NoC1, its partner on NoC0, each part of every chunk): every slot / early flag word then reads
// 0x0101 when complete, one byte per server, and the receiver compares against its arg's flag value.
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
// then. It is polled inside the weight waits and given as soon as it is ready; the first `depth` weight rounds (whose
// slots no walk of this M-block holds) and the split row gather's serving go out before it is waited for.
//
// Spare-column M-groups (descriptor SPARE_MGROUPS): M-groups transposed into the device columns past the shard grid,
// channel slice k of each on its spare column's row k, run the same N-slice walk with their ring down the column. Their
// weight rounds come over a second channel (`weight_mcast2`): the column participant after the staggered sender pushes
// each round it received and then forwards it, from its own slot, to the spare members of its slice (one row of the
// spare columns). A spare member receives like any column receiver (receive_polling_on). With the descriptor's
// NSLICE_ASYNC_BIAS the bias is read without a wait right after the first weight round is issued and pushed once it
// lands (polled in the walk's waits), so the weight stream, which paces the walk with the spare groups, starts first.
//
// N-slice column stream rate: the column sender and the forwarder multicast without waiting for the data to leave L1
// (the slot is refilled, or acked for reuse, only after a flush); the forwarder acks the column's next round before it
// forwards this one; with NSLICE_WEIGHT_PAGE_RUNS a round is one ND shard, read as one run. Measured on res4.x @ LoFi
// (unzoned, medians of 3, bit-identical): 11636 -> 10953 ns; the round period went 1.03 -> 0.74 us.
//
// N-slice round 0 (NSLICE_ROUND0, the descriptor's round-0 reader): every slot is fresh at kernel start, so round 0
// needs no handshake. One column participant per column, the round-0 reader Q (the column sender, or the row the
// descriptor picks for the shortest DRAM read path), issues its read at walk setup, before anything else of the walk,
// and multicasts it to its column (data, then the column's data-ready flag). The column sender, which sits on its
// slice's spare row, relays it to the slice's spare members (the forwarder starts at round 1; the members do not ack
// round 0), and runs the helper protocol from round 1. The N-slice async bias is read once the core's round 0 landed,
// and a fresh slot's read (round < depth) does not wait for this core's multicasts to leave L1. Measured on res5.0 @
// LoFi (zoned, us after kernel start, p50 / max): round 0 landed on spare members at 5.7 / 6.6, now 4.3 / 4.5, on grid
// members at 4.4 / 5.1, now 3.2 / 3.5; the 39 KB round-0 read took 0.9-2.2 us from the column senders' rows (NoC1
// climbs shared by up to four reads), 0.8-0.9 us from the picked rows.
//
// N-slice spread column stream (NSLICE_SPREAD, the descriptor's NSLICE_SPREAD): where every round of the core's
// stream fits its own cb_weights slot (depth = the core's rounds), no round needs a handshake (every slot is fresh at
// kernel start). Round r's reader member (runtime mask, in the round-0 args' place) reads it from DRAM into slot r and
// multicasts it down its column, then sets flag semaphore NSLICE_SPREAD_FLAG_SEM0 + r there (same path, after the
// data); the column sender relays each round, in order, as it lands on it to its slice's spare members. Every core
// pushes round r once its flag is VALID. A reader issues round r >= NSLICE_SPREAD_LOOKAHEAD once round r - lookahead
// landed on it; round 1 goes down the column only once round 0 landed on its sender. One multicast per poll (its issue
// holds this RISC for the injection, ~0.45 us for a 39 KB round), so a core pushes its own round as soon as it is
// flagged; the stream is polled from every wait of the walk (poll_credit). The handshake stream's round was a serial
// chain (a receiver acks round r + 1 once round r landed; the sender multicasts once every ack is in): ~0.85 us a
// round on res4.x, now its rounds land as their reads do (res4.x 10.53 -> 10.15 us, changelog Perf 14).
//
// Every push of cb_combine_slots is its whole capacity, so its write pointer never leaves the CB base and
// the same base address is right on every core. The counters only grow, so round j waits for
// (j + 1) x the per-round count.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "conv2d_3x3_combine_dataflow.hpp"
#include "conv2d_3x3_slab_dataflow.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"
#include "conv2d_3x3_nslice_order.hpp"
#include "conv2d_3x3_slab_direct.hpp"
#include "conv2d_3x3_slab_stream.hpp"

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
    // Spare-column M-groups (N-slice walk): a column receiver forwards each round to the spare members of its channel
    // slice (absent elsewhere; see the file comment).
    constexpr auto weight_mcast2 = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("weight_mcast2_ct_offset"),
        get_named_compile_time_arg_val("weight_mcast2_rt_offset")>();

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
    // Weight servers: on a receiver the value of a complete slot / early flag (VALID with one server; 0x0101 with two,
    // one byte lane each: conv2d_3x3_weight_server_writer.cpp), then its (first) server's (x, y).
    constexpr uint32_t server_arg = split_flag_arg + 1;
    [[maybe_unused]] const uint32_t server_flag_valid = weight_servers ? get_arg_val<uint32_t>(server_arg) : 0;
    [[maybe_unused]] const bool server_receiver = weight_servers && server_flag_valid != 0;
#ifdef NSLICE_ROUND0
    // N-slice round 0 (see the file comment), after the weight-server args (the N-slice walk has no kw-fold args):
    // [this core is its column's round-0 reader, the column's multicast rectangle (NoC-ordered sx, sy, ex, ey), the
    // column's other participants, then on the column sender the spare members' rectangle and count (0: none)].
    constexpr uint32_t round0_arg = server_arg + 3;
    [[maybe_unused]] const bool round0_reader = get_arg_val<uint32_t>(round0_arg) != 0;
    [[maybe_unused]] const uint32_t round0_spare_dests = get_arg_val<uint32_t>(round0_arg + 10);
#endif
    // Weight servers: the server's slot-reuse ack counter (slot 0's under SERVER_SLOT_ACKS).
    [[maybe_unused]] uint64_t server_ack_noc = 0;
    if (server_receiver) {
        server_ack_noc = get_noc_addr(
            get_arg_val<uint32_t>(server_arg + 1),
            get_arg_val<uint32_t>(server_arg + 2),
            get_semaphore(server_ack_sem_id));
    }

#ifdef SD_BIAS_IN_WEIGHTS
    // Slab-direct fused bias without weight servers: every weight round carries the core's bias tiles after its
    // K-block (read and rewritten once by the round's DRAM reader, the multicast sender), so no core reads the bias
    // pages from DRAM by itself (~100 cores reading them at once slowed the fill and the weight round).
    constexpr uint32_t sd_bias_round_tiles = block_n_tiles;
#else
    constexpr uint32_t sd_bias_round_tiles = 0;
#endif
    constexpr uint32_t weight_k_block_tiles = block_k_tiles * block_n_tiles + sd_bias_round_tiles;
    constexpr uint32_t row_group_tiles = out_subblock_h * block_n_tiles;
    constexpr uint32_t row_groups_per_block = block_m_tiles / out_subblock_h;
    const uint32_t weight_tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, weight_tile_bytes);

    // ---- stream_weight_blocks: K-block tiles k-major, n-fastest (matmul in1 layout).
    const uint32_t total_k_blocks = num_m_blocks_this_core * num_k_blocks;
    uint32_t k_blocks_issued = 0;
    // Slab-direct round 0 (sd_w0): its DRAM reader (the multicast sender, or a core outside the channel) issues it
    // under transaction id 8 at kernel entry (below), a receiver of the reader multicast reserves its slot before the
    // fill; it completes (sd_w0_try) the moment it is in.
    constexpr uint32_t TRID_SD_WEIGHTS = 8;
    [[maybe_unused]] bool sd_w_issued_early = false;
    [[maybe_unused]] bool sd_r0_mcast = false;  // round 0 comes over the sender reader's NoC0 multicast
    [[maybe_unused]] uint32_t sd_w0 = 0;        // 0: round 0 not taken early, 1: pending, 2: pushed
    auto finish_weight_k_block = [&](uint32_t l1) {
#if defined(SD_BIAS_IN_WEIGHTS) && defined(SD_BIAS_SPLIT)
        const uint32_t sd_bias_l1 = l1 + block_k_tiles * block_n_tiles * weight_tile_bytes;
        invalidate_l1_cache();
        for (uint32_t n = 0; n < out_valid_tiles; ++n) {
#if SD_BIAS_BF16
            conv2d_3x3::slab_direct::split_bias_tile_bf16(sd_bias_l1 + n * weight_tile_bytes);
#else
            conv2d_3x3::slab_direct::split_bias_tile_bf8(sd_bias_l1 + n * weight_tile_bytes);
#endif
        }
#endif
    };
    auto read_weight_k_block = [&](uint32_t l1, bool issue_only = false) {
        [[maybe_unused]] const uint32_t l1_start = l1;
#ifdef CONV2D_3X3_KW_FOLD
        // kw-fold: round s holds column kw of the output columns [n0, n0 + rw): W'[kh Ct + c, kw N + n0 + n] =
        // prepared tile ((kh * 3 + kw) Ct + c, n0 + n), (kh, c)-major, n-fastest (block_k_tiles = Ct, block_n_tiles =
        // 3 rw: the same round size). rw is one n-group (K-blocked sub-rounds) or all of C_out (whole-K, one n-group
        // of rounds); round s is n-group s / 3, kw = (1, 0, 2)[s % 3] (kw_fold.hpp KW_ROUND_ORDER: Q_1 is the
        // overwrite the other two accumulate onto).
        constexpr uint32_t kw_round_order[3] = {1, 0, 2};
        constexpr uint32_t kwf_rw = block_n_tiles / 3;
        const uint32_t s = k_blocks_issued % num_k_blocks;
        const uint32_t kw = kw_round_order[s % 3];
        const uint32_t n0 = (s / 3) * kwf_rw;
#if KWF_FUSED_GW
        // Fused side rounds (kw_fold.hpp): rounds 1 and 2 hold the kw = 0 | 2 pair interleaved, per (kh, c) row of
        // 2 rw tiles: each gw-wide n-group's kw = 0 columns, then its kw = 2 columns. Round 0 (kw = 1) is unchanged.
        if (s % 3 != 0) {
            constexpr uint32_t gwf = KWF_FUSED_GW;
            constexpr uint32_t frow = 2 * kwf_rw;
            constexpr uint32_t round = 3 * block_k_tiles * kwf_rw;
            const uint32_t f0 = (s % 3 - 1) * round;
            for (uint32_t f = f0; f < f0 + round; ++f, l1 += weight_tile_bytes) {
                const uint32_t kc = f / frow;  // kh * Ct + c
                const uint32_t j = f - kc * frow;
                const uint32_t grp = j / (2 * gwf);
                const uint32_t jj = j - grp * 2 * gwf;
                const uint32_t kh = kc / block_k_tiles;
                const uint32_t kwf = jj < gwf ? 0 : 2;
                const uint32_t n = grp * gwf + (jj < gwf ? jj : jj - gwf);
                noc_async_read_page(
                    ((kh * 3 + kwf) * block_k_tiles + (kc - kh * block_k_tiles)) * tensor_n_tiles + n0 + n,
                    weights,
                    l1);
            }
        } else
#endif
        {
            for (uint32_t kh = 0; kh < 3; ++kh) {
                for (uint32_t c = 0; c < block_k_tiles; ++c) {
                    const uint32_t row_page = ((kh * 3 + kw) * block_k_tiles + c) * tensor_n_tiles + n0;
                    for (uint32_t n = 0; n < kwf_rw; ++n, l1 += weight_tile_bytes) {
                        noc_async_read_page(row_page + n, weights, l1);
                    }
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
#ifdef SD_BIAS_IN_WEIGHTS
        // The round's tail: the bias tiles (same format and page size as the weights), rewritten after they landed.
        {
            const auto bias = TensorAccessor(bias_args, bias_addr, weight_tile_bytes);
            for (uint32_t n = 0; n < sd_bias_round_tiles; ++n, l1 += weight_tile_bytes) {
                if (n < out_valid_tiles) {
                    noc_async_read_page(out_n_tile_start + n, bias, l1);
                }
            }
        }
#endif
        if (issue_only) {
            return;
        }
        noc_async_read_barrier();
        finish_weight_k_block(l1_start);
    };

#ifdef CONV2D_3X3_SLAB_DIRECT
    // Slab-direct round 0, DRAM half, first thing at kernel entry (the rest of the writer's set-up took ~0.5 us before
    // the old issue point): the round's DRAM reader reserves its slot and issues the reads under TRID_SD_WEIGHTS; a
    // multicast receiver acks before its fill (below). The role is read off the channel's runtime metadata, before the
    // pipes exist.
    if (num_m_blocks_this_core > 0 && !server_receiver &&
        (!weight_mcast.can_receive() || (weight_mcast.can_send() && weight_mcast.should_send(0)))) {
        MaybeDeviceZoneScope("writer_sd_weight_issue");
        Noc w0_noc;
        cb_reserve_back(cb_weights, weight_k_block_tiles);
        dataflow_kernel_lib::set_read_trid(w0_noc, TRID_SD_WEIGHTS);
        read_weight_k_block(get_write_ptr(cb_weights), true);
        dataflow_kernel_lib::set_read_trid(w0_noc, 0);
        sd_w_issued_early = true;
    }
#endif

    // ---- Split TILE fill, writer half (declared here: kw-fold's round-0 wait below serves it before the rest of the
    // start-up; see "Split TILE fill, writer half" below).
    constexpr uint32_t fill_c_in_tiles = c_in_p / 32;
    constexpr uint32_t fill_batches_per_block = tile_fill_split ? slab_tile_rows / untilize_batch_rows : 1;
    constexpr uint32_t fill_row_bytes = fill_c_in_tiles * input_page_bytes;
    constexpr uint32_t fill_batch_bytes = untilize_batch_rows * fill_row_bytes;
    [[maybe_unused]] const uint32_t fill_ring_base = tile_fill_split ? get_write_ptr(cb_input_tiles) : 0;
    conv2d_3x3::TaggedReads<false> fill_reads;
#ifdef CONV2D_3X3_KW_FOLD
    // kw-fold: the host-planned fill (descriptor _kwf_fill_args). Per M-block this writer's part is [groups] + per
    // group [pieces] + pieces (a suffix of the block's slab: the higher shards' rows and the last own-shard rows), read
    // on NoC1 once the reader handed the block over (slab_free), each group under its own transaction id (up to
    // fill_writer_window in flight) and counted in slab_fwd_done in order as it lands. The first block's groups may go
    // out at kernel start (writer_fill_early): the slab ring is empty then.
#ifdef KWF_XCHG
    constexpr uint32_t kwf_fill_arg0 = server_arg + 3 + 10;  // after the halo exchange's 10 args
#else
    constexpr uint32_t kwf_fill_arg0 = server_arg + 3;
#endif
    constexpr uint32_t TRID_KWF_FILL = 8;
    [[maybe_unused]] uint32_t kwf_fill_arg = kwf_fill_arg0;
    [[maybe_unused]] uint32_t kwf_fill_m = 0;       // the M-block whose groups are being issued
    [[maybe_unused]] uint32_t kwf_fill_left = 0;    // its groups not issued yet
    [[maybe_unused]] uint32_t kwf_fill_blocks = 0;  // blocks with groups started (the reader hands each over once)
    [[maybe_unused]] uint32_t kwf_fill_issued = 0;  // groups issued, all blocks
    [[maybe_unused]] uint32_t kwf_fill_landed = 0;  // groups landed and counted in slab_fwd_done
    [[maybe_unused]] bool kwf_fill_open = false;    // kwf_fill_left belongs to block kwf_fill_m
    [[maybe_unused]] auto kwf_serve_fill = [&](bool early) {
        while (kwf_fill_landed < kwf_fill_issued &&
               ncrisc_noc_read_with_transaction_id_flushed(
                   fill_reads.noc.get_noc_id(), TRID_KWF_FILL + kwf_fill_landed % fill_writer_window)) {
            Semaphore<>(slab_fwd_done_sem_id).up(1);
            ++kwf_fill_landed;
        }
        while (kwf_fill_issued - kwf_fill_landed < fill_writer_window) {
            if (!kwf_fill_open) {
                if (kwf_fill_m >= num_m_blocks_this_core) {
                    return;
                }
                const uint32_t groups = get_arg_val<uint32_t>(kwf_fill_arg);
                if (groups == 0) {
                    ++kwf_fill_arg;
                    ++kwf_fill_m;
                    continue;
                }
                // Early (kernel start): only the core's first M-block, whose slab the ring holds empty. A later block's
                // slab may still be read by compute (or filled by the reader for an earlier block).
                if (!(early && kwf_fill_m == 0) && Semaphore<>(slab_free_sem_id).value() <= kwf_fill_blocks) {
                    return;
                }
                ++kwf_fill_arg;
                ++kwf_fill_blocks;
                kwf_fill_left = groups;
                kwf_fill_open = true;
            }
            MaybeDeviceZoneScope("writer_fill");
            const uint32_t slab_l1 =
                fill_ring_base + ((kwf_fill_m * fill_batches_per_block) % input_tile_depth_batches) * fill_batch_bytes;
            const uint32_t pieces = get_arg_val<uint32_t>(kwf_fill_arg++);
            fill_reads.begin(TRID_KWF_FILL + kwf_fill_issued % fill_writer_window);
            for (uint32_t i = 0; i < pieces; ++i, kwf_fill_arg += 3) {
                const uint32_t xy = get_arg_val<uint32_t>(kwf_fill_arg);
                const uint32_t dst = get_arg_val<uint32_t>(kwf_fill_arg + 2);
                fill_reads.read(
                    get_noc_addr(xy >> 16, xy & 0xFFFF, input_addr + get_arg_val<uint32_t>(kwf_fill_arg + 1)),
                    slab_l1 + (dst >> 16) * fill_row_bytes,
                    (dst & 0xFFFF) * fill_row_bytes);
            }
            dataflow_kernel_lib::set_read_trid(fill_reads.noc, 0);
            ++kwf_fill_issued;
            if (--kwf_fill_left == 0) {
                kwf_fill_open = false;
                ++kwf_fill_m;
            }
        }
    };
    [[maybe_unused]] auto kwf_fill_done = [&]() {
        return kwf_fill_m >= num_m_blocks_this_core && kwf_fill_landed == kwf_fill_issued;
    };
#endif
    // ---- kw-fold round 0 under weight servers: straight from the fill issue into round 0's wait, before the rest of
    // this kernel's start-up. The wait serves the fill (slab_fwd_done as each group lands, so the reader pushes the
    // slab) and pushes round 0's kernel-row chunks and the server's bias as they land. Everything else this writer
    // sets up (exchange, multicast channels, bias bookkeeping, the round loop) is first-run code: ~0.5 us of it before
    // the round-0 wait put the slab hand-over and round 0's push ~1.2 us after kernel start even with every read and
    // the weights stubbed (res2), while compute waited from ~0.25 us. Measured with the server's start ack gone
    // (Blackhole p150b, LoFi bf8_b, compute waits zoned only, per core from its own kernel start, median of cores):
    // slab ready res2 1.30 -> 1.04 us, res3.x 1.24 -> 0.96, center_head.1 1.44 -> 1.38; first matmul 1.63 -> 1.31, 1.88
    // -> 1.64, 1.73 -> 1.44. Later rounds take the round loop (issue_weight_k_block).
    [[maybe_unused]] bool bias_pushed = false;
#ifdef CONV2D_3X3_KW_FOLD
    if constexpr (
        weight_servers && tile_fill_split && !ring_slab && input_tile_depth_batches >= fill_batches_per_block) {
        if (server_receiver && num_m_blocks_this_core > 0) {
            if (split_here) {
                // The first block's groups (the ring is empty), without the reader's hand-over (writer_fill_early).
                kwf_serve_fill(true);
            }
            MaybeDeviceZoneScope("writer_round0");
#ifdef KWF_ROUND0_STREAM
            constexpr uint32_t r0_chunks = KWF_ROUND0_STREAM;
#else
            constexpr uint32_t r0_chunks = 1;
#endif
            constexpr uint32_t r0_chunk_tiles = weight_k_block_tiles / r0_chunks;
            // Round 0's slot (slot 0, free at kernel start), its flag, and the server's bias flag (after the slot
            // flags; kw-fold is never slab-direct, so a server multicasts the bias whenever there is one).
            constexpr bool r0_bias = has_bias && !slab_split;
            cb_reserve_back(cb_weights, weight_k_block_tiles);
            volatile tt_l1_ptr uint32_t* r0_flag =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(server_flag_sem_id));
            volatile tt_l1_ptr uint32_t* r0_bias_flag =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(server_flag_sem_id + nslice_weight_depth));
            if constexpr (r0_bias) {
                cb_reserve_back(cb_bias, out_n_tiles);
            }
            uint32_t r0_pushed = 0;  // chunks pushed
            while (true) {
                invalidate_l1_cache();
                if constexpr (r0_bias) {
                    if (!bias_pushed && *r0_bias_flag == VALID) {
                        cb_push_back(cb_bias, out_n_tiles);
                        bias_pushed = true;
                    }
                }
                if (*r0_flag == server_flag_valid) {
                    break;
                }
                if constexpr (r0_chunks > 1) {
                    // Semaphore server_flag_sem_id + depth + j (after the bias flag), j >= 1: chunks 0 .. j - 1 landed.
                    for (uint32_t j = r0_chunks - 1; j > r0_pushed; --j) {
                        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                get_semaphore(server_flag_sem_id + nslice_weight_depth + j)) == server_flag_valid) {
                            cb_push_back(cb_weights, (j - r0_pushed) * r0_chunk_tiles);
                            r0_pushed = j;
                            break;
                        }
                    }
                }
                if (split_here) {
                    kwf_serve_fill(false);
                }
            }
            *r0_flag = INVALID;
            cb_push_back(cb_weights, weight_k_block_tiles - r0_pushed * r0_chunk_tiles);
            k_blocks_issued = 1;  // round 0 is in: the round loop starts at round 1
        }
    }
#endif

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
#else
    constexpr bool kw_fold = false;
#endif
    // With the polling split TILE fill the bias reads are issued later, serving the first block's hand-overs.
#ifdef SD_BIAS_IN_WEIGHTS
    constexpr bool sd_bias_in_weights = true;  // the bias comes with the weight rounds
#else
    constexpr bool sd_bias_in_weights = false;
#endif
#ifdef NSLICE_ASYNC_BIAS
    // N-slice walk: the bias is read without a wait once the first weight round is issued, and pushed when it lands
    // (polled in the walk's waits), so the column weight stream starts at kernel start (see issue_async_bias).
    constexpr bool async_bias = has_bias;
#else
    constexpr bool async_bias = false;
#endif
    if constexpr (
        !slab_split && !tile_fill_split && !weight_servers && !kw_fold && !sd_bias_in_weights && !async_bias) {
        prefetch_bias();
    }

    // ---- Split TILE fill, writer half: the higher shards' input tile-rows of every cb_input_tiles batch the reader
    // hands over (fill_go), on NoC1, into the batch's ring slot; fill_done counts the filled batches. Every reader
    // reserve and push is one batch and the CB holds input_tile_depth_batches of them, so batch g of the kernel sits
    // at slot g % depth. Served wherever the writer would otherwise wait: compute untilizes a block before it
    // multiplies it, so the weight round two past the block's first waits on this block's fill.
    [[maybe_unused]] const uint32_t fill_hi_row = (my_shard + 1) * input_shard_rows;
    [[maybe_unused]] uint32_t fill_m = 0;  // cursor: block and batch of the next batch to look at
    [[maybe_unused]] uint32_t fill_b = 0;
    [[maybe_unused]] uint32_t fill_plan_m = 0xFFFFFFFF;
    [[maybe_unused]] uint32_t fill_served = 0;
    [[maybe_unused]] conv2d_3x3::TileSlabPlan fill_plan;
    [[maybe_unused]] conv2d_3x3::SlabSuffix fill_sfx{0, 0};  // the cursor block's slab rows this writer reads
    // Ring slab: the cursor block's first streamed row, its row end and batch count, the kernel batches before it, and
    // the previous block's window end.
    [[maybe_unused]] uint32_t ring_push_lo = 0;
    [[maybe_unused]] uint32_t ring_hi = 0;
    [[maybe_unused]] uint32_t ring_batches = 0;
    [[maybe_unused]] uint32_t ring_batch_base = 0;
    [[maybe_unused]] int32_t ring_prev_hi = -1;
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
                fill_sfx = conv2d_3x3::slab_rows_from(fill_plan, fill_hi_row);
                fill_plan_m = fill_m;
            }
            if (conv2d_3x3::batch_has_rows_from<untilize_batch_rows>(fill_sfx, fill_b)) {
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
#ifdef CONV2D_3X3_SLAB_DIRECT
    constexpr bool slab_direct = true;
#else
    constexpr bool slab_direct = false;
#endif
    // Slab-direct: this block's forward slab reads (transaction id 9) are in flight; slab_fwd_done goes up as soon as
    // they landed, from every wait that serves the fill (the weight-round waits included).
    [[maybe_unused]] bool sd_slab_pending = false;
#ifdef SD_STREAM
    // Streamed fill (conv2d_3x3_slab_stream.hpp): this block's forward part in chunks under transaction ids 9..;
    // serve_fill retires the landed ones, issues more and publishes slab_fwd_done = block * (SD_SLAB_PX + 1) + the
    // landed count (SD_SLAB_PX once every forward piece landed).
    conv2d_3x3::slab_stream::Stream<SD_SLOTS + 4, SD_STREAM_NTRID, 9, SD_STREAM_CHUNK_PX_W, input_width, SD_GAP>
        sd_wstream;
    uint32_t sd_w_base = 0;
    Noc sd_wnoc;
    const uint32_t sd_slab_l1 = get_write_ptr(cb_halo_slab);
    auto sd_wread = [&](uint32_t trid, uint32_t lo, uint32_t hi, uint32_t px) {
        const auto sd_input = TensorAccessor(input_args, input_addr, input_page_bytes);
        dataflow_kernel_lib::set_read_trid(sd_wnoc, trid);
        uint32_t l1 = sd_slab_l1 + px * input_page_bytes;
        for (uint32_t page = lo; page < hi;) {
            const uint32_t run = sd_input.num_contiguous_pages(page, hi);
            noc_async_read(sd_input.get_noc_addr(page), l1, run * input_page_bytes);
            page += run;
            l1 += run * input_page_bytes;
        }
        dataflow_kernel_lib::set_read_trid(sd_wnoc, 0);
    };
    auto sd_stream_serve = [&]() {
        const bool moved = sd_wstream.poll(noc_index);
        while (sd_wstream.issue_one(sd_wread)) {
        }
        if (moved) {
            Semaphore<>(slab_fwd_done_sem_id).set(sd_w_base + sd_wstream.landed);
        }
        if (sd_wstream.done()) {
            sd_slab_pending = false;
        }
    };
#endif
#ifdef CONV2D_3X3_HOST_FILL
    // Host-planned writer fill (descriptor _host_fill_args, after the server args and any N-slice round-0 args):
    // [gate << 16 | batches], then per batch [pieces] and (owner x << 16 | y, offset, slab row << 16 | rows) per piece.
    // One M-block per core and the whole slab in the ring (empty at kernel start), so every batch is issued from kernel
    // start (up to the window in flight) without the reader's hand-over; the first `gate` batches (the first kernel
    // row's band: BLOCK_FILL_GATE) go out alone, the rest once those landed. slab_fwd_done publishes how many batches
    // were issued; the reader polls each one's transaction id on this NoC for its landing.
    static_assert(
        fill_writer_window == HOST_FILL_WRITER_WINDOW, "the reader polls the writer's fill trids mod this window");
#if defined(NSLICE_SPREAD)
    uint32_t hf_arg = round0_arg + 12;  // after the spread stream's args
#elif defined(NSLICE_ROUND0)
    uint32_t hf_arg = round0_arg + 11;  // after the N-slice round-0 args
#else
    uint32_t hf_arg = server_arg + 3;
#endif
    const uint32_t hf_word = split_here ? get_arg_val<uint32_t>(hf_arg) : 0;
    const uint32_t hf_batches = hf_word & 0xFFFFu;
    const uint32_t hf_gate = hf_word >> 16;
    ++hf_arg;
    uint32_t hf_issued = 0;
    uint32_t hf_served = 0;  // batches landed (in order), for the window and the gate
    // Batches still to issue: the waits that serve the fill stop calling it once it is done (a call in every poll
    // iteration of the walk's waits measured +1.6% on res4.x even with nothing to read).
    bool hf_live = hf_batches > 0;
    auto hf_issuable = [&]() {
        return hf_issued < hf_batches && hf_issued - hf_served < fill_writer_window &&
               (hf_issued < hf_gate || hf_served >= hf_gate);
    };
    auto hf_serve = [&]() {
        while (hf_served < hf_issued &&
               ncrisc_noc_read_with_transaction_id_flushed(
                   fill_reads.noc.get_noc_id(), TRID_WFILL_BASE + hf_served % fill_writer_window)) {
            ++hf_served;
        }
        hf_live = hf_issued < hf_batches;
        if (hf_issuable()) {
            MaybeDeviceZoneScope("writer_fill");
            do {
                const uint32_t pieces = get_arg_val<uint32_t>(hf_arg++);
                fill_reads.begin(TRID_WFILL_BASE + hf_issued % fill_writer_window);
                for (uint32_t i = 0; i < pieces; ++i, hf_arg += 3) {
                    const uint32_t xy = get_arg_val<uint32_t>(hf_arg);
                    const uint32_t dst = get_arg_val<uint32_t>(hf_arg + 2);
                    const uint64_t src =
                        get_noc_addr(xy >> 16, xy & 0xFFFF, input_addr + get_arg_val<uint32_t>(hf_arg + 1));
                    const uint32_t l1 = fill_ring_base + (dst >> 16) * fill_row_bytes;
                    if constexpr (fill_zero_lanes) {
                        // Short last slice: each row's real tiles, its padding tiles from the zero block.
                        constexpr uint32_t real_row_bytes = (c_in / 32) * input_page_bytes;
                        for (uint32_t r = 0; r < (dst & 0xFFFF); ++r) {
                            fill_reads.read(src + r * fill_row_bytes, l1 + r * fill_row_bytes, real_row_bytes);
                            fill_reads.read(
                                fill_zero_noc,
                                l1 + r * fill_row_bytes + real_row_bytes,
                                fill_row_bytes - real_row_bytes);
                        }
                    } else {
                        fill_reads.read(src, l1, (dst & 0xFFFF) * fill_row_bytes);
                    }
                }
                ++hf_issued;
            } while (hf_issuable());
            dataflow_kernel_lib::set_read_trid(fill_reads.noc, 0);
            // Publish the issued count once the NIU counts the last batch's requests (the read-back of its counter
            // orders the publish behind the command-buffer writes).
            (void)NOC_STATUS_READ_REG(
                fill_reads.noc.get_noc_id(),
                NIU_MST_REQS_OUTSTANDING_ID(TRID_WFILL_BASE + (hf_issued - 1) % fill_writer_window));
            Semaphore<>(slab_fwd_done_sem_id).set(hf_issued);
        }
    };
#endif
    // Every batch the reader has handed over so far (windowed: retire the landed ones, issue up to the window).
    [[maybe_unused]] auto serve_fill = [&]() {
#ifdef CONV2D_3X3_HOST_FILL
        if constexpr (tile_fill_split) {
            if (hf_live) {
                hf_serve();
            }
            return;
        }
#endif
        if constexpr (slab_direct) {
#ifdef SD_STREAM
            if (sd_slab_pending) {
                sd_stream_serve();
            }
#else
            if (sd_slab_pending && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 9)) {
                Semaphore<>(slab_fwd_done_sem_id).up(1);
                sd_slab_pending = false;
            }
#endif
        }
#ifdef CONV2D_3X3_KW_FOLD
        if constexpr (tile_fill_split) {
            if (split_here) {
                kwf_serve_fill(false);
            }
            return;
        }
#endif
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
#ifdef CONV2D_3X3_HOST_FILL
            while (hf_live) {
                hf_serve();
            }
            return;
#endif
#ifdef CONV2D_3X3_KW_FOLD
            while (!kwf_fill_done()) {
                kwf_serve_fill(false);
            }
            return;
#endif
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

    // ---- Split TILE fill, first block: plan it and issue its hand-overs before the rest of the start-up. Whole-slab
    // ring only: the reader then reserves the block's whole slab at once (the ring is empty at kernel start) and hands
    // every batch over at its fill start, which waits on nothing of this writer, so the spin below ends. With a smaller
    // ring its first writer batch can wait for compute to free slots, i.e. for this writer's weights. Issued after the
    // hand-over in the weight-round wait instead, the reads went out ~0.5 us later and ran beside round 0's multicast
    // on this core's NoC1 port (res2: 10.77 -> 10.47 us unzoned, with the cheaper hand-over above). Carve-out: under
    // the in-grid weight sender (no weight servers: stem.conv2 / conv3, input on the whole grid) round 0 is multicast
    // early, right into these reads; there the early issue measured stem.conv2 @ LoFi 28.33 -> 27.42 us without it,
    // @ HiFi4 76.09 -> 74.95, stem.conv3 48.31 -> 47.84 (unzoned, 2 alternating pairs).
    if constexpr (
        tile_fill_split && !ring_slab && input_tile_depth_batches >= fill_batches_per_block && weight_servers) {
        if (split_here) {
            MaybeDeviceZoneScope("writer_fill_early");
#ifdef CONV2D_3X3_KW_FOLD
            // kw-fold: the first block's group goes out now (the ring is empty), without the reader's hand-over (a
            // server receiver issued it already, before its round-0 wait above).
            kwf_serve_fill(true);
#else
            // Only a first-block batch: the reader hands a later block's batches over only after its first block's
            // fill, so a spin on one here would deadlock (a first block without writer rows: a multi-block core whose
            // first block's slab ends inside its own shard).
            if (fill_next() && fill_m == 0) {
                while (Semaphore<>(slab_free_sem_id).value() == 0) {
                }
                serve_fill();
            }
#endif
        }
    }

#ifdef CONV2D_3X3_HOST_FILL
    // Host-planned fill: this writer's part goes out now (the slab ring is empty at kernel start).
    if constexpr (tile_fill_split) {
        if (hf_live) {
            MaybeDeviceZoneScope("writer_fill_early");
            if constexpr (fill_zero_lanes) {
                // The padding lanes' zero block: the reader zeroes it at its start too, but this writer reads it now.
                fill_reads.noc.async_write_zeros(
                    CircularBuffer(cb_zero_src), fill_row_bytes - (c_in / 32) * input_page_bytes);
                fill_reads.noc.write_zeros_l1_barrier();
            }
            hf_serve();
        }
    }
#endif
    Noc noc;
    auto weight_sender = weight_mcast.optional_sender(noc);
    auto weight_receiver = weight_mcast.optional_receiver(noc);
    auto weight2_sender = weight_mcast2.optional_sender(noc);
    [[maybe_unused]] auto weight2_receiver = weight_mcast2.optional_receiver(noc);
    // Raw weight-channel receive, bypassing ReceiverPipe::receive (which blocks on the data-ready signal and has no
    // poll / idle-callback form): the same protocol for a fixed-sender hardware multicast (ack the sender, then wait
    // for the round's data-ready signal), serving the reader's fill hand-overs while it waits, so one slow core's
    // weight round does not hold this core's untilize. Measured on sem res2.fuse_conv.1 @ LoFi: 138.6 us with the
    // blocking receive vs 129.8 us polled (split TILE fill on both).
    [[maybe_unused]] auto receive_serving = [&](auto& wm, uint32_t round) {
        using WM = std::decay_t<decltype(wm)>;
        // Dependent on WM, so a kernel without the split never instantiates the sender-coordinate reads.
        if constexpr (WM::active && (tile_fill_split || slab_direct)) {
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
    if constexpr (!slab_split && tile_fill_split && !weight_servers && !kw_fold && !async_bias) {
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
    // ---- kw-fold halo exchange (conv2d_3x3_kw_fold.hpp, one M-block per core). The boundary between this core and
    // its upper / lower neighbour lies inside an image row (runtime flags), and the neighbour, not a recomputed halo
    // tile-row, supplies the boundary rows' missing kw partial, once per n-group pass. In ACC rows (fp32, every plane
    // of the n-group): fv = this core's first valid row, lve = one past its last; up_lve = the upper core's lve, lo_fv
    // = the lower core's fv.
    //   A (lower -> upper, after the lower's first Q_1 group): rows [fv, fv + d) -> the upper's [up_lve, up_lve + d),
    //     the spill rows its Q_0 then accumulates onto.
    //   B (upper -> lower, after the upper's last Q_0 group): [lve - d, lve + d) -> the lower's [lo_fv - d, lo_fv + d):
    //     the upper's last rows (Q_1 + Q_0) become the base of the lower's top spill rows and the lower's first rows
    //     come back as Q_1 + Q_0, both for the lower's Q_2 to accumulate onto.
    //   C (lower -> upper, after the lower's first Q_2 group): [fv - d, fv) -> the upper's [up_lve - d, up_lve).
    // Every row thus sums Q_1, Q_0, Q_2 in the same order as with the recompute (bit-identical). Compute pushes one
    // cb_halo_out token per send (pack thread: after the packer drained) and waits one cb_halo_in token per arrival;
    // this writer moves the rows (write, barrier, then the receiver's semaphore) and turns semaphore counts into
    // tokens, polled inside the weight waits (from round 1 on), the weight-slot reserve and the bias wait, and drained
    // after the rounds. n-group g's counts: A 2g + 1 and C 2g + 2 from the lower neighbour, B g + 1 from the upper.
    // The reserve must poll: with n-groups, compute can hold both weight slots at a gate this writer has to open.
#ifdef KWF_XCHG
    constexpr uint32_t khx_arg = server_arg + 3;
    const bool khx_top = get_arg_val<uint32_t>(khx_arg) != 0;
    const bool khx_bot = get_arg_val<uint32_t>(khx_arg + 1) != 0;
    const uint32_t khx_fv = get_arg_val<uint32_t>(khx_arg + 6);
    const uint32_t khx_lve = get_arg_val<uint32_t>(khx_arg + 7);
    const uint32_t khx_up_lve = get_arg_val<uint32_t>(khx_arg + 8);
    const uint32_t khx_lo_fv = get_arg_val<uint32_t>(khx_arg + 9);
    const uint32_t khx_acc = get_write_ptr(KWF_CB_ACC);
    const uint64_t khx_up_noc = get_noc_addr(get_arg_val<uint32_t>(khx_arg + 2), get_arg_val<uint32_t>(khx_arg + 3), 0);
    const uint64_t khx_lo_noc = get_noc_addr(get_arg_val<uint32_t>(khx_arg + 4), get_arg_val<uint32_t>(khx_arg + 5), 0);
    // Per n-group, sends in compute's order: A (top), B (bottom), C (top); arrivals in compute's order: A (bottom),
    // B (top), C (bottom).
    const uint32_t khx_out_g = (khx_top ? 2 : 0) + (khx_bot ? 1 : 0);
    const uint32_t khx_in_g = (khx_bot ? 2 : 0) + (khx_top ? 1 : 0);
    const uint32_t khx_out_n = KWF_NG * khx_out_g;
    const uint32_t khx_in_n = KWF_NG * khx_in_g;
    uint32_t khx_out_i = 0;
    uint32_t khx_in_i = 0;
    auto khx_send =
        [&](uint64_t noc_base, uint32_t src_row, uint32_t dst_row, uint32_t rows, uint32_t sem_id, uint32_t sem_value) {
            MaybeDeviceZoneScope("writer_khx_send");
            // Each plane's two half-planes (face-split ACC, conv2d_3x3_kw_fold.hpp).
            for (uint32_t h = 0; h < 2 * KWF_PLANES; ++h) {
                const uint32_t plane = khx_acc + h * (KWF_PLANE_BYTES / 2);
                noc_async_write(
                    plane + src_row * KWF_PITCH, noc_base | (plane + dst_row * KWF_PITCH), rows * KWF_PITCH);
            }
            noc_async_write_barrier();
            // One writer per semaphore and its sends in order, so the count goes out as an inline write of its value: a
            // NoC atomic increment here slowed the weight multicast landing beside it (center_head.1: rounds 2.05 ->
            // 2.4-2.7 us apart, +3.3% whole op).
            noc_inline_dw_write(noc_base | get_semaphore(sem_id), sem_value);
        };
    auto khx_poll = [&]() {
        if (khx_out_i < khx_out_n && cb_pages_available_at_front(KWF_CB_HALO_OUT, 1)) {
            // Which send this token is: the k-th of [A?, B?, C?] of n-group gi.
            const uint32_t gi = khx_out_i / khx_out_g;
            const uint32_t j = khx_out_i - gi * khx_out_g;
            const uint32_t k = khx_top ? j : j + 1;  // 0 = A, 1 = B, 2 = C
            if (k == 0) {
                khx_send(khx_up_noc, khx_fv, khx_up_lve, KWF_D, KWF_SEM_FROM_LOWER, 2 * gi + 1);
            } else if (k == 1 && khx_bot) {
                khx_send(khx_lo_noc, khx_lve - KWF_D, khx_lo_fv - KWF_D, 2 * KWF_D, KWF_SEM_FROM_UPPER, gi + 1);
            } else {
                khx_send(khx_up_noc, khx_fv - KWF_D, khx_up_lve - KWF_D, KWF_D, KWF_SEM_FROM_LOWER, 2 * gi + 2);
            }
            cb_pop_front(KWF_CB_HALO_OUT, 1);
            ++khx_out_i;
        }
        if (khx_in_i < khx_in_n) {
            // The k-th of [A?, B?, C?] of n-group gi: A and C count on from_lower, B on from_upper.
            const uint32_t gi = khx_in_i / khx_in_g;
            const uint32_t j = khx_in_i - gi * khx_in_g;
            const uint32_t k = khx_bot ? j : j + 1;
            const bool from_upper = k == 1 && khx_top;
            const uint32_t need = from_upper ? gi + 1 : (k == 0 ? 2 * gi + 1 : 2 * gi + 2);
            invalidate_l1_cache();
            const uint32_t have = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(from_upper ? KWF_SEM_FROM_UPPER : KWF_SEM_FROM_LOWER));
            if (have >= need && cb_pages_reservable_at_back(KWF_CB_HALO_IN, 1)) {
                cb_reserve_back(KWF_CB_HALO_IN, 1);
                cb_push_back(KWF_CB_HALO_IN, 1);
                ++khx_in_i;
            }
        }
    };
    auto khx_drain = [&]() {
        if (khx_out_i < khx_out_n || khx_in_i < khx_in_n) {
            MaybeDeviceZoneScope("writer_khx_drain");
            // The writer can reach this drain while it still owes the reader split-fill batches (weight rounds that
            // landed early): compute then waits for the slab and the halo never comes, unless the fill is served here.
            while (khx_out_i < khx_out_n || khx_in_i < khx_in_n) {
                khx_poll();
                serve_fill();
            }
        }
    };
#else
    auto khx_poll = [&]() {};
    auto khx_drain = [&]() {};
#endif
    // Weight servers: on a server receiver the server multicasts the bias (see the file comment); cb_bias is reserved
    // at kernel start and pushed once the bias flag is set. A core outside the channel reads it: at kernel start, or
    // under kw-fold once round 0 landed (~100 cores reading the same bias pages slowed round 0 by ~1 us when issued at
    // kernel start). Either is pushed as soon as it landed, polled inside the round waits; the first M-block's last
    // round blocks on it.
    constexpr uint32_t TRID_BIAS_LATE = 7;
    conv2d_3x3::TaggedReads<false> late_bias_reads;
    [[maybe_unused]] bool late_bias_pending = false;
    [[maybe_unused]] bool late_bias_issued = false;
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* server_bias_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        weight_servers ? get_semaphore(server_flag_sem_id + nslice_weight_depth) : 0);
    [[maybe_unused]] auto late_bias_landed = [&]() -> bool {
        if (server_receiver) {
            invalidate_l1_cache();
            return *server_bias_flag == VALID;
        }
        return ncrisc_noc_read_with_transaction_id_flushed(late_bias_reads.noc.get_noc_id(), TRID_BIAS_LATE);
    };
    [[maybe_unused]] auto issue_late_bias = [&]() {
        if constexpr (weight_servers && has_bias && (!slab_split || slab_direct)) {
            late_bias_issued = true;
            if (bias_pushed) {
                return;  // pushed in kw-fold's round-0 wait
            }
            if (server_receiver) {
                cb_reserve_back(cb_bias, out_n_tiles);
                late_bias_pending = true;
                return;
            }
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
    if (!kw_fold || server_receiver) {
        issue_late_bias();
    }
    // kw-fold: push the late bias as soon as it landed, from inside the weight waits (the reader decodes it into the
    // final pass's fp32 rows at ~1.3 us per tile, so it must not wait for the block's last round).
    // Slab-direct fused bias at LoFi: the landed bias is rewritten for the fused K-step before it is pushed.
    [[maybe_unused]] auto sd_split_landed_bias = [&]() {
#ifdef SD_BIAS_SPLIT
        invalidate_l1_cache();
        for (uint32_t n = 0; n < out_valid_tiles; ++n) {
            const uint32_t t = get_write_ptr(cb_bias) + n * get_tile_size(cb_bias);
#if SD_BIAS_BF16
            conv2d_3x3::slab_direct::split_bias_tile_bf16(t);
#else
            conv2d_3x3::slab_direct::split_bias_tile_bf8(t);
#endif
        }
#endif
    };
    [[maybe_unused]] auto try_finish_late_bias = [&]() {
        if (late_bias_pending && late_bias_landed()) {
            late_bias_reads.wait(TRID_BIAS_LATE);
            sd_split_landed_bias();
            cb_push_back(cb_bias, out_n_tiles);
            late_bias_pending = false;
        }
    };
    [[maybe_unused]] auto finish_late_bias = [&]() {
        if (late_bias_pending) {
            MaybeDeviceZoneScope("writer_bias");
            while (!late_bias_landed()) {
                serve_fill();
                khx_poll();
            }
            late_bias_reads.wait(TRID_BIAS_LATE);
            sd_split_landed_bias();
            cb_push_back(cb_bias, out_n_tiles);
            late_bias_pending = false;
        }
    };
    // Weight servers: the slot-reuse acks sent so far (rounds nslice_weight_depth .. + server_reuse_acks - 1), and the
    // rounds this receiver takes (kw-fold whole-K keeps its weights resident across M-blocks: one stream).
    [[maybe_unused]] uint32_t server_reuse_acks = 0;
    // Reuse ack j (round nslice_weight_depth + j, slot j % depth). SERVER_SLOT_ACKS (descriptor: weight servers with
    // more rounds than slots): one ack counter per slot on the server (slot 0's is the server's ack semaphore, the
    // others from SERVER_SLOT_ACKS on). On one total counter a core that runs ahead (a short ragged core, or one whose
    // compute frees slots early) acks later slots before a slower core acked an earlier one, and those acks count
    // toward the earlier slot's reuse: the server could overwrite a slot a core was still multiplying (a receiver two
    // acks ahead, e.g. the ragged short core, is enough).
    [[maybe_unused]] auto server_reuse_ack = [&]() {
#ifdef SERVER_SLOT_ACKS
        const uint32_t slot = server_reuse_acks % nslice_weight_depth;
        noc_semaphore_inc(
            slot == 0 ? server_ack_noc
                      : get_noc_addr(
                            get_arg_val<uint32_t>(server_arg + 1),
                            get_arg_val<uint32_t>(server_arg + 2),
                            get_semaphore(SERVER_SLOT_ACKS + slot - 1)),
            1);
#else
        noc_semaphore_inc(server_ack_noc, 1);
#endif
        ++server_reuse_acks;
    };
#if defined(CONV2D_3X3_KW_FOLD) && !KWF_KBLOCKED
    [[maybe_unused]] const uint32_t server_rounds = num_k_blocks;
#else
    [[maybe_unused]] const uint32_t server_rounds = total_k_blocks;
#endif
    // Slab-direct round 0, completed the moment it is in (polled from the forward-slab issue and the round-0 wait),
    // into the slot reserved at kernel start; true once pushed. The DRAM reader: its reads landed, then the bias split,
    // and on the multicast sender the go to this core's reader, which multicasts the round on NoC0 (SD_R0_NOC0, the
    // reader's file comment). A receiver: the reader-multicast's ready flag.
    [[maybe_unused]] auto sd_w0_try = [&]() -> bool {
        if (sd_w0 != 1) {
            return sd_w0 == 2;
        }
        if (sd_w_issued_early) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, TRID_SD_WEIGHTS)) {
                return false;
            }
            finish_weight_k_block(get_write_ptr(cb_weights));
#ifdef SD_R0_NOC0
            if (weight_sender && weight_mcast.should_send(0)) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SD_R0_LOCAL_SEM)) = 1;
            }
#else
            ASSERT(!(weight_sender && weight_mcast.should_send(0)));
#endif
        } else {
#ifdef SD_R0_NOC0
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SD_R0_READY_SEM)) != 1) {
                return false;
            }
#endif
        }
        cb_push_back(cb_weights, weight_k_block_tiles);
        ++k_blocks_issued;
        sd_w0 = 2;
        return true;
    };
    // Round 0 streamed (kw-fold under weight servers, KWF_ROUND0_STREAM = the server's chunks per round): its chunks
    // pushed so far. Semaphore server_flag_sem_id + depth + j (after the bias flag), j >= 1, is the server's early flag
    // "chunks 0 .. j - 1 landed" (conv2d_3x3_weight_server_writer.cpp).
#ifdef KWF_ROUND0_STREAM
    constexpr uint32_t round0_chunks = KWF_ROUND0_STREAM;
#else
    constexpr uint32_t round0_chunks = 1;
#endif
    [[maybe_unused]] uint32_t round0_pushed = 0;
    auto issue_weight_k_block = [&]() {
        if (sd_w0 == 1) {
            // Slab-direct round 0 (taken early): wait for it, serving the fill.
            MaybeDeviceZoneScope("writer_weight_fetch");
            if (sd_w_issued_early) {
                MaybeDeviceZoneScope("writer_sd_weight_land");
                while (!sd_w0_try()) {
                    serve_fill();
                }
            } else {
                while (!sd_w0_try()) {
                    serve_fill();
                }
            }
            if constexpr (weight_servers) {
                finish_late_bias();  // a core outside the servers' channel reads its own round and bias
            }
            return;
        }
        {
            // Back-pressure from compute's matmul.
            MaybeDeviceZoneScope("writer_weight_reserve");
            // Compute may need this core's fill, or sit at a halo-exchange gate (see the exchange above), before it
            // frees the slot: one loop serves both, or the writer can spin on one while compute waits on the other.
            while (!cb_pages_reservable_at_back(cb_weights, weight_k_block_tiles)) {
                serve_fill();
                khx_poll();
            }
            cb_reserve_back(cb_weights, weight_k_block_tiles);
        }
        const uint32_t l1 = get_write_ptr(cb_weights);
        // DRAM read (+ multicast) on a sender or own-stream core; the multicast wait on a receiver.
        MaybeDeviceZoneScope("writer_weight_fetch");
        if (server_receiver) {
            // Weight server round (see the file comment); serves the fill hand-overs while it waits.
            if (k_blocks_issued >= nslice_weight_depth &&
                server_reuse_acks < k_blocks_issued + 1 - nslice_weight_depth) {
                server_reuse_ack();
            }
            // Round k + 1's reuse ack, early: round k is reserved and not pushed, so two free rounds at the back mean
            // compute freed round k + 1's slot.
            const bool early_ack = nslice_weight_depth >= 2 && k_blocks_issued + 1 >= nslice_weight_depth &&
                                   k_blocks_issued + 1 < server_rounds;
            volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(server_flag_sem_id + k_blocks_issued % nslice_weight_depth));
            while (true) {
                invalidate_l1_cache();
                if (*flag == server_flag_valid) {
                    break;
                }
                if (round0_chunks > 1 && k_blocks_issued == 0) {
                    // Round 0 streams: push the chunks the server's early flags report landed.
                    for (uint32_t j = round0_chunks - 1; j > round0_pushed; --j) {
                        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                get_semaphore(server_flag_sem_id + nslice_weight_depth + j)) == server_flag_valid) {
                            cb_push_back(cb_weights, (j - round0_pushed) * (weight_k_block_tiles / round0_chunks));
                            round0_pushed = j;
                            break;
                        }
                    }
                }
                if (early_ack && server_reuse_acks < k_blocks_issued + 2 - nslice_weight_depth &&
                    cb_pages_reservable_at_back(cb_weights, 2 * weight_k_block_tiles)) {
                    server_reuse_ack();
                }
                serve_fill();
                if constexpr (kw_fold) {
                    try_finish_late_bias();
                    // The exchange's first event follows compute's first group: nothing to poll in the round-0 wait.
                    if (k_blocks_issued > 0) {
                        khx_poll();
                    }
                }
            }
            *flag = INVALID;
        } else if (weight_sender && weight_mcast.should_send(k_blocks_issued)) {
#ifdef SD_R0_NOC0
            // Round 0's slot is the reader's NoC0 multicast source until the reader has sent it (SD_R0_LOCAL_SEM 2).
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SD_R0_LOCAL_SEM)) != 2) {
                invalidate_l1_cache();
                serve_fill();
            }
#endif
            read_weight_k_block(l1);
            MaybeDeviceZoneScope("writer_weight_send");
            weight_sender->send(l1, l1, weight_k_block_tiles * weight_tile_bytes);
        } else if (weight_receiver) {
            if constexpr (tile_fill_split || slab_direct) {
                // Round 0 of a slab-direct receiver came over the reader multicast, not this channel.
                receive_serving(weight_mcast, k_blocks_issued - (sd_r0_mcast ? 1 : 0));
            } else {
                weight_receiver->receive(k_blocks_issued);
            }
        } else {
            read_weight_k_block(l1);
        }
        cb_push_back(cb_weights, weight_k_block_tiles - round0_pushed * (weight_k_block_tiles / round0_chunks));
        round0_pushed = 0;
        ++k_blocks_issued;
        if constexpr (weight_servers) {
            if constexpr (kw_fold) {
                // The first n-group's final pass needs the bias: complete it once that group's three rounds are out
                // (the writer may block on a slot only a later n-group frees).
                if (k_blocks_issued == 1) {
                    if (!late_bias_issued) {
                        issue_late_bias();
                    }
                    try_finish_late_bias();
                } else if (k_blocks_issued == 3) {
                    finish_late_bias();
                } else {
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
    // Rounds issued before the split row gather's serving.
    constexpr uint32_t nslice_weight_lookahead = 2;
    static_assert(!nslice_outer || nslice_weight_depth == 1 || nslice_weight_depth >= nslice_weight_lookahead + 2);
    // Every participant pushes one slice per round from the CB base, so slot round % depth is the same L1 address on
    // every core of the column.
    const uint32_t weights_base = get_write_ptr(cb_weights);
    conv2d_3x3::TaggedReads<false> weight_reads;
    constexpr uint32_t TRID_WEIGHTS = 2;  // 2 and 3 alternate; the face gather uses 1
    uint32_t prefetched_round = 0xFFFFFFFF;
    [[maybe_unused]] auto start_weight_reads = [&](uint32_t round, uint32_t y) {
        // The multicasts do not wait for their data to leave L1 (CallerManaged): the slot's last multicast must have
        // left it before it is refilled. A fresh slot (round < depth) never was a multicast source: the flush would
        // only wait for the multicasts just issued from the other slots (round 0's two under NSLICE_ROUND0).
        if (round >= nslice_weight_depth) {
            noc_async_writes_flushed();
        }
        weight_reads.begin(TRID_WEIGHTS + (round & 1));
        uint32_t l1 = weights_base + (round % nslice_weight_depth) * slice_weight_tiles * weight_tile_bytes;
        const uint32_t n0 = y * out_n_tiles;
        const uint32_t valid = tensor_n_tiles - n0 < out_n_tiles ? tensor_n_tiles - n0 : out_n_tiles;
#ifdef NSLICE_WEIGHT_PAGE_RUNS
        // One ND shard per (channel slice, N-slice) round (conv2d_3x3._block_weight_memory_config): its tiles are
        // consecutive pages of one bank in slot order (k-major, out_n_tiles per k), so the round is one read. One-tile
        // reads are NIU-issue-bound (~42 cycles each): with the multicast no longer blocking, their issue ran under
        // it and slowed aspp.convs.3 @ LoFi by 3.1% (20493 -> 21126 ns), which the single read does not.
        if (valid == out_n_tiles) {
            weight_reads.read(
                weights.get_noc_addr(weight_k_tile_start * tensor_n_tiles + n0),
                l1,
                slice_weight_tiles * weight_tile_bytes);
        } else {
            // A short last N-slice (only in a hand-placed tensor: the prepared one splits C_out evenly): a run per k.
            for (uint32_t k = 0; k < nslice_k_tiles; ++k, l1 += out_n_tiles * weight_tile_bytes) {
                weight_reads.read(
                    weights.get_noc_addr((weight_k_tile_start + k) * tensor_n_tiles + n0),
                    l1,
                    valid * weight_tile_bytes);
            }
        }
#else
        for (uint32_t k = 0; k < nslice_k_tiles; ++k) {
            const uint32_t row_page = (weight_k_tile_start + k) * tensor_n_tiles + n0;
            for (uint32_t n = 0; n < valid; ++n) {
                weight_reads.read(weights.get_noc_addr(row_page + n), l1 + n * weight_tile_bytes, weight_tile_bytes);
            }
            l1 += out_n_tiles * weight_tile_bytes;
        }
#endif
        dataflow_kernel_lib::set_read_trid(weight_reads.noc, 0);
    };
#if defined(NSLICE_ROUND0) && !defined(NSLICE_SPREAD)
    // The round-0 reader's read (M-block 0's first owner slice), before anything else of the walk; issue_weight_slice
    // finds it in flight.
    if constexpr (nslice_outer) {
        if (num_m_blocks_this_core > 0 && round0_reader) {
            uint32_t y0 = k_cols;
            for (uint32_t j = 0; j < k_cols && y0 == k_cols; ++j) {
                const uint32_t owner = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
                if (conv2d_3x3::ring_is_owner<num_owning_cols>(owner)) {
                    y0 = conv2d_3x3::ring_slice_of(owner);
                }
            }
            if (y0 < k_cols) {
                MaybeDeviceZoneScope("writer_round0_issue");
                cb_reserve_back(cb_weights, slice_weight_tiles);
                start_weight_reads(0, y0);
                prefetched_round = 0;
            }
        }
    }
#endif
#ifdef NSLICE_SPREAD
    // ---- Spread column stream (see the file comment). Args (round0_arg ..): [rounds read (mask), column rectangle,
    // column dests, spare rectangle, spare dests, rounds relayed (mask)].
    constexpr uint32_t sp_rounds = NSLICE_SPREAD_ROUNDS;
    static_assert(nslice_outer && nslice_weight_depth >= sp_rounds && sp_rounds <= 32);
    constexpr uint32_t SP_TRID0 = 2;  // 2..5: the handshake stream's ids, unused here
    constexpr uint32_t SP_NTRID = 4;
    const bool sp_live = num_m_blocks_this_core > 0;
    const uint32_t sp_mask = sp_live ? get_arg_val<uint32_t>(round0_arg) : 0;
    const uint32_t sp_col_dests = get_arg_val<uint32_t>(round0_arg + 5);
    const uint32_t sp_spare_dests = get_arg_val<uint32_t>(round0_arg + 10);
    const uint32_t sp_relay_mask = (sp_live && sp_spare_dests != 0) ? get_arg_val<uint32_t>(round0_arg + 11) : 0;
    uint32_t sp_mine[sp_rounds];    // rounds this core reads, in order
    uint32_t sp_relays[sp_rounds];  // rounds this core relays, in order
    uint32_t sp_nmine = 0;
    uint32_t sp_nrelay = 0;
    for (uint32_t r = 0; r < sp_rounds; ++r) {
        if ((sp_mask >> r) & 1u) {
            sp_mine[sp_nmine++] = r;
        }
        if ((sp_relay_mask >> r) & 1u) {
            sp_relays[sp_nrelay++] = r;
        }
    }
    // Round r is the M-block's (r mod slices)-th owner slice of the walk order.
    uint32_t sp_order[k_cols];
    uint32_t sp_nslices = 0;
    for (uint32_t j = 0; j < k_cols; ++j) {
        const uint32_t owner = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
        if (conv2d_3x3::ring_is_owner<num_owning_cols>(owner)) {
            sp_order[sp_nslices++] = conv2d_3x3::ring_slice_of(owner);
        }
    }
    uint32_t sp_k_issued = 0;  // this core's reads issued,
    uint32_t sp_k_landed = 0;  // landed (flagged here),
    uint32_t sp_k_sent = 0;    // and multicast down the column
    uint32_t sp_relay_k = 0;
    auto sp_flag = [&](uint32_t r) -> uint32_t { return get_semaphore(NSLICE_SPREAD_FLAG_SEM0 + r); };
    auto sp_landed = [&](uint32_t r) -> bool {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp_flag(r)) == VALID;
    };
    auto sp_slot = [&](uint32_t r) -> uint32_t { return weights_base + r * slice_weight_tiles * weight_tile_bytes; };
    // Round r's slot to the rectangle at arg a (NoC1-ordered), then its flag (same path, after the data).
    auto sp_mcast = [&](uint32_t r, uint32_t a, uint32_t dests) {
        const uint32_t sx = get_arg_val<uint32_t>(a);
        const uint32_t sy = get_arg_val<uint32_t>(a + 1);
        const uint32_t ex = get_arg_val<uint32_t>(a + 2);
        const uint32_t ey = get_arg_val<uint32_t>(a + 3);
        noc_async_write_multicast(
            sp_slot(r),
            get_noc_multicast_addr(sx, sy, ex, ey, sp_slot(r)),
            slice_weight_tiles * weight_tile_bytes,
            dests,
            true);
        noc_semaphore_set_multicast(sp_flag(r), get_noc_multicast_addr(sx, sy, ex, ey, sp_flag(r)), dests);
    };
    auto sp_read = [&](uint32_t r, uint32_t trid) {
        weight_reads.begin(trid);
        const uint32_t y = sp_order[r % sp_nslices];
#if NSLICE_SPREAD_BANKS
        // ND shard (channel slice c, N-slice y) = shard c * num_owning_cols + y, round-robin over the banks, shards one
        // after another in a bank (the descriptor checked the placement): one read, no accessor arithmetic.
        const uint32_t shard = (weight_k_tile_start / nslice_k_tiles) * num_owning_cols + y;
        weight_reads.read(
            get_noc_addr_from_bank_id<true>(
                shard % NSLICE_SPREAD_BANKS,
                weight_addr + (shard / NSLICE_SPREAD_BANKS) * slice_weight_tiles * weight_tile_bytes,
                weight_reads.noc.get_noc_id()),
            sp_slot(r),
            slice_weight_tiles * weight_tile_bytes);
#else
        // Any other placement: start_weight_reads' reads (a run per k under NSLICE_WEIGHT_PAGE_RUNS, else per tile).
        uint32_t l1 = sp_slot(r);
        const uint32_t n0 = y * out_n_tiles;
        const uint32_t valid = tensor_n_tiles - n0 < out_n_tiles ? tensor_n_tiles - n0 : out_n_tiles;
        for (uint32_t k = 0; k < nslice_k_tiles; ++k, l1 += out_n_tiles * weight_tile_bytes) {
#ifdef NSLICE_WEIGHT_PAGE_RUNS
            weight_reads.read(
                weights.get_noc_addr((weight_k_tile_start + k) * tensor_n_tiles + n0), l1, valid * weight_tile_bytes);
#else
            const uint32_t row_page = (weight_k_tile_start + k) * tensor_n_tiles + n0;
            for (uint32_t n = 0; n < valid; ++n) {
                weight_reads.read(weights.get_noc_addr(row_page + n), l1 + n * weight_tile_bytes, weight_tile_bytes);
            }
#endif
        }
#endif
        dataflow_kernel_lib::set_read_trid(weight_reads.noc, 0);
    };
    auto sp_issue = [&]() {
        while (sp_k_issued < sp_nmine && sp_k_issued - sp_k_landed < SP_NTRID) {
            const uint32_t r = sp_mine[sp_k_issued];
            if (r >= NSLICE_SPREAD_LOOKAHEAD && !sp_landed(r - NSLICE_SPREAD_LOOKAHEAD)) {
                break;
            }
            sp_read(r, SP_TRID0 + sp_k_issued % SP_NTRID);
            ++sp_k_issued;
        }
    };
    // Flags every landed read, issues what the lookahead allows, then at most one multicast: a relay of round r before
    // this core's own round r + 1.
    auto sp_poll = [&]() {
        invalidate_l1_cache();
        while (sp_k_landed < sp_k_issued && ncrisc_noc_read_with_transaction_id_flushed(
                                                weight_reads.noc.get_noc_id(), SP_TRID0 + sp_k_landed % SP_NTRID)) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp_flag(sp_mine[sp_k_landed])) = VALID;
            ++sp_k_landed;
        }
        sp_issue();
        const uint32_t own = sp_k_sent < sp_k_landed ? sp_mine[sp_k_sent] : sp_rounds;
        const uint32_t rel = sp_relay_k < sp_nrelay ? sp_relays[sp_relay_k] : sp_rounds;
        const bool own_ok = own < sp_rounds && (own != 1 || sp_landed(0));
        const bool relay_ok = rel < sp_rounds && sp_landed(rel);
        if (relay_ok && (!own_ok || rel < own)) {
            MaybeDeviceZoneScope("writer_spread_relay");
            sp_mcast(rel, round0_arg + 6, sp_spare_dests);
            ++sp_relay_k;
        } else if (own_ok) {
            MaybeDeviceZoneScope("writer_spread_mcast");
            sp_mcast(own, round0_arg + 1, sp_col_dests);
            ++sp_k_sent;
        }
    };
    // This core's duties for the rounds below `end` (its reads multicast, its relays) are done.
    auto sp_idle = [&](uint32_t end) -> bool {
        return (sp_k_sent == sp_nmine || sp_mine[sp_k_sent] >= end) &&
               (sp_relay_k == sp_nrelay || sp_relays[sp_relay_k] >= end);
    };
    if (sp_nmine > 0) {
        MaybeDeviceZoneScope("writer_spread_issue");
        sp_poll();
    }
#else
    auto sp_poll = [&]() {};
#endif
    // The ring credit of M-block credit_m (see nslice_round), given from inside the weight waits as soon as it is
    // ready: the slab is dead (shared-L1 walk) and compute popped the previous M-block's slots (it pushes and pops the
    // whole CB once per M-block, so its acked count is then credit_m whole CBs). Giving it only after the first weight
    // rounds were out held the ring behind them: on aspp.convs.1 / .2 the columns 0, 8, 9, 10 read the same N-slice
    // every round (same owner order), so the same DRAM bank, and their rounds 1-2 land 4-7 us late.
    bool credit_pending = false;
    uint32_t credit_m = 0;
    auto credit_ready = [&]() -> bool {
        if constexpr (alias_slab_combine) {
            if (Semaphore<>(slab_dead_sem_id).value() < credit_m + 1) {
                return false;
            }
        }
        const uintptr_t acked_ptr = (uintptr_t)get_cb_tiles_acked_ptr(cb_combine_slots);
        const uint16_t acked_target = static_cast<uint16_t>(credit_m * k_cols * slot_tiles);
        invalidate_l1_cache();
        const uint16_t acked = static_cast<uint16_t>(reg_read(acked_ptr));
        // Or one whole CB more: this M-block's walk already ended, which it only can without an incoming partial (no
        // real step j >= 1), so no peer writes these slots this block. A non-owner whose only real step is j = 0 pops
        // its slots right after it; with the weight round pushed before a spare forward (or any slow writer) that can
        // come first, and the credit then waited for forever (hung under --dev on a 2-slice, 1-owner shape).
        return acked == acked_target || acked == static_cast<uint16_t>(acked_target + k_cols * slot_tiles);
    };
    // Async bias (NSLICE_ASYNC_BIAS): one read per output column under its own transaction id; pushed once landed.
    bool bias_pending = false;
    constexpr uint32_t TRID_ASYNC_BIAS = 7;
    [[maybe_unused]] auto issue_async_bias = [&]() {
        if constexpr (async_bias) {
            const uint32_t bias_tile_bytes = get_tile_size(cb_bias);
            const auto bias = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);
            cb_reserve_back(cb_bias, out_n_tiles);
            uint32_t l1 = get_write_ptr(cb_bias);
            Noc bias_noc;
            dataflow_kernel_lib::set_read_trid(bias_noc, TRID_ASYNC_BIAS);
            for (uint32_t n = 0; n < out_valid_tiles; ++n, l1 += bias_tile_bytes) {
                noc_async_read(bias.get_noc_addr(out_n_tile_start + n), l1, bias_tile_bytes);
            }
            dataflow_kernel_lib::set_read_trid(bias_noc, 0);
            bias_pending = true;
        }
    };
    auto try_finish_bias = [&](bool block) {
        if constexpr (async_bias) {
            if (bias_pending) {
                while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, TRID_ASYNC_BIAS)) {
                    if (!block) {
                        return;
                    }
                }
                cb_push_back(cb_bias, out_n_tiles);
                bias_pending = false;
            }
        }
    };
    // One credit per M-block to column my_col - 1, the only core that writes this core's slots.
    auto poll_credit = [&]() {
        sp_poll();
        try_finish_bias(false);
        serve_fill();
        if (credit_pending && credit_ready()) {
            MaybeDeviceZoneScope("writer_ring_credit");
            const uint32_t prev_col = my_col == 0 ? k_cols - 1 : my_col - 1;
            combine_credit.up(noc, peer_x(prev_col), peer_y(prev_col), 1);
            credit_pending = false;
        }
    };
    auto reserve_polling = [&](uint32_t pages) {
        while (!cb_pages_reservable_at_back(cb_weights, pages)) {
            poll_credit();
        }
        cb_reserve_back(cb_weights, pages);
    };
    // The fixed-sender multicast receive (ack the sender, then the round's data-ready signal), polling while it waits.
    // Raw for the same reason as receive_serving: ReceiverPipe::receive has no poll form. ct_base_c: the channel's
    // compile-time offset (its semaphore ids), as a std::integral_constant.
    // acked: this round's ack already went out (a forwarder's early ack).
    auto receive_polling_on = [&](auto& wm, uint32_t round, auto ct_base_c, bool acked = false) {
        using WM = std::decay_t<decltype(wm)>;
        if constexpr (WM::active) {
            static_assert(WM::transfer_mode == dataflow_kernel_lib::TransferMode::Multicast && !WM::rotating);
            constexpr uint32_t ct_base = decltype(ct_base_c)::value;
            Semaphore<> data_ready(
                dataflow_kernel_lib::detail::
                    positional_mcast_semaphore<ct_base, dataflow_kernel_lib::mcast_wire::DATA_READY>());
            if constexpr (WM::pre_handshake) {
                if (!acked) {
                    if (weight2_sender) {
                        // A forwarder: the ack lets the column overwrite a slot one of its forwards read.
                        noc_async_writes_flushed();
                    }
                    Semaphore<>(
                        dataflow_kernel_lib::detail::
                            positional_mcast_semaphore<ct_base, dataflow_kernel_lib::mcast_wire::CONSUMER_READY>())
                        .up(noc, wm.sender_x(), wm.sender_y(), 1);
                }
            }
            if constexpr (WM::signal == dataflow_kernel_lib::DataReadySignal::Counter) {
                while (data_ready.value() < round + 1) {
                    poll_credit();
                }
            } else {
                while (data_ready.value() != VALID) {
                    poll_credit();
                }
                data_ready.set(INVALID);
            }
        }
    };
    auto receive_polling = [&](auto& wm, uint32_t round, bool acked = false) {
        receive_polling_on(
            wm,
            round,
            std::integral_constant<uint32_t, get_named_compile_time_arg_val("weight_mcast_ct_offset")>{},
            acked);
    };
    // A sender's wait for every receiver's ack of its round, polling the ring credit (the helper's send then finds them
    // at once). Generic: kernel_main is not a template, so only a generic lambda discards an absent channel's branch.
    auto wait_acks_polling_on = [&](auto& wm, auto ct_base_c) {
        using WM = std::decay_t<decltype(wm)>;
        if constexpr (WM::active) {
            if constexpr (WM::pre_handshake) {
                Semaphore<> ready(dataflow_kernel_lib::detail::positional_mcast_semaphore<
                                  decltype(ct_base_c)::value,
                                  dataflow_kernel_lib::mcast_wire::CONSUMER_READY>());
                while (ready.value() < WM::ack_count) {
                    poll_credit();
                }
            }
        }
    };
    // The column sender's ack of a forwarder's next round (see issue_weight_slice).
    auto early_ack_on = [&](auto& wm, auto ct_base_c) -> bool {
        using WM = std::decay_t<decltype(wm)>;
        if constexpr (WM::active) {
            if constexpr (WM::pre_handshake) {
                noc_async_writes_flushed();
                Semaphore<>(dataflow_kernel_lib::detail::positional_mcast_semaphore<
                                decltype(ct_base_c)::value,
                                dataflow_kernel_lib::mcast_wire::CONSUMER_READY>())
                    .up(noc, wm.sender_x(), wm.sender_y(), 1);
                return true;
            }
        }
        return false;
    };
    // A forwarder's round whose slot it already reserved and acked.
    uint32_t early_acked_round = 0xFFFFFFFF;
    // Weights of slice y; next_y is the M-block's next slice, or k_cols after the last.
    [[maybe_unused]] auto issue_weight_slice = [&](uint32_t y, uint32_t next_y) {
        const uint32_t round = k_blocks_issued;
        bool column_received = false;
        MaybeDeviceZoneScope("writer_weight_fetch");
#ifdef NSLICE_SPREAD
        {
            // The round lands in its own slot (see the spread stream): push it once flagged.
            poll_credit();
            while (!sp_landed(round)) {
                poll_credit();
            }
            cb_reserve_back(cb_weights, slice_weight_tiles);
            cb_push_back(cb_weights, slice_weight_tiles);
            if (round == 0) {
                issue_async_bias();
            }
            ++k_blocks_issued;
            return;
        }
#endif
        // Every round, not only inside the waits: once a round lands before its receiver waits for it, a forwarder that
        // never waits gave the credit only at the blocking wait after round depth, ~2.5 us late, and its grid row's
        // ring stalled ~2 us behind it (res4.x).
        poll_credit();
        bool round0_done = false;
#ifdef NSLICE_ROUND0
        // Round 0 of a column participant (see the file comment).
        constexpr uint32_t m1_ct = get_named_compile_time_arg_val("weight_mcast_ct_offset");
        const bool column_sender = weight_sender && weight_mcast.should_send(0);
        // The column sender's round-1 read: its slot is fresh, so it goes out under round 0's multicasts.
        auto prefetch_round1 = [&]() {
            if (column_sender && nslice_weight_depth > 1 && next_y < k_cols &&
                cb_pages_reservable_at_back(cb_weights, 2 * slice_weight_tiles)) {
                cb_reserve_back(cb_weights, 2 * slice_weight_tiles);
                start_weight_reads(1, next_y);
                prefetched_round = 1;
            }
        };
        if (round == 0 && !weight2_receiver && (weight_receiver || column_sender)) {
            round0_done = true;
            if (round0_reader) {
                if (prefetched_round != 0) {
                    reserve_polling(slice_weight_tiles);
                    start_weight_reads(0, y);
                }
                prefetch_round1();
                while (!ncrisc_noc_read_with_transaction_id_flushed(weight_reads.noc.get_noc_id(), TRID_WEIGHTS)) {
                    poll_credit();
                }
                weight_reads.wait(TRID_WEIGHTS);
                // The flag's source: a cell nothing else writes on this core: the data-ready cell on the sender (the
                // helper's own flag source), the consumer-ready cell on a receiver (receivers ack the sender's).
                const uint32_t flag_src = get_semaphore(
                    column_sender
                        ? dataflow_kernel_lib::detail::
                              positional_mcast_semaphore<m1_ct, dataflow_kernel_lib::mcast_wire::DATA_READY>()
                        : dataflow_kernel_lib::detail::
                              positional_mcast_semaphore<m1_ct, dataflow_kernel_lib::mcast_wire::CONSUMER_READY>());
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_src) = VALID;
                MaybeDeviceZoneScope("writer_round0_send");
                const uint32_t sx = get_arg_val<uint32_t>(round0_arg + 1);
                const uint32_t sy = get_arg_val<uint32_t>(round0_arg + 2);
                const uint32_t ex = get_arg_val<uint32_t>(round0_arg + 3);
                const uint32_t ey = get_arg_val<uint32_t>(round0_arg + 4);
                const uint32_t dests = get_arg_val<uint32_t>(round0_arg + 5);
                noc_async_write_multicast(
                    weights_base,
                    get_noc_multicast_addr(sx, sy, ex, ey, weights_base),
                    slice_weight_tiles * weight_tile_bytes,
                    dests,
                    true);
                noc_semaphore_set_multicast(
                    flag_src,
                    get_noc_multicast_addr(
                        sx,
                        sy,
                        ex,
                        ey,
                        get_semaphore(
                            dataflow_kernel_lib::detail::
                                positional_mcast_semaphore<m1_ct, dataflow_kernel_lib::mcast_wire::DATA_READY>())),
                    dests);
            } else {
                reserve_polling(slice_weight_tiles);
                if (column_sender) {
                    // Round 0 comes from the column's round-0 reader into this core's data-ready cell.
                    prefetch_round1();
                    Semaphore<> ready(
                        dataflow_kernel_lib::detail::
                            positional_mcast_semaphore<m1_ct, dataflow_kernel_lib::mcast_wire::DATA_READY>());
                    while (ready.value() != VALID) {
                        poll_credit();
                    }
                    ready.set(INVALID);
                } else {
                    receive_polling(weight_mcast, round, true);
                    column_received = true;
                }
            }
            issue_async_bias();
        }
#endif
        if (round0_done) {
        } else if (weight2_receiver) {
            // Spare-column member: the column stream's second multicast.
            reserve_polling(slice_weight_tiles);
            receive_polling_on(
                weight_mcast2,
                round,
                std::integral_constant<uint32_t, get_named_compile_time_arg_val("weight_mcast2_ct_offset")>{},
#ifdef NSLICE_ROUND0
                // Round 0 comes from the column sender into a fresh slot: no ack to the forwarder.
                round == 0
#else
                false
#endif
            );
            // Read once round 0 landed: the column stream, which paces the walk, starts first.
            if (round == 0) {
                issue_async_bias();
            }
        } else if (!weight_receiver || (weight_sender && weight_mcast.should_send(round))) {
            // A sender spends its rounds issuing reads and multicasting, rarely in a wait: it polls the credit between
            // the steps too (res5: a sender that reached the credit only after its first rounds gave it ~3 us late, and
            // its grid row's ring paid for it).
            poll_credit();
            if (prefetched_round != round) {
                reserve_polling(slice_weight_tiles);
                start_weight_reads(round, y);
                poll_credit();
            }
            // Prefetch only into a slot that is already free: a blocking reserve here waits for compute, and held this
            // round's multicast (and, before the credit, the ring) behind it.
            if (nslice_weight_depth > 1 && next_y < k_cols &&
                cb_pages_reservable_at_back(cb_weights, 2 * slice_weight_tiles)) {
                cb_reserve_back(cb_weights, 2 * slice_weight_tiles);
                start_weight_reads(round + 1, next_y);
                prefetched_round = round + 1;
                poll_credit();
            }
            while (!ncrisc_noc_read_with_transaction_id_flushed(
                weight_reads.noc.get_noc_id(), TRID_WEIGHTS + (round & 1))) {
                poll_credit();
            }
            weight_reads.wait(TRID_WEIGHTS + (round & 1));
            if (weight_sender && weight_mcast.should_send(round)) {
                const uint32_t l1 =
                    weights_base + (round % nslice_weight_depth) * slice_weight_tiles * weight_tile_bytes;
                // Returns once the multicast is issued: the next round's reads then go out under it.
                weight_sender->template send<dataflow_kernel_lib::SourceL1Guard::CallerManaged>(
                    l1, l1, slice_weight_tiles * weight_tile_bytes);
            }
            // Read once this core's round 0 is out (see the spare member's).
            if (round == 0) {
                issue_async_bias();
            }
            poll_credit();
        } else {
            reserve_polling(slice_weight_tiles);
            receive_polling(weight_mcast, round, early_acked_round == round);
            column_received = true;
            if (round == 0) {
                issue_async_bias();
            }
        }
        // The round is this core's own first: the forward only reads its slot, which is reserved again (overwritten)
        // only after the forward's source reads completed, on this RISC (forwarding before the push held the
        // forwarders' own walks one forward behind: res5.0's forwarders ended ~1 us after their rows).
        cb_push_back(cb_weights, slice_weight_tiles);
#ifdef NSLICE_ROUND0
        if (round == 0 && round0_spare_dests != 0) {
            // The column sender relays round 0 to its slice's spare members (on its own row), after its push (the
            // multicast only reads the slot). Its data-ready cell is its own flag source from round 1 on.
            MaybeDeviceZoneScope("writer_round0_relay");
            const uint32_t flag_src =
                get_semaphore(dataflow_kernel_lib::detail::
                                  positional_mcast_semaphore<m1_ct, dataflow_kernel_lib::mcast_wire::DATA_READY>());
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_src) = VALID;
            const uint32_t sx = get_arg_val<uint32_t>(round0_arg + 6);
            const uint32_t sy = get_arg_val<uint32_t>(round0_arg + 7);
            const uint32_t ex = get_arg_val<uint32_t>(round0_arg + 8);
            const uint32_t ey = get_arg_val<uint32_t>(round0_arg + 9);
            noc_async_write_multicast(
                weights_base,
                get_noc_multicast_addr(sx, sy, ex, ey, weights_base),
                slice_weight_tiles * weight_tile_bytes,
                round0_spare_dests,
                true);
            noc_semaphore_set_multicast(
                flag_src,
                get_noc_multicast_addr(
                    sx,
                    sy,
                    ex,
                    ey,
                    get_semaphore(dataflow_kernel_lib::detail::positional_mcast_semaphore<
                                  get_named_compile_time_arg_val("weight_mcast2_ct_offset"),
                                  dataflow_kernel_lib::mcast_wire::DATA_READY>())),
                round0_spare_dests);
        }
#endif
        if (weight2_sender) {
            // A forwarder acks the column's next round (its slot reserved, if already free) before it forwards this
            // one: acking only after the forward put the forward (~0.35-0.45 us) between every two column multicasts.
            // Safe: the next round lands in another slot, and this round's data-ready flag is already cleared.
            if (column_received && next_y < k_cols && cb_pages_reservable_at_back(cb_weights, slice_weight_tiles)) {
                cb_reserve_back(cb_weights, slice_weight_tiles);
                if (early_ack_on(
                        weight_mcast,
                        std::integral_constant<uint32_t, get_named_compile_time_arg_val("weight_mcast_ct_offset")>{})) {
                    early_acked_round = round + 1;
                }
            }
            // Spare-column groups: forward the round to the spare members of this slice. A member acks once the
            // round's slot is free, rounds [0, depth) use fresh slots, and this core gives its ring credit before round
            // depth, so no member waits on anything this core still owes. The ack wait polls the credit; the send
            // returns once the multicast is issued (the slot is acked for reuse only after a flush).
#ifdef NSLICE_ROUND0
            // Round 0 went to the spare members from the column sender.
            if (round != 0)
#endif
            {
                MaybeDeviceZoneScope("writer_spare_forward");
                const uint32_t l1 =
                    weights_base + (round % nslice_weight_depth) * slice_weight_tiles * weight_tile_bytes;
                wait_acks_polling_on(
                    weight_mcast2,
                    std::integral_constant<uint32_t, get_named_compile_time_arg_val("weight_mcast2_ct_offset")>{});
                weight2_sender->template send<dataflow_kernel_lib::SourceL1Guard::CallerManaged>(
                    l1, l1, slice_weight_tiles * weight_tile_bytes);
            }
        }
        ++k_blocks_issued;
    };
    // before_sends runs once the first weight rounds are out, before the credit (the split row gather's serving).
    // A spare-column member walks its slice's owner order, as its column does, so each forward is that round.
    [[maybe_unused]] auto nslice_round = [&](uint32_t m_block_idx, auto&& before_sends) {
        uint32_t order[k_cols];
        uint32_t num_slices = 0;
        for (uint32_t j = 0; j < k_cols; ++j) {
            const uint32_t owner = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
            if (conv2d_3x3::ring_is_owner<num_owning_cols>(owner)) {
                order[num_slices++] = conv2d_3x3::ring_slice_of(owner);
            }
        }
        uint32_t weights_issued = 0;
        auto issue_weights_through = [&](uint32_t end) {
            for (; weights_issued < end && weights_issued < num_slices; ++weights_issued) {
                issue_weight_slice(
                    order[weights_issued], weights_issued + 1 < num_slices ? order[weights_issued + 1] : k_cols);
            }
        };
        // Rounds [0, depth) of the M-block reuse slots of earlier M-blocks only, so issuing them never waits on this
        // block's walk: they go out before the credit, which their waits poll. Every later round can wait on compute,
        // which can wait on the ring, so the credit is given (blocking) before them.
        credit_m = m_block_idx;
        credit_pending = true;
        issue_weights_through(1 + nslice_weight_lookahead);
        before_sends();
        issue_weights_through(nslice_weight_depth);
        if (credit_pending) {
            // The combine slots are the slab's L1 (shared-L1 walk): column my_col - 1 may write them only once this
            // core's gather landed.
            MaybeDeviceZoneScope("writer_slab_dead_wait");
            while (!credit_ready()) {
                serve_fill();
                try_finish_bias(false);
                sp_poll();
            }
            poll_credit();
        }
        issue_weights_through(num_slices);
#ifdef NSLICE_SPREAD
        // This M-block's rounds: every multicast and relay of them out (a later M-block's wait for the next).
        while (!sp_idle((m_block_idx + 1) * num_slices)) {
            poll_credit();
        }
#endif
        try_finish_bias(true);
        noc_async_writes_flushed();
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
#if defined(CONV2D_3X3_SLAB_DIRECT) && defined(SD_STREAM)
        {
            // Streamed: the pieces at and above flat pixel B (the higher shards), the first window of chunks issued.
            const uint32_t B = (my_shard + 1) * shard_rows;
            sd_wstream.reset(SD_SLAB_PX);
            sd_w_base = m_block_idx * (SD_SLAB_PX + 1);
            // The round-0 DRAM reader completes round 0 (bias split, multicast go) as soon as it landed, from inside
            // the planning and the issue (a receiver only waits for it later: its fill issue is the critical path).
            // Under weight servers it completes in issue_weight_k_block, with its late bias.
            const bool w0_poll = !weight_servers && sd_w_issued_early && sd_w0 == 1;
            if (w0_poll) {
                sd_w0_try();
            }
            conv2d_3x3::slab_direct::Geom<
                input_height,
                input_width,
                output_width,
                stride,
                padding,
                dilation,
                SD_GAP,
                SD_PITCH,
                input_page_bytes,
                SD_HALF_BYTES>::fill_pieces<band_last_tap>(o_first, o_end, [&](uint32_t lo, uint32_t hi, uint32_t off) {
                const uint32_t px = off / input_page_bytes;
                if (hi > B) {
                    sd_wstream.add(lo > B ? lo : B, hi, lo > B ? px : px + (B - lo));
                }
                if (w0_poll) {
                    sd_w0_try();
                }
            });
            while (sd_wstream.issue_one(sd_wread)) {
                if (w0_poll) {
                    sd_w0_try();
                }
            }
            return;
        }
#endif
#ifdef CONV2D_3X3_SLAB_DIRECT
        // Slab-direct: the padded slab layout (conv2d_3x3_slab_direct.hpp), the higher shards' pieces, under their
        // own transaction id (see the M-block loop).
        Noc sd_fill_noc;
        dataflow_kernel_lib::set_read_trid(sd_fill_noc, 9);
        conv2d_3x3::slab_direct::Geom<
            input_height,
            input_width,
            output_width,
            stride,
            padding,
            dilation,
            SD_GAP,
            SD_PITCH,
            input_page_bytes,
            SD_HALF_BYTES>::fill_pieces<band_last_tap>(o_first, o_end, [&](uint32_t lo, uint32_t hi, uint32_t off) {
            uint32_t l1 = slab_l1 + off;
            for (uint32_t page = lo; page < hi;) {
                const uint32_t run = input.num_contiguous_pages(page, hi);
                if (page / shard_rows > my_shard) {
                    noc_async_read(input.get_noc_addr(page), l1, run * input_page_bytes);
                }
                page += run;
                l1 += run * input_page_bytes;
            }
        });
        dataflow_kernel_lib::set_read_trid(sd_fill_noc, 0);
        if (true) {
            // slab_fwd_done follows the block's weight round (see the M-block loop).
            return;
        }
#endif
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

#if defined(CONV2D_3X3_SLAB_DIRECT)
    // Slab-direct round 0 ahead of the fill: a receiver of the sender's reader-multicast (SD_R0_NOC0) reserves its
    // slot, with no ack: the slot is fresh at kernel start (see the reader's file comment), so the multicast may
    // already have landed. Its DRAM reader issued it at kernel entry; either completes it the moment it is in
    // (sd_w0_try).
    if (num_m_blocks_this_core > 0 && !server_receiver) {
        MaybeDeviceZoneScope("writer_sd_weight_early");
#ifdef SD_R0_NOC0
        if (!sd_w_issued_early && weight_receiver) {
            cb_reserve_back(cb_weights, weight_k_block_tiles);
            sd_r0_mcast = std::decay_t<decltype(weight_mcast)>::active;
        }
#endif
        sd_w0 = (sd_w_issued_early || sd_r0_mcast) ? 1 : 0;
        if constexpr (!weight_servers) {
            sd_w0_try();
        }
    }
#endif
    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        if constexpr (slab_split) {
#ifdef CONV2D_3X3_SLAB_DIRECT
            // Slab-direct: compute multiplies as soon as the slab and the weights are in. The forward slab reads go
            // out first (transaction id 9, or streamed: the first window of chunks), then the block's weight round (a
            // DRAM read + multicast, or the multicast wait, which polls the slab reads); slab_fwd_done goes up as soon
            // as they landed (streamed: as each chunk lands, sd_stream_serve). The fused bias comes
            // with the weight rounds (SD_BIAS_IN_WEIGHTS) or from the weight server; otherwise (the bias-add pass) the
            // bias reads (block 0, id 7) go out only once this core's slab landed: issued with the fill, ~100 cores
            // reading the same bias pages slowed the high-index cores' forward fill erratically (256x512x8 s2, 103
            // cores: 10-18 us against 8 us).
            fill_forward_slab(m_block_idx);
            [[maybe_unused]] auto sd_issue_bias = [&]() {
                if constexpr (has_bias) {
                    cb_reserve_back(cb_bias, out_n_tiles);
                    const uint32_t bias_tile_bytes = get_tile_size(cb_bias);
                    const auto bias = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);
                    uint32_t l1 = get_write_ptr(cb_bias);
                    Noc bias_noc;
                    dataflow_kernel_lib::set_read_trid(bias_noc, 7);
                    for (uint32_t n = 0; n < out_valid_tiles; ++n, l1 += bias_tile_bytes) {
                        noc_async_read(bias.get_noc_addr(out_n_tile_start + n), l1, bias_tile_bytes);
                    }
                    dataflow_kernel_lib::set_read_trid(bias_noc, 0);
                }
            };
            sd_slab_pending = true;
            while (k_blocks_issued < (m_block_idx + 1) * num_k_blocks) {
                issue_weight_k_block();
            }
            {
                MaybeDeviceZoneScope("writer_slab_bias_wait");
                bool slab_done = !sd_slab_pending;
                // Block 0 reads the bias (and pushes cb_bias even if the core owns no valid output column), unless it
                // comes with the weight rounds (SD_BIAS_IN_WEIGHTS) or from a weight server (the late bias above).
#ifdef SD_BIAS_IN_WEIGHTS
                constexpr bool own_bias = false;
#else
                constexpr bool own_bias = has_bias && !weight_servers;
#endif
                const bool bias_block = own_bias && m_block_idx == 0;
                bool bias_issued = !bias_block;
                uint32_t bias_left = bias_block ? out_valid_tiles : 0;
                while (!slab_done || !bias_issued || bias_left > 0) {
#ifdef SD_STREAM
                    if (!slab_done) {
                        serve_fill();
                        slab_done = !sd_slab_pending;
                    }
#else
                    if (!slab_done && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 9)) {
                        Semaphore<>(slab_fwd_done_sem_id).up(1);
                        slab_done = true;
                        sd_slab_pending = false;
                    }
#endif
                    if (slab_done && !bias_issued) {
                        sd_issue_bias();
                        bias_issued = true;
                        if (bias_left == 0) {
                            cb_push_back(cb_bias, out_n_tiles);  // no valid output column: nothing was read
                        }
                    }
                    if constexpr (own_bias) {
                        if (bias_issued && bias_left > 0 && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 7)) {
                            bias_left = 0;
                            cb_push_back(cb_bias, out_n_tiles);
                        }
                    }
                }
            }
#else
            fill_forward_slab(m_block_idx);
            if (m_block_idx == 0) {
                prefetch_bias();
            }
#endif
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
                // The first n-group's final pass needs the bias: read it once that group's three rounds are out, before
                // the writer can block on a weight slot only a later n-group frees.
                if constexpr (!(weight_servers && !slab_split)) {
                    if (k_blocks_issued == 3) {
                        kwf_bias_read();
                    }
                }
#endif
            }
            khx_drain();
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
#ifdef KWF_XCHG
    // The last semaphore write of the halo exchange.
    noc_async_write_barrier();
#endif
    if constexpr (k_split || weight_servers) {
        noc_async_atomic_barrier();
    }
}
