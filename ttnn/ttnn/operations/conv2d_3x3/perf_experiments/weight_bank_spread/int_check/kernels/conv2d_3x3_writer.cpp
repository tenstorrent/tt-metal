// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 writer (BRISC / NoC1): prefetch_bias once, then per M-block stream_weight_blocks and
// store_block. Weight K-blocks of the next M-block are prefetched between output row groups, but only
// into slots that are already free, so the writer never blocks on cb_weights while compute waits on
// the output CB.
//
// Sharded output: cb_output_tiles is backed on the output shard, so compute packs in place and
// store_block is empty. The writer only streams bias and weights; compute never waits on the output
// CB there, so blocking weight reserves cannot deadlock.
//
// Weight multicast (regime R1, the `weight_mcast` channel): every K-block is one round. The sender
// reads it from DRAM into its own cb_weights slot and multicasts it to the same slot on every
// receiver; a receiver acks only after reserving that slot, so the slot is free. All participants
// push the same quanta in the same order, so their cb_weights write pointers agree every round.
// A core outside the channel (one with fewer M-blocks, or an absent channel) reads its own weights.
// Under a K split the channel has one group per grid column, whose sender reads that column's K-slice.
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
// Split TILE fill (HEIGHT-sharded TILE input, pure weight-multicast receivers): the reader reserves each cb_input_tiles
// batch and, when it holds tile-rows of higher-index input shards, hands it over (fill_go); the writer reads those rows
// on NoC1 into the batch's ring slot and counts it (fill_done), and the reader pushes the batch after both halves. The
// writer serves hand-overs wherever it waits (weight reserve and receive, bias read, row-gather serving), because
// compute untilizes a block before it multiplies it.
//
// Split face-direct gather: after issuing each K-block's weights the writer waits for the reader's gather_go (the
// in0 block is reserved and the slab complete), then writes every odd row group (in0_stream_rows tile-rows) on
// NoC1, raising gather_done after each; the reader pushes the groups in order.
//
// N-slice-outer K split (the default under a K split when the resident in0 fits L1): compute multiplies one owner
// N-slice at a time over the whole K, in the order of conv2d_3x3_nslice_order.hpp, so the partials of slice y are
// final while slice y + 1 multiplies. The writer streams each slice's weights (K x out_n tiles, one multicast round
// per slice down the column) two rounds ahead of the slice it sends next; the column's weight-reading core issues
// round r + 1's DRAM reads (own transaction id) before it multicasts round r. Each slice is sent by one RISC: the
// -d slices and the own slice by the writer on NoC1 from cb_matmul_partials, the +d slices by the reader on NoC0
// from cb_slice_r. Both RISCs wait on the owners' credits and bump an owner's arrival count after their write to it
// is acked; the owner pushes its slots after k_cols arrivals, as in the K-blocked round.
//
// Shared-L1 N-slice walk (whenever every core runs one M-block): cb_combine_slots
// shares cb_halo_slab's L1 and cb_combined cb_zero_src's, both dead once the core's gather landed. The owner's credits
// therefore wait for the reader's slab_dead, so no peer writes a slot before then; the first weight rounds and the
// split row gather's serving still go out before that wait.
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
    constexpr auto weight_args = TensorAccessorArgs<64>();
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

    constexpr uint32_t weight_k_block_tiles = block_k_tiles * block_n_tiles;
    constexpr uint32_t row_group_tiles = out_subblock_h * block_n_tiles;
    constexpr uint32_t row_groups_per_block = block_m_tiles / out_subblock_h;
    const uint32_t weight_tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, weight_tile_bytes);

    // K split weight layout (prepare_conv2d_weights): inside a column's page range, owner slice y's tiles (K_col x
    // valid, dense index j = k * valid + n) are the page run [P_y, P_y + L). Page P_y + m + NUM_DRAM_BANKS * t holds
    // dense tile cum(m) + t, so the round's tiles in one bank are one contiguous bank range and one contiguous L1
    // range: a round is one NoC read per bank. Every other placement keeps row-major pages (k * tensor_n_tiles + n).
    constexpr uint32_t num_banks = NUM_DRAM_BANKS;
    constexpr uint32_t col_k_tiles = block_k_tiles * num_k_blocks;  // the column's K-slice
    const uint32_t col_page_base = weight_k_tile_start * tensor_n_tiles;
    // Page of tile (k, n) of this column's K-slice under the K split layout.
    [[maybe_unused]] auto k_split_weight_page = [&](uint32_t k, uint32_t n) -> uint32_t {
        const uint32_t n0 = (n / out_n_tiles) * out_n_tiles;
        const uint32_t valid = tensor_n_tiles - n0 < out_n_tiles ? tensor_n_tiles - n0 : out_n_tiles;
        const uint32_t run_page = col_page_base + n0 * col_k_tiles;
        const uint32_t j = k * valid + (n - n0);
        const uint32_t q = col_k_tiles * valid / num_banks;
        const uint32_t r = col_k_tiles * valid % num_banks;
        if (j < r * (q + 1)) {
            return run_page + j / (q + 1) + num_banks * (j % (q + 1));
        }
        const uint32_t jj = j - r * (q + 1);
        return run_page + r + jj / q + num_banks * (jj % q);
    };

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
    // With the polling split TILE fill the bias reads are issued later, serving the first block's hand-overs.
    if constexpr (!slab_split && !tile_fill_split) {
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
    conv2d_3x3::TaggedReads<false> fill_reads;
    constexpr uint32_t TRID_FILL = 6;
    // Moves the cursor to the next batch holding higher shards' rows; false past the last block.
    [[maybe_unused]] auto fill_next = [&]() -> bool {
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
    [[maybe_unused]] auto fill_one = [&]() {
        MaybeDeviceZoneScope("writer_fill");
        const auto input = TensorAccessor(input_args, input_addr, input_page_bytes);
        const uint32_t slot =
            fill_ring_base + ((fill_m * fill_batches_per_block + fill_b) % input_tile_depth_batches) * fill_batch_bytes;
        fill_reads.begin(TRID_FILL);
        conv2d_3x3::tile_batch_rows<untilize_batch_rows>(
            fill_plan, fill_b, [&](uint32_t row0, uint32_t rows, uint32_t batch_row) {
                const uint32_t lo = row0 > fill_hi_row ? row0 : fill_hi_row;
                if (lo >= row0 + rows) {
                    return;
                }
                uint32_t l1 = slot + (batch_row + lo - row0) * fill_row_bytes;
                for (uint32_t page = lo * fill_c_in_tiles, end_page = (row0 + rows) * fill_c_in_tiles;
                     page < end_page;) {
                    const uint32_t run = input.num_contiguous_pages(page, end_page);
                    fill_reads.read(input.get_noc_addr(page), l1, run * input_page_bytes);
                    page += run;
                    l1 += run * input_page_bytes;
                }
            });
        fill_reads.wait(TRID_FILL);
        dataflow_kernel_lib::set_read_trid(fill_reads.noc, 0);
        Semaphore<>(slab_fwd_done_sem_id).up(1);
        ++fill_served;
        if (++fill_b == fill_batches_per_block) {
            fill_b = 0;
            ++fill_m;
        }
    };
    // Every batch the reader has handed over so far.
    [[maybe_unused]] auto serve_fill = [&]() {
        if constexpr (tile_fill_split) {
            while (Semaphore<>(slab_free_sem_id).value() > fill_served) {
                fill_next();
                fill_one();
            }
        }
    };
    // The rest of the kernel's batches, waiting for each hand-over.
    [[maybe_unused]] auto drain_fill = [&]() {
        if constexpr (tile_fill_split) {
            while (fill_next()) {
                Semaphore<>(slab_free_sem_id).wait_min(fill_served + 1);
                fill_one();
            }
        }
    };

    // ---- stream_weight_blocks: K-block tiles k-major, n-fastest (matmul in1 layout).
    const uint32_t total_k_blocks = num_m_blocks_this_core * num_k_blocks;
    uint32_t k_blocks_issued = 0;
    auto read_weight_k_block = [&](uint32_t l1) {
        const uint32_t k_tile0 = (k_blocks_issued % num_k_blocks) * block_k_tiles;
        for (uint32_t k = 0; k < block_k_tiles; ++k) {
            const uint32_t row_page = (weight_k_tile_start + k_tile0 + k) * tensor_n_tiles + n_tile_start;
            for (uint32_t n = 0; n < block_n_tiles; ++n, l1 += weight_tile_bytes) {
                noc_async_read_page(
                    k_split ? k_split_weight_page(k_tile0 + k, n_tile_start + n) : row_page + n, weights, l1);
            }
        }
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
    if constexpr (!slab_split && tile_fill_split) {
        prefetch_bias_serving();
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
        if (weight_sender && weight_mcast.should_send(k_blocks_issued)) {
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
    constexpr uint32_t slice_tiles = block_m_tiles * out_n_tiles;
    // Rounds issued ahead of the slice the writer sends next.
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
        // One read per bank (see k_split_weight_page): bank chunk m holds dense tiles [cum(m), cum(m) + cnt) at pages
        // run_page + m + NUM_DRAM_BANKS * t, contiguous in the bank (interleaved page p sits at (p / NUM_DRAM_BANKS) x
        // the page size in bank p % NUM_DRAM_BANKS; prepare_conv2d_weights emits page size == tile size).
        const uint32_t run_page = col_page_base + n0 * nslice_k_tiles;
        const uint32_t len = nslice_k_tiles * valid;
        const uint32_t q = len / num_banks;
        const uint32_t r = len % num_banks;
        const uint32_t chunks = len < num_banks ? len : num_banks;
        uint32_t j = 0;  // dense tile index, k * valid + n
        for (uint32_t m = 0; m < chunks; ++m) {
            uint32_t cnt = q + (m < r ? 1 : 0);
            uint64_t src = weights.get_noc_addr(run_page + m);
            if (valid == out_n_tiles) {
                weight_reads.read(src, l1 + j * weight_tile_bytes, cnt * weight_tile_bytes);
                j += cnt;
                continue;
            }
            // Short last slice: L1 rows are out_n_tiles apart, so the chunk splits at row ends.
            while (cnt > 0) {
                const uint32_t k = j / valid;
                const uint32_t n = j - k * valid;
                const uint32_t seg = valid - n < cnt ? valid - n : cnt;
                weight_reads.read(src, l1 + (k * out_n_tiles + n) * weight_tile_bytes, seg * weight_tile_bytes);
                src += seg * weight_tile_bytes;
                j += seg;
                cnt -= seg;
            }
        }
        dataflow_kernel_lib::set_read_trid(weight_reads.noc, 0);
    };
    // Weights of slice y; next_y is the M-block's next slice, or k_cols after the last.
    [[maybe_unused]] auto issue_weight_slice = [&](uint32_t y, uint32_t next_y) {
        const uint32_t round = k_blocks_issued;
        MaybeDeviceZoneScope("writer_weight_fetch");
        if (!weight_receiver || (weight_sender && weight_mcast.should_send(round))) {
            if (prefetched_round != round) {
                cb_reserve_back(cb_weights, slice_weight_tiles);
                start_weight_reads(round, y);
            }
            if (nslice_weight_depth > 1 && next_y < k_cols) {
                cb_reserve_back(cb_weights, 2 * slice_weight_tiles);
                start_weight_reads(round + 1, next_y);
                prefetched_round = round + 1;
            }
            weight_reads.wait(TRID_WEIGHTS + (round & 1));
            if (weight_sender && weight_mcast.should_send(round)) {
                const uint32_t l1 =
                    weights_base + (round % nslice_weight_depth) * slice_weight_tiles * weight_tile_bytes;
                weight_sender->send(l1, l1, slice_weight_tiles * weight_tile_bytes);
            }
        } else {
            cb_reserve_back(cb_weights, slice_weight_tiles);
            weight_receiver->receive(round);
        }
        cb_push_back(cb_weights, slice_weight_tiles);
        ++k_blocks_issued;
    };
    // before_sends runs once the first weight rounds are out, before the first slice send (the split row gather).
    [[maybe_unused]] auto nslice_round = [&](uint32_t m_block_idx, auto&& before_sends) {
        auto give_credits = [&]() {
            if (out_valid_tiles > 0) {
                cb_reserve_back(cb_combine_slots, k_cols * slot_tiles);
                for (uint32_t col = 0; col < k_cols; ++col) {
                    combine_credit.up(noc, peer_x(col), peer_y(col), 1);
                }
            }
        };
        if constexpr (!alias_slab_combine) {
            give_credits();
        }
        uint32_t order[k_cols];
        uint32_t step[k_cols];
        uint32_t num_slices = 0;
        for (uint32_t j = 0; j < k_cols; ++j) {
            const uint32_t y = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
            if (y < num_owning_cols) {
                step[num_slices] = j;
                order[num_slices++] = y;
            }
        }
        const uint32_t tile_bytes = get_tile_size(cb_partials);
        bool credited = false;
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
            // The combine slots are the slab's L1: peers may write them only once this core's gather landed.
            {
                MaybeDeviceZoneScope("writer_slab_dead_wait");
                Semaphore<>(slab_dead_sem_id).wait_min(m_block_idx + 1);
            }
            give_credits();
        }
        for (uint32_t j = 0; j < num_slices; ++j) {
            issue_weights_through(j + 1 + nslice_weight_lookahead);
            const uint32_t y = order[j];
            if (conv2d_3x3::nslice_reader_sends<k_cols>(step[j])) {
                continue;
            }
            {
                MaybeDeviceZoneScope("writer_combine_wait_partials");
                cb_wait_front(cb_partials, slice_tiles);
            }
            if (!credited) {
                MaybeDeviceZoneScope("writer_combine_wait_credit");
                combine_credit.wait_min((m_block_idx + 1) * num_owning_cols);
                credited = true;
            }
            {
                MaybeDeviceZoneScope("writer_combine_send");
                conv2d_3x3::send_slice<block_m_tiles, out_n_tiles, tensor_n_tiles>(
                    get_read_ptr(cb_partials), combine_base, tile_bytes, my_col, y, row_peer_coords_arg);
                noc_async_write_barrier();
            }
            combine_arrival.up(noc, peer_x(y), peer_y(y), 1);
            cb_pop_front(cb_partials, slice_tiles);
        }
        if (out_valid_tiles > 0) {
            {
                MaybeDeviceZoneScope("writer_combine_wait_arrivals");
                combine_arrival.wait_min((m_block_idx + 1) * k_cols);
            }
            cb_push_back(cb_combine_slots, k_cols * slot_tiles);
        }
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
            while (k_blocks_issued < (m_block_idx + 1) * num_k_blocks) {
                issue_weight_k_block();
                if constexpr (gather_split) {
                    gather_odd_groups(k_blocks_issued - 1);
                }
            }
            if constexpr (k_split) {
                combine_round(m_block_idx);
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
    if constexpr (!output_sharded) {
        noc_async_write_barrier();
    }
    if constexpr (k_split) {
        noc_async_atomic_barrier();
    }
}
