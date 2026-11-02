// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope writer (BRISC): streams this core's W column slice into cb_weight_block, then
// scatters each valid output row into q|k|v head layout.
// R2: when the "wmc" channel is present, one core of the N-group reads each W K-block and multicasts it
// into the same cb_weight_block slot on the rest of the group; otherwise every core reads its own (R1).
// weight_depth > 2 (R2 only): the sender of round k+1 issues its reads into the next slot before round k's
// transfer, so its DRAM reads overlap round k's multicast. With two channels ("wmc1" present) the senders
// alternate rounds, so one reads while the other multicasts.
// Output rows: every out_write_nocs-th row from cb_out_tiles (the reader stores the others on NoC0).

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "qkv_proj_heads_rope_store.hpp"
#include "fv_schedule.hpp"

void kernel_main() {
    constexpr uint32_t cb_weight_block = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_m = get_compile_time_arg_val(2);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t k_block = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t weight_width_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_heads = get_compile_time_arg_val(9);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(10);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(11);
    // S % 32: valid rows of the last tile-row of each image (0 when S is tile-aligned).
    constexpr uint32_t last_valid_rows = get_compile_time_arg_val(12);
    constexpr uint32_t out_write_nocs = get_compile_time_arg_val(13);
    constexpr uint32_t weight_depth = get_compile_time_arg_val(14);
    constexpr auto weight_args = TensorAccessorArgs<15>();
    constexpr auto q_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t third = get_arg_val<uint32_t>(4);
    const uint32_t first_col_tile = get_arg_val<uint32_t>(5);
    const uint32_t first_head = get_arg_val<uint32_t>(6);
    const uint32_t row_start = get_arg_val<uint32_t>(7);
    const uint32_t core_rows = get_arg_val<uint32_t>(8);
    // Cores sharing a W slice (one N-group) start each K-block at different tiles so they don't read in lockstep.
    const uint32_t stagger_idx = get_arg_val<uint32_t>(9);
    const uint32_t stagger_count = get_arg_val<uint32_t>(10);
    constexpr auto w_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wmc_ct_offset"), get_named_compile_time_arg_val("wmc_rt_offset")>();
    // Second R2 channel (own fixed sender): when present, even rounds go on w_mcast and odd rounds on it.
    constexpr auto w_mcast_alt = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wmc1_ct_offset"), get_named_compile_time_arg_val("wmc1_rt_offset")>();
    constexpr bool alternate = std::decay_t<decltype(w_mcast_alt)>::active;
#if defined(FV_BMC)
    constexpr uint32_t round_offset = std::decay_t<decltype(w_mcast)>::active ? 1 : 0;
#else
    constexpr uint32_t round_offset = 0;
#endif

    // K' = hidden tiles: the bias is seeded into the partials by compute (the reader reads it).
    using Sched = fv_schedule::Schedule<k_block, hidden_tiles>;
    static_assert(Sched::num_blocks == num_k_blocks);
    constexpr uint32_t weight_block_tiles = k_block * n_tiles;

    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, tile_bytes);
    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);

    Noc noc;

    // Issue the reads of W K-block k_blk into a slot; no barrier.
    // `between` runs after every issued tile (fill_v2: polls for the bias to multicast).
    auto issue_weight_reads_with = [&](uint32_t k_blk, uint32_t block_addr, auto&& between) {
        const uint32_t k_size = Sched::size(k_blk);
        const uint32_t block_tiles = k_size * n_tiles;
        const uint32_t start = stagger_idx * block_tiles / stagger_count;
        uint32_t k = start / n_tiles;
        uint32_t n = start % n_tiles;
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            const uint32_t kt = Sched::first(k_blk) + k;
            const uint32_t l1_addr = block_addr + (k * n_tiles + n) * tile_bytes;
            noc_async_read_page(kt * weight_width_tiles + first_col_tile + n, weight_accessor, l1_addr);
            if (++n == n_tiles) {
                n = 0;
                if (++k == k_size) {
                    k = 0;
                }
            }
            between();
        }
    };
    auto issue_weight_reads = [&](uint32_t k_blk, uint32_t block_addr) {
        issue_weight_reads_with(k_blk, block_addr, [] {});
    };

    auto read_weight_block = [&](uint32_t k_blk, uint32_t block_addr) {
        MaybeDeviceZoneScope("writer_w_read");
        issue_weight_reads(k_blk, block_addr);
        noc_async_read_barrier();
    };

    const uint32_t num_rounds = (core_rows + block_m - 1) / block_m * num_k_blocks;
    // Rounds walk the ring one slot each, so round r lands in slot r % weight_depth on every core.
    const uint32_t ring_base = get_write_ptr(cb_weight_block);
    auto slot_addr = [&](uint32_t r) { return ring_base + (r % weight_depth) * weight_block_tiles * tile_bytes; };
    // This core issued the current round's reads during the previous round.
    bool issued_ahead = false;

    // Generic so an absent channel's branches are discarded, not instantiated.
    auto with_channel = [&](uint32_t r, auto&& fn) {
        if constexpr (alternate) {
            if (r & 1) {
                fn(w_mcast_alt);
                return;
            }
        }
        fn(w_mcast);
    };

    auto fill_weight_slot = [&](const auto& channel, uint32_t k_blk, uint32_t round, uint32_t block_addr) {
        if constexpr (std::decay_t<decltype(channel)>::active) {
            const bool sending = channel.should_send(round);
            if (sending) {
                MaybeDeviceZoneScope("writer_w_read");
                if (!issued_ahead) {
                    issue_weight_reads(k_blk, block_addr);
                }
                noc_async_read_barrier();
            }
            // Read ahead when the next round is this core's to send and its slot is already free (waiting
            // for it would hold back this round, which the whole N-group needs). The reads overlap this
            // round's handshake and multicast; on round 0 they go after it, as round 0 is every core's
            // pipeline fill.
            bool read_ahead = false;
            const uint32_t next = round + 1;
            if constexpr (weight_depth > 2) {
                if (next < num_rounds) {
                    with_channel(next, [&](const auto& next_channel) { read_ahead = next_channel.should_send(next); });
                    read_ahead = read_ahead && cb_pages_reservable_at_back(cb_weight_block, 2 * weight_block_tiles);
                }
                if (read_ahead && round > 0) {
                    issue_weight_reads(next % num_k_blocks, slot_addr(next));
                }
            }
            if (sending) {
                MaybeDeviceZoneScope("writer_w_mcast_send");
                channel.sender(noc).send(block_addr, block_addr, Sched::size(k_blk) * n_tiles * tile_bytes);
            } else {
                MaybeDeviceZoneScope("writer_w_receive");
                channel.receiver(noc).receive(round + round_offset);
            }
            if (read_ahead && round == 0) {
                issue_weight_reads(next % num_k_blocks, slot_addr(next));
            }
            issued_ahead = read_ahead;
        } else {
            read_weight_block(k_blk, block_addr);
        }
    };

#if defined(FV_BMC)
    // fill_v2: the bias row does not vary along token_rows either, so the N-group's W sender (whose reader
    // read it) multicasts it on the W channel as an extra leading round. Its round-0 W reads are issued
    // first, so they overlap the bias wait and send.
    // Generic so an absent channel's (R1) sender/receiver are discarded, not instantiated.
    [&](const auto& w_mcast) {
        if constexpr (std::decay_t<decltype(w_mcast)>::active) {
            constexpr uint32_t cb_bias = FV_CB_BIAS;
            constexpr uint32_t face_bytes = 16 * 16 * 2;
            // Up to tile n_tiles - 1's face-1 row 0: the only bytes compute reads.
            constexpr uint32_t bias_page = FV_BIAS_PAGE;
            constexpr uint32_t bias_bytes = (n_tiles - 1) * bias_page + face_bytes + 64;
            if (w_mcast.should_send(0)) {
                const uint32_t bias_l1 = get_read_ptr(cb_bias);
                bool bias_sent = false;
                auto send_bias = [&] {
                    MaybeDeviceZoneScope("writer_bias_mcast");
                    w_mcast.sender(noc).send(bias_l1, bias_l1, bias_bytes);
                    bias_sent = true;
                };
#if defined(FV_BWRITER)
                {
                    // The sender's own bias read on NoC1, ahead of its W reads: 9 readers DRAM-wide instead of 108.
                    MaybeDeviceZoneScope("writer_bias_read");
                    const auto bias_accessor = TensorAccessor(bias_args, get_arg_val<uint32_t>(11), tile_bytes);
                    cb_reserve_back(cb_bias, n_tiles);
#if defined(FV_BTRID)
                    // Bias reads on their own transaction id, so the bias barrier does not wait for the W reads.
                    noc_async_read_set_trid(1);
#endif
                    for (uint32_t n = 0; n < n_tiles; ++n) {
                        const uint64_t src = bias_accessor.get_noc_addr(first_col_tile + n);
                        noc_async_read(src, bias_l1 + n * bias_page, 64);
                        noc_async_read(src + face_bytes, bias_l1 + n * bias_page + face_bytes, 64);
                    }
#if defined(FV_BTRID)
                    noc_async_read_set_trid(0);
                    if (core_rows > 0) {
                        cb_reserve_back(cb_weight_block, weight_block_tiles);
                        issue_weight_reads(0, get_write_ptr(cb_weight_block));
                        issued_ahead = true;
                    }
                    noc_async_read_barrier_with_trid(1);
#else
                    noc_async_read_barrier();
#endif
                    cb_push_back(cb_bias, n_tiles);
                }
                send_bias();
#endif
                if (core_rows > 0 && !issued_ahead) {
                    cb_reserve_back(cb_weight_block, weight_block_tiles);
#if defined(FV_BPOLL)
                    // Send the bias as soon as the reader lands it, between W tile issues.
                    issue_weight_reads_with(0, get_write_ptr(cb_weight_block), [&] {
                        if (!bias_sent && cb_pages_available_at_front(cb_bias, n_tiles)) {
                            send_bias();
                        }
                    });
#else
                    issue_weight_reads(0, get_write_ptr(cb_weight_block));
#endif
                    issued_ahead = true;
                }
                if (!bias_sent) {
                    cb_wait_front(cb_bias, n_tiles);
                    send_bias();
                }
            } else {
                MaybeDeviceZoneScope("writer_bias_receive");
                cb_reserve_back(cb_bias, n_tiles);
                w_mcast.receiver(noc).receive(0);
                cb_push_back(cb_bias, n_tiles);
            }
        }
    }(w_mcast);
#endif

    // Every core of an N-group walks the same number of blocks (host gate), so the rounds and the
    // cb_weight_block slot sequence line up across the group.
    uint32_t round = 0;

    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;

        for (uint32_t k_blk = 0; k_blk < num_k_blocks; ++k_blk, ++round) {
            {
                MaybeDeviceZoneScope("writer_w_reserve");
                cb_reserve_back(cb_weight_block, weight_block_tiles);
            }
            const uint32_t block_addr = get_write_ptr(cb_weight_block);
            with_channel(round, [&](const auto& channel) { fill_weight_slot(channel, k_blk, round, block_addr); });
            cb_push_back(cb_weight_block, weight_block_tiles);
        }

        {
            // Waiting for the block's first row = waiting for the K-loop + the first epilogue row.
            MaybeDeviceZoneScope("writer_store_wait_first");
            cb_wait_front(cb_out_tiles, n_tiles);
        }
        // Rows alternate between this kernel's output CB and the reader's (out_write_nocs == 2).
        MaybeDeviceZoneScope("writer_store");
        for (uint32_t row = 0; row < rows; row += out_write_nocs) {
            qkv_store::
                store_row<cb_out_tiles, n_tiles, tiles_per_head, num_heads, seq_tiles, tile_bytes, last_valid_rows>(
                    noc, row_start + block_row + row, third, first_head, q_accessor, k_accessor, v_accessor);
        }
    }
    MaybeDeviceZoneScope("writer_write_barrier");
    noc_async_write_barrier();
}
