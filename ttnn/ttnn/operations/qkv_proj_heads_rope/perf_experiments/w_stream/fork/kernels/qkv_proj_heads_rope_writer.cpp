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

#if defined(WS_SPLIT) || defined(WS_FWD)
namespace ws {
namespace wire = dataflow_kernel_lib::mcast_wire;
template <uint32_t CT_BASE, wire::SemaphoreRole ROLE>
constexpr uint32_t sem_id() {
    if constexpr (get_compile_time_arg_val(CT_BASE) == wire::ABSENT) {
        return 0;
    } else {
        return dataflow_kernel_lib::detail::positional_mcast_semaphore<CT_BASE, ROLE>();
    }
}
template <uint32_t CT_BASE>
struct Sems {
    static constexpr uint32_t consumer_ready = sem_id<CT_BASE, wire::CONSUMER_READY>();
    static constexpr uint32_t data_ready = sem_id<CT_BASE, wire::DATA_READY>();
};
}  // namespace ws
#endif

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
#ifdef WS_SPLIT
    // w_stream WS_SPLIT: every round is split into one contiguous tile range per channel; each channel's
    // fixed sender reads and multicasts only its range, so the N-group's DRAM reads and sends run in parallel.
    constexpr bool alternate = false;
    constexpr auto w_mcast_2 = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wmc2_ct_offset"), get_named_compile_time_arg_val("wmc2_rt_offset")>();
    constexpr auto w_mcast_3 = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wmc3_ct_offset"), get_named_compile_time_arg_val("wmc3_rt_offset")>();
    constexpr uint32_t num_parts = 1 + std::decay_t<decltype(w_mcast_alt)>::active +
                                   std::decay_t<decltype(w_mcast_2)>::active +
                                   std::decay_t<decltype(w_mcast_3)>::active;
#else
    constexpr bool alternate = std::decay_t<decltype(w_mcast_alt)>::active;
#endif

    // K' = hidden tiles: the bias is seeded into the partials by compute (the reader reads it).
    static_assert(num_k_blocks * k_block == hidden_tiles);
    constexpr uint32_t weight_block_tiles = k_block * n_tiles;

    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, tile_bytes);
    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);

    Noc noc;

    // Issue the reads of W K-block k_blk into a slot; no barrier.
    auto issue_weight_reads = [&](uint32_t k_blk, uint32_t block_addr) {
#ifdef WS_RRD
        // The reader owns tiles [0, WS_RRD_TILES) on the sender (see the reader); only R2 senders get here.
        if constexpr (std::decay_t<decltype(w_mcast)>::active) {
            for (uint32_t tile = WS_RRD_TILES; tile < weight_block_tiles; ++tile) {
                const uint32_t kt = k_blk * k_block + tile / n_tiles;
                noc_async_read_page(
                    kt * weight_width_tiles + first_col_tile + tile % n_tiles,
                    weight_accessor,
                    block_addr + tile * tile_bytes);
            }
            return;
        }
#endif
        const uint32_t start = stagger_idx * weight_block_tiles / stagger_count;
        uint32_t k = start / n_tiles;
        uint32_t n = start % n_tiles;
        for (uint32_t tile = 0; tile < weight_block_tiles; ++tile) {
            const uint32_t kt = k_blk * k_block + k;
            const uint32_t l1_addr = block_addr + (k * n_tiles + n) * tile_bytes;
            noc_async_read_page(kt * weight_width_tiles + first_col_tile + n, weight_accessor, l1_addr);
            if (++n == n_tiles) {
                n = 0;
                if (++k == k_block) {
                    k = 0;
                }
            }
        }
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
#ifndef WS_RA_OVERLAP
                if (read_ahead && round > 0) {
                    issue_weight_reads(next % num_k_blocks, slot_addr(next));
                }
#endif
            }
#ifdef WS_RA_OVERLAP
            // w_stream WS_RA_OVERLAP: issue the next round's reads while this round's multicast is on the wire.
            // The source slot is next written 3 rounds later, after every receiver acked a later round.
            if (sending && read_ahead && round > 0) {
                MaybeDeviceZoneScope("writer_w_mcast_send");
                channel.sender(noc).template send<dataflow_kernel_lib::SourceL1Guard::CallerManaged>(
                    block_addr, block_addr, weight_block_tiles * tile_bytes);
                issue_weight_reads(next % num_k_blocks, slot_addr(next));
                issued_ahead = true;
                return;
            }
#endif
            if (sending) {
#ifdef WS_RRD
                {
                    MaybeDeviceZoneScope("writer_w_reader_wait");
                    volatile tt_l1_ptr uint32_t* w_done =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(WS_SEM_DONE));
                    while (*w_done < round + 1) {
                    }
                }
#endif
                MaybeDeviceZoneScope("writer_w_mcast_send");
                channel.sender(noc).send(block_addr, block_addr, weight_block_tiles * tile_bytes);
            } else {
                MaybeDeviceZoneScope("writer_w_receive");
                channel.receiver(noc).receive(round);
            }
            if (read_ahead && round == 0) {
                issue_weight_reads(next % num_k_blocks, slot_addr(next));
            }
            issued_ahead = read_ahead;
        } else {
            read_weight_block(k_blk, block_addr);
        }
    };

#ifdef WS_SPLIT
    // Tiles [part_lo(c), part_lo(c + 1)) of a round belong to channel c.
    auto part_lo = [&](uint32_t c) { return c * weight_block_tiles / num_parts; };
    auto issue_part_reads = [&](uint32_t c, uint32_t k_blk, uint32_t block_addr) {
        for (uint32_t tile = part_lo(c); tile < part_lo(c + 1); ++tile) {
            const uint32_t kt = k_blk * k_block + tile / n_tiles;
            noc_async_read_page(
                kt * weight_width_tiles + first_col_tile + tile % n_tiles,
                weight_accessor,
                block_addr + tile * tile_bytes);
        }
    };
    // Apply fn(channel, index) to every present channel.
    auto for_each_channel = [&](auto&& fn) {
        fn(w_mcast, 0u, ws::Sems<get_named_compile_time_arg_val("wmc_ct_offset")>{});
        if constexpr (num_parts > 1) {
            fn(w_mcast_alt, 1u, ws::Sems<get_named_compile_time_arg_val("wmc1_ct_offset")>{});
        }
        if constexpr (num_parts > 2) {
            fn(w_mcast_2, 2u, ws::Sems<get_named_compile_time_arg_val("wmc2_ct_offset")>{});
        }
        if constexpr (num_parts > 3) {
            fn(w_mcast_3, 3u, ws::Sems<get_named_compile_time_arg_val("wmc3_ct_offset")>{});
        }
    };
    // The channel this core sends (at most one), or num_parts.
    uint32_t my_part = num_parts;
    for_each_channel([&](const auto& channel, uint32_t c, auto) {
        if constexpr (std::decay_t<decltype(channel)>::active) {
            if (channel.should_send(0)) {
                my_part = c;
            }
        }
    });
    // Mirrors the helper's receive(): ack = consumer_ready +1 on the sender, then wait data_ready VALID and clear.
    // Split here so a core acks every channel before it waits on any of them.
    auto ack = [&](const auto& channel, auto sems) {
        using Ch = std::decay_t<decltype(channel)>;
        if constexpr (Ch::active) {
            Semaphore<ProgrammableCoreType::TENSIX> consumer_ready(decltype(sems)::consumer_ready);
            consumer_ready.up(noc, channel.sender_x(), channel.sender_y(), 1);
        }
    };
    auto wait_data = [&](const auto& channel, auto sems) {
        using Ch = std::decay_t<decltype(channel)>;
        if constexpr (Ch::active) {
            Semaphore<ProgrammableCoreType::TENSIX> data_ready(decltype(sems)::data_ready);
            data_ready.wait(VALID);
            data_ready.set(INVALID);
        }
    };
    auto fill_split = [&](uint32_t k_blk, uint32_t round, uint32_t block_addr) {
        {
            MaybeDeviceZoneScope("writer_w_ack");
            for_each_channel([&](const auto& channel, uint32_t c, auto sems) {
                if (c != my_part) {
                    ack(channel, sems);
                }
            });
        }
        if (my_part < num_parts) {
            {
                MaybeDeviceZoneScope("writer_w_read");
                if (!issued_ahead) {
                    issue_part_reads(my_part, k_blk, block_addr);
                }
                noc_async_read_barrier();
            }
            const uint32_t next = round + 1;
            bool read_ahead = false;
            if constexpr (weight_depth > 2) {
                read_ahead = next < num_rounds && cb_pages_reservable_at_back(cb_weight_block, 2 * weight_block_tiles);
                if (read_ahead && round > 0) {
                    issue_part_reads(my_part, next % num_k_blocks, slot_addr(next));
                }
            }
            {
                MaybeDeviceZoneScope("writer_w_mcast_send");
                const uint32_t lo = block_addr + part_lo(my_part) * tile_bytes;
                const uint32_t bytes = (part_lo(my_part + 1) - part_lo(my_part)) * tile_bytes;
                for_each_channel([&](const auto& channel, uint32_t c, auto) {
                    if constexpr (std::decay_t<decltype(channel)>::active) {
                        if (c == my_part) {
                            channel.sender(noc).send(lo, lo, bytes);
                        }
                    }
                });
            }
            if (read_ahead && round == 0) {
                issue_part_reads(my_part, next % num_k_blocks, slot_addr(next));
            }
            issued_ahead = read_ahead;
        }
        MaybeDeviceZoneScope("writer_w_receive");
        for_each_channel([&](const auto& channel, uint32_t c, auto sems) {
            if (c != my_part) {
                wait_data(channel, sems);
            }
        });
    };
#endif

#ifdef WS_FWD
    // w_stream WS_FWD: role 1 = forwarder (a receiver of the main rectangle that relays every round to the one
    // core outside it), role 2 = that core. Sem 14 on the forwarder counts the target's free slots, sem 13 on
    // the target counts landed rounds.
    const uint32_t fwd_role = get_arg_val<uint32_t>(11);
    const uint32_t peer_x = get_arg_val<uint32_t>(12);
    const uint32_t peer_y = get_arg_val<uint32_t>(13);
    volatile tt_l1_ptr uint32_t* fwd_free = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(14));
    volatile tt_l1_ptr uint32_t* fwd_landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13));
    bool acked_ahead = false;
    using MainSems = ws::Sems<get_named_compile_time_arg_val("wmc_ct_offset")>;
    auto main_ack_on = [&](const auto& channel) {
        if constexpr (std::decay_t<decltype(channel)>::active) {
            Semaphore<ProgrammableCoreType::TENSIX> consumer_ready(MainSems::consumer_ready);
            consumer_ready.up(noc, channel.sender_x(), channel.sender_y(), 1);
        }
    };
    auto main_ack = [&]() { main_ack_on(w_mcast); };
    auto main_wait = [&]() {
        if constexpr (std::decay_t<decltype(w_mcast)>::active) {
            Semaphore<ProgrammableCoreType::TENSIX> data_ready(MainSems::data_ready);
            data_ready.wait(VALID);
            data_ready.set(INVALID);
        }
    };
    auto fwd_fill = [&](uint32_t k_blk, uint32_t round, uint32_t block_addr) -> bool {
        if (fwd_role == 2) {
            MaybeDeviceZoneScope("writer_w_receive");
            noc_semaphore_inc(get_noc_addr(peer_x, peer_y, get_semaphore(14)), 1);
            while (*fwd_landed < round + 1) {
            }
            return true;
        }
        if (fwd_role == 1) {
            MaybeDeviceZoneScope("writer_w_receive");
            if (!acked_ahead) {
                main_ack();
            }
            main_wait();
            acked_ahead = false;
            return true;
        }
        return false;
    };
    auto fwd_after_push = [&](uint32_t round, uint32_t block_addr) {
        if (fwd_role != 1) {
            return;
        }
        // Ack the next round first so the main sender is not held by this relay.
        if (round + 1 < num_rounds && cb_pages_reservable_at_back(cb_weight_block, weight_block_tiles)) {
            main_ack();
            acked_ahead = true;
        }
        MaybeDeviceZoneScope("writer_w_forward");
        while (*fwd_free < round + 1) {
        }
        noc_async_write(block_addr, get_noc_addr(peer_x, peer_y, block_addr), weight_block_tiles * tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(peer_x, peer_y, get_semaphore(13)), 1);
    };
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
#ifdef WS_SPLIT
            if constexpr (std::decay_t<decltype(w_mcast)>::active) {
                fill_split(k_blk, round, block_addr);
            } else {
                read_weight_block(k_blk, block_addr);
            }
#else
#ifdef WS_FWD
            if (!fwd_fill(k_blk, round, block_addr))
#endif
                with_channel(round, [&](const auto& channel) { fill_weight_slot(channel, k_blk, round, block_addr); });
#endif
            cb_push_back(cb_weight_block, weight_block_tiles);
#ifdef WS_FWD
            fwd_after_push(round, block_addr);
#endif
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
