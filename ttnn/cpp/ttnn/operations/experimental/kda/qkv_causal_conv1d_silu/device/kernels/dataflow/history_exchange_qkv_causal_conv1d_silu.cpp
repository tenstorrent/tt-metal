// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sequence-parallel history exchange of qkv_causal_conv1d_silu, on one core outside the convolution grid.
//
// The exchange buffer and the three global semaphores persist across calls. Ranks first meet at an entry barrier,
// so no rank writes a peer's buffer before that peer has finished the previous call. Each rank then sends the three
// rows preceding the next physical rank's segment, which its convolution cores staged, to that rank, and releases its
// convolution cores once its own predecessor's rows have landed. The owner of the interval's final segment assembles
// the replacement carry and multicasts it to every other rank; every rank writes it to final_history.
//
// A 2D multicast leaves every hop in one ETH direction, so on a 2D fabric whose SP axis turns the barrier and the
// carry go to each other rank as a 2D unicast instead (unicast_peers).
//
// A history slab is SP-sharded: each rank holds only its slice of every Q/K/V branch of an entry. Only
// the chronologically first rank reads the old tail, so after the entry barrier every other rank sends its slices of
// the slot's old tail into that rank's exchange history rows and bumps its history semaphore; the first rank copies
// its own slices there, waits for the others' and releases its convolution cores (history_ready), which read the old
// tail from those rows. Each rank then writes back only its own slices of the replacement carry.

#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/qkv_conv_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"

using namespace tt::tt_fabric::linear::experimental;
using namespace qkv_conv;

// Per connection: direction tag, ETH channel, teardown semaphore id, buffer-index semaphore id.
constexpr size_t connection_args = 4;
constexpr size_t teardown_arg = 2;
constexpr size_t buffer_index_arg = 3;

// The routes array holds the fabric connections, then each convolution core's NoC x and y, then (unicast_peers)
// each other rank's connection, chip and mesh; the kernel reads it by runtime-arg index. sp_rank, the successor and
// the routes are per device, so one program serves a whole range.
void kernel_main() {
    constexpr uint32_t channel_tiles = get_arg(args::channel_tiles);
    constexpr uint32_t max_payload = get_arg(args::max_payload);
    constexpr uint32_t num_connections = get_arg(args::num_connections);
    constexpr uint32_t forward_connection = get_arg(args::forward_connection);
    constexpr uint32_t forward_range = get_arg(args::forward_range);
    constexpr uint32_t backward_connection = get_arg(args::backward_connection);
    constexpr uint32_t backward_range = get_arg(args::backward_range);
    constexpr uint32_t successor_connection = get_arg(args::successor_connection);
    constexpr uint32_t successor_hops = get_arg(args::successor_hops);
    constexpr uint32_t num_convolution_cores = get_arg(args::num_convolution_cores);
    constexpr uint32_t sp_size = get_arg(args::sp_size);
    constexpr uint32_t local_rows = get_arg(args::local_rows);
    constexpr uint32_t has_actual_end = get_arg(args::has_actual_end);
    constexpr uint32_t token_paged = get_arg(args::token_paged);
    constexpr uint32_t page_channels = get_arg(args::page_channels);
    constexpr uint32_t unicast_peers = get_arg(args::unicast_peers);
    constexpr uint32_t slab = get_arg(args::slab);
    constexpr uint32_t slab_index_limit = get_arg(args::slab_index_limit);
    constexpr uint32_t slab_scale = get_arg(args::slab_scale);
    constexpr uint32_t slab_offset = get_arg(args::slab_offset);
    constexpr uint32_t entry_pages = get_arg(args::entry_pages);
    constexpr uint32_t peer_routes = get_arg(args::peer_routes);
    static_assert(!slab || (sp_size > 1 && !token_paged), "a history slab is channel-paged and SP-sharded");
    // Native tail pages per Q/K/V branch; a slab rank stores branch_pages / sp_size of each.
    constexpr uint32_t branch_pages[3] = {get_arg(args::q_pages), get_arg(args::k_pages), get_arg(args::v_pages)};
    const uint32_t entry_semaphore = get_arg(args::entry_semaphore);
    const uint32_t predecessor_semaphore = get_arg(args::predecessor_semaphore);
    const uint32_t carry_semaphore = get_arg(args::carry_semaphore);
    const uint32_t history_semaphore = get_arg(args::history_semaphore);
    const uint32_t sp_rank = get_arg(args::sp_rank);
    const uint32_t successor_chip = get_arg(args::successor_chip);
    const uint32_t successor_mesh = get_arg(args::successor_mesh);
    constexpr size_t routes_arg = args::routes.byte_offset / sizeof(uint32_t);
    constexpr uint32_t row_bytes = channel_tiles * tile_row_bytes;
    static_assert(max_payload % read_alignment == 0, "row chunks must stay aligned in DRAM");
    // `history` and `final_history` are token-paged or channel-paged (qkv_conv_dataflow.hpp); the exchange buffer and
    // the L1 scratch keep full rows.
    static_assert(
        token_paged || channel_tiles % (page_channels / tile_width) == 0,
        "the convolution tail is stored in whole pages");
    Noc noc;
    Scratchpad<uint32_t> scratch(scratch::exchange_scratch);
    const auto history = TensorAccessor(tensor::history);
    const auto exchange_buffer = TensorAccessor(tensor::exchange_buffer);
    const auto final_history = TensorAccessor(tensor::final_history);
    Semaphore staged(sem::staged);
    Semaphore predecessor_ready(sem::predecessor_ready);
    Semaphore history_ready(sem::history_ready);

    // Scratch, from an aligned base: a slot for the chronology scalars, then three outgoing rows and three carry rows.
    // Raw: base/offset arithmetic below feeds raw reads/writes into scratch; no handle form yet.
    const uint32_t scratch_base = local_mem::detail::AddressAccess::address(scratch.local_mem());
    const uint32_t scalar_slot = align_up(scratch_base, read_alignment);
    const uint32_t outgoing = scalar_slot + read_alignment;
    const uint32_t carry = outgoing + history_rows * row_bytes;
    const auto offset_of = [&](uint32_t address) { return address - scratch_base; };
    const auto interval = read_chronology<has_actual_end, sp_size, local_rows>(noc, scalar_slot, sp_rank);
    const auto layer_carry =
        read_slab_carry<slab, slab_index_limit, slab_scale, slab_offset, entry_pages>(noc, scalar_slot);
    const auto read_rows = [&](uint32_t first_page, uint32_t destination) {
        for (uint32_t row = 0; row < history_rows; ++row) {
            noc.async_read(
                exchange_buffer,
                scratch,
                row_bytes,
                {.page_id = first_page + row},
                {.offset_bytes = offset_of(destination) + row * row_bytes});
        }
    };

    // The slab: this rank's slices, as (branch, row) runs of whole pages' rows. Run (b, r) is the rank's
    // channels of branch b in tail row r: native pages [native_page(b), + own_pages(b)) of the full row, stored pages
    // [stored_page(b), + own_pages(b)) of the slab entry, each page holding history_rows rows of page_channels.
    constexpr uint32_t page_row_bytes = page_channels * element_bytes;
    const auto own_pages = [&](uint32_t b) { return branch_pages[b] / sp_size; };
    const auto stored_page = [&](uint32_t b) {
        uint32_t page = 0;
        for (uint32_t i = 0; i < b; ++i) {
            page += own_pages(i);
        }
        return page;
    };
    const auto native_page = [&](uint32_t b) {
        uint32_t page = 0;
        for (uint32_t i = 0; i < b; ++i) {
            page += branch_pages[i];
        }
        return page + sp_rank * own_pages(b);
    };
    // Run (b, r) at byte offset run_offset(b, r) of a staging area: branch after branch, row after row.
    const auto run_offset = [&](uint32_t b, uint32_t row) {
        return (stored_page(b) * history_rows + row * own_pages(b)) * page_row_bytes;
    };
    // The slab entry's slices into `staging` (scratch), as runs.
    const auto stage_own_slices = [&](uint32_t staging) {
        for (uint32_t b = 0; b < 3; ++b) {
            for (uint32_t i = 0; i < own_pages(b); ++i) {
                for (uint32_t row = 0; row < history_rows; ++row) {
                    noc.async_read(
                        history,
                        scratch,
                        page_row_bytes,
                        {.page_id = layer_carry.page_base + stored_page(b) + i, .offset_bytes = row * page_row_bytes},
                        {.offset_bytes = offset_of(staging) + run_offset(b, row) + i * page_row_bytes});
                }
            }
        }
        noc.async_read_barrier();
    };

    // The replacement carry, on the owner of the interval's final segment: its last valid rows, preceded by the
    // caller's history or the predecessor's outgoing rows when that segment holds fewer than three.
    const auto assemble_carry = [&]() {
        const uint32_t carried = interval.carried_rows();
        for (uint32_t row = 0; row < history_rows; ++row) {
            const uint32_t position = carried + row;
            const bool from_layer_history = position < history_rows && interval.carry_continues_layer_history();
            const uint32_t destination = offset_of(carry) + row * row_bytes;
            if (from_layer_history && layer_carry.zeros) {
                noc.async_write_zeros(scratch, row_bytes, {.offset_bytes = destination});
                noc.write_zeros_l1_barrier();
            } else if (from_layer_history && slab) {
                // Gathered into this (the first) rank's history rows.
                noc.async_read(
                    exchange_buffer,
                    scratch,
                    row_bytes,
                    {.page_id = history_page + position},
                    {.offset_bytes = destination});
            } else if (from_layer_history) {
                read_tail_row<token_paged, page_channels>(
                    noc, history, scratch, position, 0, channel_tiles, destination, layer_carry.page_base);
            } else {
                const uint32_t page =
                    position >= history_rows ? end_page + position - history_rows : predecessor_page + position;
                noc.async_read(exchange_buffer, scratch, row_bytes, {.page_id = page}, {.offset_bytes = destination});
            }
        }
        noc.async_read_barrier();
    };

    if constexpr (sp_size == 1) {
        // A single rank owns the whole interval and has no predecessor.
        staged.wait_min(num_convolution_cores);
        assemble_carry();
    } else {
        // The host computes the connection arguments without this kernel's semaphore ids; bind them before opening.
        const uint32_t teardown[] = {sem::teardown0.id, sem::teardown1.id, sem::teardown2.id, sem::teardown3.id};
        const uint32_t buffer_index[] = {
            sem::buffer_index0.id, sem::buffer_index1.id, sem::buffer_index2.id, sem::buffer_index3.id};
        auto* connection_arg_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(routes_arg));
        for (uint32_t c = 0; c < num_connections; ++c) {
            connection_arg_words[c * connection_args + teardown_arg] = teardown[c];
            connection_arg_words[c * connection_args + buffer_index_arg] = buffer_index[c];
        }
        size_t arg_index = routes_arg;
        tt::tt_fabric::RoutingPlaneConnectionManager connections;
        open_connections(connections, num_connections, arg_index);
        const size_t convolution_core_args = arg_index;
        // With unicast_peers: per other rank, its connection, fabric chip and mesh.
        const size_t peer_args = convolution_core_args + 2 * num_convolution_cores;
        constexpr size_t args_per_peer = 3;
        auto* header = PacketHeaderPool::allocate_header();

        const auto semaphore_noc_address = [](uint32_t address) {
            return safe_get_noc_addr(my_x[0], my_y[0], address, 0);
        };
        // Connections past the two SP neighbors only carry unicasts (unicast_peers).
        uint8_t ranges[max_fabric_connections] = {};
        if constexpr (forward_connection != no_fabric_connection) {
            ranges[forward_connection] = forward_range;
        }
        if constexpr (backward_connection != no_fabric_connection) {
            ranges[backward_connection] = backward_range;
        }
        // Point the header at another rank: unicast_peers only.
        const auto route_to_peer = [&](uint32_t peer) -> auto& {
            const size_t args = peer_args + peer * args_per_peer;
#if defined(FABRIC_2D)
            fabric_set_unicast_route(header, get_arg_val<uint32_t>(args + 1), get_arg_val<uint32_t>(args + 2));
#endif
            return connections.get(get_arg_val<uint32_t>(args)).sender;
        };
        // Bump `semaphore` on every other rank.
        const auto multicast_increment = [&](uint32_t semaphore) {
            if constexpr (unicast_peers) {
                for (uint32_t peer = 0; peer + 1 < sp_size; ++peer) {
                    auto& sender = route_to_peer(peer);
                    fabric_unicast_noc_unicast_atomic_inc(
                        &sender,
                        header,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_address(semaphore), 1, true},
                        1);
                    noc_async_writes_flushed();
                }
                return;
            }
            for (uint32_t c = 0; c < num_connections; ++c) {
                if (ranges[c] == 0) {
                    continue;
                }
                fabric_set_mcast_route(connections, header, ranges, c);
                fabric_multicast_noc_unicast_atomic_inc(
                    &connections.get(c).sender,
                    header,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_address(semaphore), 1, true},
                    1,
                    ranges[c]);
                noc_async_writes_flushed();
            }
        };
        // Write three rows from `source` to exchange pages [first_page, +3) on every other rank, then bump `semaphore`.
        const auto multicast_rows = [&](uint32_t source, uint32_t first_page, uint32_t semaphore) {
            if constexpr (unicast_peers) {
                for (uint32_t peer = 0; peer + 1 < sp_size; ++peer) {
                    for (uint32_t row = 0; row < history_rows; ++row) {
                        for (uint32_t offset = 0; offset < row_bytes; offset += max_payload) {
                            auto& sender = route_to_peer(peer);
                            fabric_unicast_noc_unicast_write(
                                &sender,
                                header,
                                source + row * row_bytes + offset,
                                row_bytes - offset < max_payload ? row_bytes - offset : max_payload,
                                tt::tt_fabric::NocUnicastCommandHeader{
                                    addrgen_detail::get_noc_address(exchange_buffer, first_page + row, offset)},
                                1);
                            noc_async_writes_flushed();
                        }
                    }
                }
                multicast_increment(semaphore);
                return;
            }
            for (uint32_t c = 0; c < num_connections; ++c) {
                if (ranges[c] == 0) {
                    continue;
                }
                for (uint32_t row = 0; row < history_rows; ++row) {
                    for (uint32_t offset = 0; offset < row_bytes; offset += max_payload) {
                        fabric_set_mcast_route(connections, header, ranges, c);
                        fabric_multicast_noc_unicast_write(
                            &connections.get(c).sender,
                            header,
                            source + row * row_bytes + offset,
                            row_bytes - offset < max_payload ? row_bytes - offset : max_payload,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                addrgen_detail::get_noc_address(exchange_buffer, first_page + row, offset)},
                            1,
                            ranges[c]);
                        noc_async_writes_flushed();
                    }
                }
            }
            multicast_increment(semaphore);
        };
        const auto route_to_successor = [&]() {
#if defined(FABRIC_2D)
            fabric_set_unicast_route(header, successor_chip, successor_mesh);
#endif
        };
        // Write three rows from `source` to exchange pages [first_page, +3) on the successor, then bump `semaphore`.
        const auto unicast_rows = [&](uint32_t source, uint32_t first_page, uint32_t semaphore) {
            auto& sender = connections.get(successor_connection).sender;
            for (uint32_t row = 0; row < history_rows; ++row) {
                for (uint32_t offset = 0; offset < row_bytes; offset += max_payload) {
                    route_to_successor();
                    fabric_unicast_noc_unicast_write(
                        &sender,
                        header,
                        source + row * row_bytes + offset,
                        row_bytes - offset < max_payload ? row_bytes - offset : max_payload,
                        tt::tt_fabric::NocUnicastCommandHeader{
                            addrgen_detail::get_noc_address(exchange_buffer, first_page + row, offset)},
                        successor_hops);
                    noc_async_writes_flushed();
                }
            }
            route_to_successor();
            fabric_unicast_noc_unicast_atomic_inc(
                &sender,
                header,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_address(semaphore), 1, true},
                successor_hops);
            noc_async_writes_flushed();
        };
        const auto wait_for_arrival = [](uint32_t semaphore) { kda_counter::wait_and_consume(semaphore, 1); };
        // Sharded slab: the staged slices (in the carry rows) into `first_rank`'s exchange history rows, then its
        // history semaphore. On a 2D fabric through the peer's route; on 1D along the SP axis, forward when the
        // forward range reaches it, else backward.
        const auto send_slices_to_first_rank = [&](uint32_t first_rank) {
            const uint32_t step = (first_rank + sp_size - sp_rank) % sp_size;
            decltype(&connections.get(0).sender) sender;
            uint32_t hops = 1;
            if constexpr (peer_routes) {
                sender = &route_to_peer(step - 1);
            } else {
                const bool forward = step <= forward_range;
                sender = &connections.get(forward ? forward_connection : backward_connection).sender;
                hops = forward ? step : sp_size - step;
            }
            for (uint32_t b = 0; b < 3; ++b) {
                const uint32_t run_bytes = own_pages(b) * page_row_bytes;
                for (uint32_t row = 0; row < history_rows; ++row) {
                    for (uint32_t offset = 0; offset < run_bytes; offset += max_payload) {
                        if constexpr (peer_routes) {
                            route_to_peer(step - 1);
                        }
                        fabric_unicast_noc_unicast_write(
                            sender,
                            header,
                            carry + run_offset(b, row) + offset,
                            run_bytes - offset < max_payload ? run_bytes - offset : max_payload,
                            tt::tt_fabric::NocUnicastCommandHeader{addrgen_detail::get_noc_address(
                                exchange_buffer, history_page + row, native_page(b) * page_row_bytes + offset)},
                            hops);
                        noc_async_writes_flushed();
                    }
                }
            }
            if constexpr (peer_routes) {
                route_to_peer(step - 1);
            }
            fabric_unicast_noc_unicast_atomic_inc(
                sender,
                header,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_address(history_semaphore), 1, true},
                hops);
            noc_async_writes_flushed();
        };

        // Entry barrier: every other rank's arrival.
        multicast_increment(entry_semaphore);
        staged.wait_min(num_convolution_cores);
        read_rows(outgoing_page, outgoing);
        kda_counter::wait_and_consume(entry_semaphore, sp_size - 1);
        noc.async_read_barrier();

        unicast_rows(outgoing, predecessor_page, predecessor_semaphore);
        if constexpr (slab) {
            if (!layer_carry.zeros) {
                // The carry rows are free until the carry: stage this rank's slices there.
                stage_own_slices(carry);
                if (interval.first_rank == sp_rank) {
                    for (uint32_t b = 0; b < 3; ++b) {
                        for (uint32_t row = 0; row < history_rows; ++row) {
                            noc.async_write(
                                scratch,
                                exchange_buffer,
                                own_pages(b) * page_row_bytes,
                                {.offset_bytes = offset_of(carry) + run_offset(b, row)},
                                {.page_id = history_page + row, .offset_bytes = native_page(b) * page_row_bytes});
                        }
                    }
                    noc.async_write_barrier();
                    kda_counter::wait_and_consume(history_semaphore, sp_size - 1);
                    for (uint32_t core = 0; core < num_convolution_cores; ++core) {
                        history_ready.up(
                            noc,
                            get_arg_val<uint32_t>(convolution_core_args + 2 * core),
                            get_arg_val<uint32_t>(convolution_core_args + 2 * core + 1),
                            1);
                    }
                } else {
                    send_slices_to_first_rank(interval.first_rank);
                }
            }
        }
        wait_for_arrival(predecessor_semaphore);
        for (uint32_t core = 0; core < num_convolution_cores; ++core) {
            const uint32_t x = get_arg_val<uint32_t>(convolution_core_args + 2 * core);
            const uint32_t y = get_arg_val<uint32_t>(convolution_core_args + 2 * core + 1);
            predecessor_ready.up(noc, x, y, 1);
        }
        if (interval.final_owner == sp_rank) {
            assemble_carry();
            multicast_rows(carry, carry_page, carry_semaphore);
        } else {
            wait_for_arrival(carry_semaphore);
            read_rows(carry_page, carry);
            noc.async_read_barrier();
        }
        noc_async_atomic_barrier();
        close_connections(connections);
    }
    // The replacement carry, three full rows in scratch, goes out in history's layout; the outgoing rows have been
    // sent, so their scratch gathers its pages.
    if (layer_carry.writes) {
        if constexpr (slab) {
            // This rank's slices only: native page native_page(b) + i is stored page stored_page(b) + i.
            write_tail<token_paged, page_channels>(
                noc,
                scratch,
                final_history,
                channel_tiles,
                offset_of(carry),
                offset_of(outgoing),
                layer_carry.page_base,
                [&](uint32_t page) -> uint32_t {
                    for (uint32_t b = 0; b < 3; ++b) {
                        if (page >= native_page(b) && page < native_page(b) + own_pages(b)) {
                            return stored_page(b) + page - native_page(b);
                        }
                    }
                    return no_page;
                });
        } else {
            write_tail<token_paged, page_channels>(
                noc,
                scratch,
                final_history,
                channel_tiles,
                offset_of(carry),
                offset_of(outgoing),
                layer_carry.page_base);
        }
    }
    noc.async_write_barrier();
}
