// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sequence-parallel history exchange of qkv_causal_conv1d_silu, on one core outside the convolution grid.
//
// The exchange buffer and the three global semaphores persist across calls. Ranks first meet at an entry barrier,
// so no rank writes a peer's buffer before that peer has finished the previous call. Each rank then sends the three
// rows preceding the next physical rank's segment, which its convolution cores staged, to that rank, and releases its
// convolution cores once its own predecessor's rows have landed. The owner of the interval's final segment assembles
// the replacement carry and multicasts it to every other rank; every rank writes it to final_history.

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

using namespace tt::tt_fabric::linear::experimental;
using namespace qkv_conv;

// Per connection: direction tag, ETH channel, teardown semaphore id, buffer-index semaphore id.
constexpr size_t connection_args = 4;
constexpr size_t teardown_arg = 2;
constexpr size_t buffer_index_arg = 3;

template <
    uint32_t channel_tiles,
    uint32_t max_payload,
    uint32_t named_runtime_args,
    uint32_t num_connections,
    uint32_t forward_connection,
    uint32_t forward_range,
    uint32_t backward_connection,
    uint32_t backward_range,
    uint32_t successor_connection,
    uint32_t successor_hops,
    uint32_t successor_chip,
    uint32_t successor_mesh,
    uint32_t num_convolution_cores,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows,
    uint32_t has_actual_end>
TT_KERNEL void exchange(uint32_t entry_semaphore, uint32_t predecessor_semaphore, uint32_t carry_semaphore) {
    constexpr uint32_t row_bytes = channel_tiles * tile_row_bytes;
    static_assert(max_payload % read_alignment == 0, "row chunks must stay aligned in DRAM");
    Noc noc;
    Scratchpad<uint32_t> scratch(scratch::exchange_scratch);
    const auto history = TensorAccessor(tensor::history);
    const auto exchange_buffer = TensorAccessor(tensor::exchange_buffer);
    const auto final_history = TensorAccessor(tensor::final_history);
    Semaphore staged(sem::staged);
    Semaphore predecessor_ready(sem::predecessor_ready);

    // Scratch, from an aligned base: a slot for the chronology scalars, then three outgoing rows and three carry rows.
    // Raw: base/offset arithmetic below feeds raw reads/writes into scratch; no handle form yet.
    const uint32_t scratch_base = local_mem::detail::AddressAccess::address(scratch.local_mem());
    const uint32_t scalar_slot = align_up(scratch_base, read_alignment);
    const uint32_t outgoing = scalar_slot + read_alignment;
    const uint32_t carry = outgoing + history_rows * row_bytes;
    const auto offset_of = [&](uint32_t address) { return address - scratch_base; };
    const auto interval = read_chronology<has_actual_end, sp_rank, sp_size, local_rows>(noc, scalar_slot);
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

    // The replacement carry, on the owner of the interval's final segment: its last valid rows, preceded by the
    // caller's history or the predecessor's outgoing rows when that segment holds fewer than three.
    const auto assemble_carry = [&]() {
        const uint32_t carried = interval.carried_rows();
        for (uint32_t row = 0; row < history_rows; ++row) {
            const uint32_t position = carried + row;
            const bool from_layer_history = position < history_rows && interval.carry_continues_layer_history();
            const uint32_t page = position >= history_rows ? end_page + position - history_rows
                                  : from_layer_history     ? position
                                                           : predecessor_page + position;
            const uint32_t destination = offset_of(carry) + row * row_bytes;
            if (from_layer_history) {
                noc.async_read(history, scratch, row_bytes, {.page_id = page}, {.offset_bytes = destination});
            } else {
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
        const uint32_t teardown[] = {sem::teardown0.id, sem::teardown1.id};
        const uint32_t buffer_index[] = {sem::buffer_index0.id, sem::buffer_index1.id};
        auto* connection_arg_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(named_runtime_args));
        for (uint32_t c = 0; c < num_connections; ++c) {
            connection_arg_words[c * connection_args + teardown_arg] = teardown[c];
            connection_arg_words[c * connection_args + buffer_index_arg] = buffer_index[c];
        }
        size_t arg_index = named_runtime_args;
        tt::tt_fabric::RoutingPlaneConnectionManager connections;
        open_connections(connections, num_connections, arg_index);
        const size_t convolution_core_args = arg_index;
        auto* header = PacketHeaderPool::allocate_header();

        const auto semaphore_noc_address = [](uint32_t address) {
            return safe_get_noc_addr(my_x[0], my_y[0], address, 0);
        };
        uint8_t ranges[2] = {0, 0};
        if constexpr (forward_connection != no_fabric_connection) {
            ranges[forward_connection] = forward_range;
        }
        if constexpr (backward_connection != no_fabric_connection) {
            ranges[backward_connection] = backward_range;
        }
        // Bump `semaphore` on every other rank.
        const auto multicast_increment = [&](uint32_t semaphore) {
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
        // Wait for one arrival on a persistent semaphore, then rearm it. Only this core waits on it, and no rank sends
        // the next call's arrival before every rank has passed the next call's entry barrier.
        const auto wait_for_arrival = [](uint32_t semaphore) {
            auto* value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);
            noc_semaphore_wait_min(value, 1);
            noc_semaphore_set(value, 0);
        };

        // Entry barrier. A peer may already be bumping the next call's arrival, so consume this call's arrivals with an
        // atomic decrement rather than a reset.
        multicast_increment(entry_semaphore);
        staged.wait_min(num_convolution_cores);
        read_rows(outgoing_page, outgoing);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(entry_semaphore), sp_size - 1);
        noc_semaphore_inc(get_noc_addr(entry_semaphore), static_cast<uint32_t>(-static_cast<int32_t>(sp_size - 1)));
        noc_async_atomic_barrier();
        noc.async_read_barrier();

        unicast_rows(outgoing, predecessor_page, predecessor_semaphore);
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
    for (uint32_t row = 0; row < history_rows; ++row) {
        noc.async_write(
            scratch, final_history, row_bytes, {.offset_bytes = offset_of(carry) + row * row_bytes}, {.page_id = row});
    }
    noc.async_write_barrier();
}
