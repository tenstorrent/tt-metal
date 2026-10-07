// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t rows_cb = get_compile_time_arg_val(0);
constexpr uint32_t scalar_cb = get_compile_time_arg_val(1);
constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
constexpr uint32_t chunk_bytes = get_compile_time_arg_val(3);
constexpr uint32_t sp_rank = get_compile_time_arg_val(4);
constexpr uint32_t sp_size = get_compile_time_arg_val(5);
constexpr uint32_t local_rows = get_compile_time_arg_val(6);
constexpr bool has_actual_end = get_compile_time_arg_val(7);
constexpr uint32_t line_targets = get_compile_time_arg_val(8);
constexpr uint32_t start_hops_forward = get_compile_time_arg_val(9);
constexpr uint32_t range_hops_forward = get_compile_time_arg_val(10);
constexpr uint32_t start_hops_backward = get_compile_time_arg_val(11);
constexpr uint32_t range_hops_backward = get_compile_time_arg_val(12);
constexpr uint32_t successor_hops = get_compile_time_arg_val(13);
// The successor is the forward line neighbor, so the line's first connection reaches it.
constexpr bool successor_on_line = get_compile_time_arg_val(14);
constexpr uint32_t gather_workers = get_compile_time_arg_val(15);
constexpr uint32_t gathered_semaphore = get_compile_time_arg_val(16);
constexpr auto predecessor_args = TensorAccessorArgs<17>();
constexpr auto final_args = TensorAccessorArgs<predecessor_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<final_args.next_compile_time_args_offset()>();
constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();

constexpr uint32_t history_rows = kda_chronology::selection::history_rows;
constexpr uint32_t chunks_per_row = row_bytes / chunk_bytes;

// Send this rank's outgoing history to the next physical rank, which receives it as its predecessor history, and,
// on the rank owning the chronologically last valid token, its local final history to every rank as the carry. The
// gather workers stage both from the projection while this worker runs the line barrier. Phase one opens the line
// connections for the barrier and the final multicast; phase two opens one connection to the successor (on a 2D
// fabric a connection routes to its own destination, and two connections may not share a direction).
void kernel_main() {
    size_t arg = 0;
    const uint32_t predecessor_address = get_arg_val<uint32_t>(arg++);
    const uint32_t final_address = get_arg_val<uint32_t>(arg++);
    const uint32_t start_address = get_arg_val<uint32_t>(arg++);
    const uint32_t end_address = get_arg_val<uint32_t>(arg++);
    const size_t barrier_semaphore = get_arg_val<uint32_t>(arg++);
    const size_t arrival_semaphore = get_arg_val<uint32_t>(arg++);
    const uint8_t worker_x = get_arg_val<uint32_t>(arg++);
    const uint8_t worker_y = get_arg_val<uint32_t>(arg++);
    const uint32_t line_connections = get_arg_val<uint32_t>(arg++);

    const auto predecessor = TensorAccessor(predecessor_args, predecessor_address);
    const auto final_history = TensorAccessor(final_args, final_address);

    // Chronology: only the owner of the last valid token sends a final history.
    cb_reserve_back(scalar_cb, 1);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scalar_cb));
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(scalar_cb), sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t actual_start = words[0];
    kda_chronology::Topology topology{};
    if constexpr (has_actual_end) {
        const auto end = TensorAccessor(end_args, end_address);
        noc_async_read(end.get_noc_addr(0), get_write_ptr(scalar_cb), sizeof(uint32_t));
        noc_async_read_barrier();
        topology = kda_chronology::derive_interval(actual_start, words[0], sp_rank, sp_size, local_rows);
    } else {
        topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    }
    const bool owner = topology.final_owner == sp_rank;

    // The gather workers stage the six rows here: outgoing history, then local final history.
    cb_reserve_back(rows_cb, 1);
    const uint32_t rows = get_write_ptr(rows_cb);

    const uint64_t arrival_noc_address = safe_get_noc_addr(worker_x, worker_y, arrival_semaphore, 0);
    // The outgoing rows go to the successor's predecessor history, followed by its arrival increment. A one-header
    // route uses the manager's first connection.
    const auto send_to_successor = [&](tt::tt_fabric::RoutingPlaneConnectionManager& fabric) {
        auto write_route = PacketHeaderPool::allocate_header_n(1);
        auto semaphore_route = PacketHeaderPool::allocate_header_n(1);
        uint8_t hops[] = {static_cast<uint8_t>(successor_hops)};
        for (uint32_t row = 0; row < history_rows; ++row) {
            for (uint32_t chunk = 0; chunk < chunks_per_row; ++chunk) {
                fabric_unicast_noc_unicast_write(
                    fabric,
                    write_route,
                    rows + row * row_bytes + chunk * chunk_bytes,
                    chunk_bytes,
                    tt::tt_fabric::NocUnicastCommandHeader{
                        linear::addrgen_detail::get_noc_address(predecessor, row, chunk * chunk_bytes)},
                    hops);
            }
        }
        fabric_unicast_noc_unicast_atomic_inc(
            fabric,
            semaphore_route,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{arrival_noc_address, 1u, true},
            hops);
    };
    {
        auto write_route = PacketHeaderPool::allocate_header_n(line_connections);
        auto semaphore_route = PacketHeaderPool::allocate_header_n(line_connections);
        tt::tt_fabric::RoutingPlaneConnectionManager fabric;
        open_connections(fabric, line_connections, arg);
        uint8_t starts[] = {static_cast<uint8_t>(start_hops_forward), static_cast<uint8_t>(start_hops_backward)};
        uint8_t ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
        if (ranges[0] == 0) {
            starts[0] = starts[1];
            ranges[0] = ranges[1];
        }
        fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            fabric, write_route, starts, ranges, nullptr, chunk_bytes);
        fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            fabric, semaphore_route, starts, ranges, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1u});

        // Every rank on the line must have entered this program before anyone writes into its outputs.
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric,
            semaphore_route,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                safe_get_noc_addr(worker_x, worker_y, barrier_semaphore, 0), 0});
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_semaphore), line_targets);
        // Subtract instead of clearing: a faster rank may already have announced the next call.
        noc_semaphore_inc(get_noc_addr(barrier_semaphore), 0u - line_targets);
        noc_async_atomic_barrier();
        noc_semaphore_wait(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gathered_semaphore)), gather_workers);

        if (owner) {
            for (uint32_t row = 0; row < history_rows; ++row) {
                const uint32_t source = rows + (history_rows + row) * row_bytes;
                noc_async_write(source, final_history.get_noc_addr(row), row_bytes);
                for (uint32_t chunk = 0; chunk < chunks_per_row; ++chunk) {
                    fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        fabric,
                        write_route,
                        source + chunk * chunk_bytes,
                        tt::tt_fabric::NocUnicastCommandHeader{
                            linear::addrgen_detail::get_noc_address(final_history, row, chunk * chunk_bytes)});
                }
            }
            // Flushed after the rows on these connections, the increment tells each receiver they have landed.
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric, semaphore_route, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{arrival_noc_address, 0});
        }
        if constexpr (successor_on_line) {
            send_to_successor(fabric);
        }
        close_connections(fabric);
    }
    if constexpr (!successor_on_line) {
        const uint32_t successor_connections = get_arg_val<uint32_t>(arg++);
        tt::tt_fabric::RoutingPlaneConnectionManager fabric;
        open_connections(fabric, successor_connections, arg);
        send_to_successor(fabric);
        close_connections(fabric);
    }

    // The predecessor's outgoing history, and the owner's final history unless this rank owns it.
    const uint32_t arrivals = owner ? 1 : 2;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_semaphore), arrivals);
    noc_semaphore_inc(get_noc_addr(arrival_semaphore), 0u - arrivals);
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    cb_push_back(rows_cb, 1);
    cb_push_back(scalar_cb, 1);
}
