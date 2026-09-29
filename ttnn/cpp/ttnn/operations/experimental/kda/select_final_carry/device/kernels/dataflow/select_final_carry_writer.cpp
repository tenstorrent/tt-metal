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
#include "ttnn/cpp/ttnn/operations/experimental/kda/select_final_carry/device/kernels/dataflow/select_final_carry_common.hpp"

using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t tiles_cb = get_compile_time_arg_val(0);
constexpr uint32_t mode_cb = get_compile_time_arg_val(1);
constexpr uint32_t packet_pages = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t line_targets = get_compile_time_arg_val(4);
constexpr uint32_t start_hops_forward = get_compile_time_arg_val(5);
constexpr uint32_t range_hops_forward = get_compile_time_arg_val(6);
constexpr uint32_t start_hops_backward = get_compile_time_arg_val(7);
constexpr uint32_t range_hops_backward = get_compile_time_arg_val(8);
constexpr auto output_args = TensorAccessorArgs<9>();

// Write the final carry: copy the prefix locally when unsplit; when split, the owner writes its tail's final
// state locally and multicasts it along the sequence-parallel line, and the other ranks wait for its arrival.
void kernel_main() {
    using namespace kda_select_final_carry;
    size_t arg = 0;
    const uint32_t output_address = get_arg_val<uint32_t>(arg++);
    const uint32_t copy_begin = get_arg_val<uint32_t>(arg++);
    const uint32_t copy_end = get_arg_val<uint32_t>(arg++);
    const uint32_t send_begin = get_arg_val<uint32_t>(arg++);
    const uint32_t send_end = get_arg_val<uint32_t>(arg++);
    const bool fabric_worker = get_arg_val<uint32_t>(arg++);
    const auto output = TensorAccessor(output_args, output_address);

    cb_wait_front(mode_cb, 1);
    const Mode mode = static_cast<Mode>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(mode_cb))[0]);
    cb_pop_front(mode_cb, 1);

    if (mode == copy_prefix) {
        for (uint32_t tile = copy_begin; tile < copy_end;) {
            cb_wait_front(tiles_cb, packet_pages);
            uint32_t l1_address = get_read_ptr(tiles_cb);
            const uint32_t packet_end = tile + packet_pages < copy_end ? tile + packet_pages : copy_end;
            for (; tile < packet_end; ++tile, l1_address += page_size) {
                noc_async_write_page(tile, output, l1_address);
            }
            noc_async_writes_flushed();
            cb_pop_front(tiles_cb, packet_pages);
        }
        noc_async_write_barrier();
        return;
    }
    if (!fabric_worker) {
        return;
    }

    const size_t arrival_semaphore = get_arg_val<uint32_t>(arg++);
    const uint8_t drain_x = get_arg_val<uint32_t>(arg++);
    const uint8_t drain_y = get_arg_val<uint32_t>(arg++);
    const bool drain = get_arg_val<uint32_t>(arg++);
    const uint32_t arrivals = get_arg_val<uint32_t>(arg++);
    const size_t barrier_semaphore = get_arg_val<uint32_t>(arg++);
    const uint8_t barrier_x = get_arg_val<uint32_t>(arg++);
    const uint8_t barrier_y = get_arg_val<uint32_t>(arg++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg++);

    auto write_route = PacketHeaderPool::allocate_header_n(num_connections);
    auto scatter_route = PacketHeaderPool::allocate_header_n(num_connections);
    auto semaphore_route = PacketHeaderPool::allocate_header_n(num_connections);
    tt::tt_fabric::RoutingPlaneConnectionManager fabric;
    open_connections(fabric, num_connections, arg);

    uint8_t starts[] = {static_cast<uint8_t>(start_hops_forward), static_cast<uint8_t>(start_hops_backward)};
    uint8_t ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {
        starts[0] = starts[1];
        ranges[0] = ranges[1];
    }
    fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        fabric, write_route, starts, ranges, nullptr, page_size);
    fabric_multicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        fabric,
        scatter_route,
        starts,
        ranges,
        NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(page_size)}),
        page_size * 2);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric, semaphore_route, starts, ranges, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1u});

    // Every rank on the line must have entered this program before the owner overwrites its output.
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric,
        semaphore_route,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(barrier_x, barrier_y, barrier_semaphore, 0), 0});
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_semaphore), line_targets);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_semaphore), 0);

    if (mode == broadcast) {
        for (uint32_t tile = send_begin; tile < send_end;) {
            cb_wait_front(tiles_cb, packet_pages);
            uint32_t l1_address = get_read_ptr(tiles_cb);
            const uint32_t packet_end = tile + packet_pages < send_end ? tile + packet_pages : send_end;
            while (tile < packet_end) {
                // Scatter writes carry at most two destination pages.
                if (packet_end - tile == 1) {
                    noc_async_write_page(tile, output, l1_address);
                    fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        fabric,
                        write_route,
                        l1_address,
                        tt::tt_fabric::NocUnicastCommandHeader{
                            linear::addrgen_detail::get_noc_address(output, tile, 0)});
                    l1_address += page_size;
                    tile += 1;
                } else {
                    noc_async_write_page(tile, output, l1_address);
                    noc_async_write_page(tile + 1, output, l1_address + page_size);
                    fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                        fabric,
                        scatter_route,
                        l1_address,
                        tt::tt_fabric::NocUnicastScatterCommandHeader(
                            {linear::addrgen_detail::get_noc_address(output, tile, 0),
                             linear::addrgen_detail::get_noc_address(output, tile + 1, 0)}),
                        page_size * 2);
                    l1_address += 2 * page_size;
                    tile += 2;
                }
                noc_async_writes_flushed();
            }
            cb_pop_front(tiles_cb, packet_pages);
        }
        // Flushed after this link's data, the increment tells each receiver that its share has landed.
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric,
            semaphore_route,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                safe_get_noc_addr(drain_x, drain_y, arrival_semaphore, 0), 0});
    } else if (drain) {
        auto* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_semaphore);
        noc_semaphore_wait(arrived, arrivals);
        noc_semaphore_set(arrived, 0);
    }
    close_connections(fabric);
    noc_async_write_barrier();
}
