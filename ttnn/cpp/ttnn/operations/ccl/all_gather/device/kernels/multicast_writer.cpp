// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>

#include "multicast_common.hpp"

using address_t = uint32_t;

void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(2);
    constexpr uint32_t num_devices = get_compile_time_arg_val(3);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t packet_size = get_compile_time_arg_val(6);
    constexpr bool load_balance_across_alt_routes = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(8);
    constexpr bool do_init_barrier = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t xfer_cap = get_compile_time_arg_val(10);  // longest run the walk may emit, in chunks
    constexpr auto output_tensor_args = TensorAccessorArgs<11>();

    constexpr bool enable_fabric = (num_connections > 0);
    constexpr uint32_t chunks_per_cb_entry = cb_page_size / output_chunk_size;
    constexpr uint32_t xfer_uncapped = chunks_per_transfer(packet_size, output_chunk_size);
    constexpr uint32_t xfer_max = xfer_uncapped < xfer_cap ? xfer_uncapped : xfer_cap;
    // A chunk bigger than a burst cannot be one NOC command, so it takes the generic path.
    constexpr bool one_command = output_chunk_size <= NOC_MAX_BURST_SIZE;
    // A run is emitted as one scatter chunk starting at its source offset within the packet, so every chunk
    // size has to keep source and destination NoC-write aligned.
    static_assert(output_chunk_size % 16 == 0, "chunk size must be a multiple of the NoC write alignment");

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t slice_first_chunk = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t device_idx = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t line_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_e_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_w_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_spine_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t line_hops_alt = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_e_hops_alt = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_w_hops_alt = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_spine_hops_alt = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t line_dir = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_e_dir = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_w_dir = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_spine_dir = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    if constexpr (enable_fabric) {
        open_connections(fabric_connection, num_connections, arg_for_fab);
    }

    // Build the line then rect ranges, in connection order (matches the host). Each hop count is placed at
    // its physical slot; an absent branch (hop == 0) is skipped so it can't clobber a live slot.
    FabricRange ranges[2] = {};
    FabricRange ranges_alt[2] = {};
    uint32_t conn = 0;
    if (line_hops > 0) {
        uint8_t hops[4] = {}, hops_alt[4] = {};
        hops[line_dir] = line_hops;
        hops_alt[line_dir] = line_hops_alt;
        ranges[conn] = make_fabric_range(hops[0], hops[1], hops[2], hops[3]);
        ranges_alt[conn] = make_fabric_range(hops_alt[0], hops_alt[1], hops_alt[2], hops_alt[3]);
        ++conn;
    }
    if (rect_spine_hops > 0) {
        uint8_t hops[4] = {}, hops_alt[4] = {};
        if (rect_e_hops > 0) {
            hops[rect_e_dir] = rect_e_hops;
            hops_alt[rect_e_dir] = rect_e_hops_alt;
        }
        if (rect_w_hops > 0) {
            hops[rect_w_dir] = rect_w_hops;
            hops_alt[rect_w_dir] = rect_w_hops_alt;
        }
        hops[rect_spine_dir] = rect_spine_hops;
        hops_alt[rect_spine_dir] = rect_spine_hops_alt;
        ranges[conn] = make_fabric_range(hops[0], hops[1], hops[2], hops[3]);
        ranges_alt[conn] = make_fabric_range(hops_alt[0], hops_alt[1], hops_alt[2], hops_alt[3]);
        ++conn;
    }

    // Allocate header and set state for data sends
    FabricWriter<output_chunk_size, packet_size, load_balance_across_alt_routes> fabric(
        noc, fabric_connection, num_connections, ranges, ranges_alt);

    // Allocate header and set state for semaphore sends
    uint8_t sem_route_id = 0;
    if constexpr (enable_fabric) {
        sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
        uint8_t starts[1] = {1};

        fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            fabric_connection,
            sem_route_id,
#ifndef FABRIC_2D
            starts,
#endif
            ranges,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0u,    // ignore
                1u});  // increment 1
    }

    // Initialization barrier:
    // In some cases we don't have a guarantee that the output tensor has been allocated
    // on remote devices (every device's command queue executes asynchronously). So we wait
    // for this kernel to begin execution on all remote devices before sending any data.
    //
    // Mechanism:
    // Each worker core syncs with its mirror core (the same core) on all remote devices.
    // Reader fires sem increment forward, and also owns sem wait + decrement.
    // Writer fires sem increment backward, and implicitly gets blocked waiting for CB to
    // contain valid data.
    if constexpr (do_init_barrier && enable_fabric) {
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
    }

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    const auto plan = walk_plan<output_chunks_per_page, output_chunk_size, xfer_max>(output_tensor_accessor);

    TiledWalk walk;
    walk.init(slice_first_chunk, slice_chunks, 0, plan.stride, plan.xfer);
    StripeMap<output_chunks_per_stripe, num_devices> map;
    map.init(device_idx);

    auto output_addr = [&](uint32_t global) {
        return output_tensor_accessor.get_noc_addr(
            page_of<output_chunks_per_page>(global),
            byte_off_of<output_chunks_per_page, output_chunk_size>(global),
            noc.get_noc_id());
    };
    auto run_addr = [&](uint32_t chunk) { return output_addr(map.at(chunk).global); };

    auto local_write = [&](uint32_t l1_read_addr, uint64_t dst, uint32_t chunks) {
        // Posted write on a separate VC so it doesn't contend with the fabric writes on the same NOC.
        if constexpr (one_command) {
            noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                CoreLocalMem<uint32_t>(l1_read_addr),
                tensor_accessor::Page(dst, 0),
                chunks * output_chunk_size,
                {},
                {},
                {.vc = NOC_UNICAST_WRITE_VC + 1});
        } else {
            noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
                CoreLocalMem<uint32_t>(l1_read_addr),
                tensor_accessor::Page(dst, 0),
                output_chunk_size,
                {},
                {},
                {.vc = NOC_UNICAST_WRITE_VC + 1});
        }
    };

    for (uint32_t chunks_sent = 0; chunks_sent < slice_chunks;) {
        const uint32_t batch = std::min(chunks_per_cb_entry, slice_chunks - chunks_sent);
        cb.wait_front(1);
        uint32_t l1_read_addr = cb.get_read_ptr();

        for (uint32_t left = batch; left > 0;) {
            const auto pos = map.at(walk.chunk());
            const uint32_t run =
                next_run<output_chunks_per_page>(walk, output_tensor_accessor, plan.out, pos.global, pos.row_end, left);
            const uint64_t dst = output_addr(pos.global);
            ASSERT(run_is_linear(walk, run, output_chunk_size, dst, run_addr));
            if constexpr (enable_fabric) {
                const uint32_t page = page_of<output_chunks_per_page>(pos.global);
                const uint32_t off = byte_off_of<output_chunks_per_page, output_chunk_size>(pos.global);
                fabric.queue_segment(
                    l1_read_addr,
                    tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page, off),
                    run * output_chunk_size);
            }
            local_write(l1_read_addr, dst, run);
            l1_read_addr += run * output_chunk_size;
            left -= run;
            walk.advance(run);
        }

        noc.async_writes_flushed<NocOptions::POSTED>();  // wait for local writes
        if constexpr (enable_fabric) {
            fabric.flush_packet_and_wait();  // wait for Fabric writes
        }
        cb.pop_front(1);
        chunks_sent += batch;
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion barrier:
    // We must only exit this op after guaranteeing that all remote data has arrived.
    //
    // Mechanism:
    // Each worker core sends a sem to its mirror core (the same core) on all remote devices. The sem
    // is sent after all data sends on a particular link, so it's correctly ordered at the receiver.
    // Reader fires sem increment forward, and also owns sem wait + decrement.
    // Writer fires sem increment backward, and exits immediately.
    if constexpr (enable_fabric) {
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
    }

    if constexpr (enable_fabric) {
        close_connections(fabric_connection);
    }
    noc.async_write_barrier();
}
