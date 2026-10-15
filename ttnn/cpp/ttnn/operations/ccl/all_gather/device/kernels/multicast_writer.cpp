// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>

#include "gather_walk.hpp"
#include "gather_noc.hpp"
#include "gather_fabric.hpp"

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
    constexpr uint32_t run_cap_bytes = get_compile_time_arg_val(9);  // longest run the walk may emit; 0 = no cap
    constexpr auto output_tensor_args = TensorAccessorArgs<10>();

    constexpr bool enable_fabric = (num_connections > 0);
    constexpr uint32_t chunks_per_cb_entry = cb_page_size / output_chunk_size;
    constexpr uint32_t xfer_max = chunks_per_transfer(packet_size, output_chunk_size, run_cap_bytes);
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
    const auto route_args = McastRouteArgs::from_args(arg_idx);
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

    FabricRange ranges[2] = {};
    FabricRange ranges_alt[2] = {};
    build_mcast_ranges(route_args, ranges, ranges_alt);

    // Allocate header and set state for data sends
    McastSink<load_balance_across_alt_routes> sink(noc, fabric_connection, num_connections, ranges, ranges_alt);
    PacketPacker<packet_size, output_chunk_size, McastSink<load_balance_across_alt_routes>> fabric(sink);

    // Allocate header and set state for semaphore sends
    uint8_t sem_route_id = 0;
    if constexpr (enable_fabric) {
        sem_route_id = mcast_sem_route(fabric_connection, num_connections, ranges);
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
    if constexpr (enable_fabric) {
        mcast_sem_inc(fabric_connection, sem_route_id, barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem);
    }

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    const auto plan = walk_plan<output_chunks_per_page, output_chunk_size, xfer_max>(output_tensor_accessor);

    TiledWalk walk;
    walk.init(slice_first_chunk, slice_chunks, 0, plan.stride, plan.xfer);
    StripeMap<output_chunks_per_stripe, num_devices> map;
    map.init(device_idx);

    auto output_addr = [&](uint32_t page, uint32_t off) {
        return output_tensor_accessor.get_noc_addr(page, off, noc.get_noc_id());
    };

    for (uint32_t chunks_sent = 0; chunks_sent < slice_chunks;) {
        const uint32_t batch = std::min(chunks_per_cb_entry, slice_chunks - chunks_sent);
        cb.wait_front(1);
        emit_runs<output_chunks_per_page, output_chunk_size>(
            walk,
            map,
            output_tensor_accessor,
            plan.out,
            cb.get_read_ptr(),
            batch,
            output_addr,
            [&](uint32_t l1, uint32_t page, uint32_t off, uint32_t run) {
                if constexpr (enable_fabric) {
                    fabric.queue_segment(
                        l1,
                        tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page, off),
                        run * output_chunk_size);
                }
                local_write<output_chunk_size>(noc, l1, output_addr(page, off), run);
            });

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
        mcast_sem_inc(fabric_connection, sem_route_id, barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem);
    }

    if constexpr (enable_fabric) {
        close_connections(fabric_connection);
    }
    noc.async_write_barrier();
}
