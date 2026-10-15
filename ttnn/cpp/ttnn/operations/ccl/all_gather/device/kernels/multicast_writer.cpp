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
#include "gather_args.hpp"
#include "gather_noc.hpp"
#include "gather_fabric.hpp"

using address_t = uint32_t;

void kernel_main() {
    namespace ct = gather_args::multicast_writer::ct;
    namespace rt = gather_args::multicast_writer::rt;

    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t chunk_size = get_compile_time_arg_val(ct::chunk_size);
    constexpr uint32_t out_cpp = get_compile_time_arg_val(ct::out_cpp);
    constexpr uint32_t cps = get_compile_time_arg_val(ct::cps);
    constexpr uint32_t num_devices = get_compile_time_arg_val(ct::num_devices);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(ct::cb_id);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(ct::cb_page_size);
    constexpr uint32_t packet_size = get_compile_time_arg_val(ct::packet_size);
    constexpr bool load_balance_across_alt_routes = get_compile_time_arg_val(ct::alt_routes) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(ct::num_connections);
    constexpr uint32_t k = get_compile_time_arg_val(ct::k);
    constexpr uint32_t walk_stride = get_compile_time_arg_val(ct::walk_stride);
    constexpr uint32_t walk_xfer = get_compile_time_arg_val(ct::walk_xfer);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(ct::slot_bytes);
    constexpr RunSource out_runs = run_source_from_bits(get_compile_time_arg_val(ct::out_runs));
    constexpr auto output_tensor_args = TensorAccessorArgs<ct::tensor_args>();

    constexpr bool enable_fabric = (num_connections > 0);
    static_assert(out_cpp % k == 0 && cps % k == 0, "pages and stripes are whole walk pages");
    static_assert(chunk_size <= slot_bytes && cb_page_size % slot_bytes == 0, "a CB entry is whole slots");
    static_assert(slot_bytes == chunk_size || (out_cpp == 1 && k == 1), "a padded slot is one chunk per page");
    static_assert(walk_xfer == 1 || walk_xfer * slot_bytes <= packet_size, "a column fits one packet");
    static_assert(walk_xfer == 1 || walk_xfer * slot_bytes <= NOC_MAX_BURST_SIZE, "a column is one NoC command");
    constexpr uint32_t chunks_per_cb_entry = cb_page_size / slot_bytes;
    // A run is emitted as one scatter chunk starting at its source offset within the packet, so every chunk
    // size has to keep source and destination NoC-write aligned.
    static_assert(
        chunk_size % 16 == 0 && slot_bytes % 16 == 0, "chunk size must be a multiple of the NoC write alignment");

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    const address_t output_tensor_address = get_arg_val<address_t>(rt::output_addr);
    const uint32_t slice_first_chunk = get_arg_val<uint32_t>(rt::slice_first);
    const uint32_t slice_chunks = get_arg_val<uint32_t>(rt::slice_chunks);
    const uint32_t device_idx = get_arg_val<uint32_t>(rt::device_idx);
    const address_t barrier_sem = get_arg_val<uint32_t>(rt::barrier_sem);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(rt::barrier_noc_x);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(rt::barrier_noc_y);
    const uint32_t rot = get_arg_val<uint32_t>(rt::rot);
    size_t arg_idx = rt::route_args;
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
    PacketPacker<packet_size, slot_bytes, McastSink<load_balance_across_alt_routes>> fabric(sink);

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

    // The walk stride was derived on the host from this accessor's contiguous_page_stride().
    ASSERT(out_cpp > k || output_tensor_accessor.contiguous_page_stride() == walk_stride);
    const RunSource out_src = checked_run_source<k, out_cpp>(out_runs, output_tensor_accessor, walk_stride);
    const uint32_t slice_end = slice_first_chunk + slice_chunks;

    // The reader fills the CB with this same walk (same rot), so every entry holds the same seqnos.
    TiledWalk<k, walk_stride, walk_xfer> walk;
    walk.init(slice_end, rot, walk_start(slice_first_chunk));
    StripeMap<cps, num_devices> map;
    map.init(device_idx, slice_end);

    auto output_addr = [&](uint32_t page, uint32_t off) {
        return output_tensor_accessor.get_noc_addr(page, off, noc.get_noc_id());
    };

    for (uint32_t chunks_sent = 0; chunks_sent < slice_chunks;) {
        const uint32_t batch = std::min(chunks_per_cb_entry, slice_chunks - chunks_sent);
        cb.wait_front(1);
        emit_runs<out_cpp, chunk_size, slot_bytes>(
            walk,
            map,
            output_tensor_accessor,
            out_src,
            cb.get_read_ptr(),
            batch,
            output_addr,
            [&](uint32_t l1, uint32_t page, uint32_t off, uint32_t run) {
                if constexpr (enable_fabric) {
                    fabric.queue_segment(
                        l1,
                        tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page, off),
                        run * slot_bytes);
                }
                local_write<slot_bytes>(noc, l1, output_addr(page, off), run);
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
