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
    namespace ct = gather_args::multicast_reader::ct;
    namespace rt = gather_args::multicast_reader::rt;

    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t in_cpp = get_compile_time_arg_val(ct::in_cpp);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(ct::chunk_size);
    constexpr uint32_t out_cpp = get_compile_time_arg_val(ct::out_cpp);
    constexpr uint32_t cps = get_compile_time_arg_val(ct::cps);
    constexpr uint32_t num_devices = get_compile_time_arg_val(ct::num_devices);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(ct::cb_id);
    constexpr uint32_t cb_depth = get_compile_time_arg_val(ct::cb_depth);
    constexpr uint32_t cb_entry_bytes = get_compile_time_arg_val(ct::cb_entry_bytes);
    constexpr uint32_t packet_size = get_compile_time_arg_val(ct::packet_size);
    constexpr bool load_balance_across_alt_routes = get_compile_time_arg_val(ct::alt_routes) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(ct::num_connections);
    constexpr uint32_t k = get_compile_time_arg_val(ct::k);
    constexpr uint32_t walk_stride = get_compile_time_arg_val(ct::walk_stride);
    constexpr uint32_t walk_xfer = get_compile_time_arg_val(ct::walk_xfer);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(ct::slot_bytes);
    constexpr RunSource in_runs = run_source_from_bits(get_compile_time_arg_val(ct::in_runs));
    constexpr RunSource out_runs = run_source_from_bits(get_compile_time_arg_val(ct::out_runs));
    constexpr auto input_tensor_args = TensorAccessorArgs<ct::tensor_args>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr bool enable_fabric = (num_connections > 0);
    check_walk_ct<k, in_cpp, out_cpp, cps, chunk_size, slot_bytes, cb_entry_bytes, walk_xfer, packet_size>();
    constexpr uint32_t chunks_per_entry = cb_entry_bytes / slot_bytes;

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    const address_t input_tensor_address = get_arg_val<address_t>(rt::input_addr);
    const address_t output_tensor_address = get_arg_val<address_t>(rt::output_addr);
    const uint32_t slice_first = get_arg_val<uint32_t>(rt::slice_first);
    const uint32_t slice_count = get_arg_val<uint32_t>(rt::slice_count);
    const uint32_t device_idx = get_arg_val<uint32_t>(rt::device_idx);
    const address_t barrier_sem = get_arg_val<uint32_t>(rt::barrier_sem);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(rt::barrier_noc_x);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(rt::barrier_noc_y);
    const uint32_t barrier_wait_value = get_arg_val<uint32_t>(rt::barrier_wait_value);
    size_t arg_idx = rt::route_args;
    const auto route_args = McastRouteArgs::from_args(arg_idx);
    size_t arg_for_fab = arg_idx;

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
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
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), barrier_wait_value);
    // Subtract, don't clear: a peer running ahead may already have posted credits for its next invocation
    noc_semaphore_inc(get_noc_addr(barrier_sem), uint32_t{0} - barrier_wait_value);

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    // NOC transaction IDs to cycle between, and vars to keep track of state
    constexpr uint32_t max_trid = cb_depth;
    static_assert(max_trid <= NOC_MAX_TRANSACTION_ID, "max_trid exceeds max supported value");
    uint32_t curr_trid = 1;
    uint32_t wait_trid = 1;
    bool txns_in_flight = false;

    // Get write pointer (to write to CB) and read pointer (to read from CB).
    // We need to manually keep track of these pointers since we don't push_back
    // after every reserve_back when using NOC transaction IDs, so get_read/write_ptr()
    // will return stale values.
    auto l1_base_addr = cb.get_write_ptr();
    auto l1_end_addr = l1_base_addr + (cb_depth * cb_entry_bytes);
    auto l1_write_addr = l1_base_addr;
    auto l1_read_addr = l1_base_addr;

    check_walk_stride(output_tensor_accessor, k, out_cpp, walk_stride);
    const RunSource in_src = checked_run_source(in_runs, input_tensor_accessor, k, in_cpp, walk_stride);
    const RunSource out_src = checked_run_source(out_runs, output_tensor_accessor, k, out_cpp, walk_stride);
    const uint32_t slice_end = slice_first + slice_count;

    StripeMap<cps, num_devices> map;
    map.init(device_idx, slice_end);
    // Same walk, one CB entry apart: reads stay in flight while the entry before them is sent. The writer
    // drains the CB with the same walk, so every entry holds the same seqnos on both sides.
    TiledWalk<k, walk_stride, walk_xfer> read_walk;
    read_walk.init(slice_end, walk_start(slice_first));
    auto send_walk = read_walk;
    uint32_t chunks_read = 0;
    uint32_t chunks_sent = 0;

    auto input_addr = [&](uint32_t chunk) {
        return input_tensor_accessor.get_noc_addr(
            page_of<in_cpp>(chunk), byte_off_of<in_cpp, chunk_size>(chunk), noc.get_noc_id());
    };

    auto output_addr = [&](uint32_t page, uint32_t off) {
        return output_tensor_accessor.get_noc_addr(page, off, noc.get_noc_id());
    };

    // Read input tensor and fill CB page
    auto fill_entry = [&]() __attribute__((always_inline)) {
        const uint32_t batch = std::min(chunks_per_entry, slice_count - chunks_read);
        l1_write_addr = read_input_runs<in_cpp, chunk_size, slot_bytes>(
            read_walk,
            input_tensor_accessor,
            in_src,
            slice_end,
            l1_write_addr,
            batch,
            input_addr,
            [&](uint64_t src, uint32_t l1, uint32_t run) {
                read_run<slot_bytes, chunk_size>(noc, src, l1, run, curr_trid);
            });
        chunks_read += batch;
        if (l1_write_addr == l1_end_addr) {
            l1_write_addr = l1_base_addr;
        }
    };

    // Send Fabric data in our dir
    auto send_entry = [&]() __attribute__((always_inline)) {
        if constexpr (enable_fabric) {
            const uint32_t batch = std::min(chunks_per_entry, slice_count - chunks_sent);
            l1_read_addr = emit_runs<out_cpp, chunk_size, slot_bytes>(
                send_walk,
                map,
                output_tensor_accessor,
                out_src,
                l1_read_addr,
                batch,
                output_addr,
                [&](uint32_t l1, uint32_t page, uint32_t off, uint32_t run) {
                    fabric.queue_segment(
                        l1,
                        tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page, off),
                        run * slot_bytes);
                });
            chunks_sent += batch;
            fabric.flush_packet_and_wait();
            if (l1_read_addr == l1_end_addr) {
                l1_read_addr = l1_base_addr;
            }
        }
    };

    // We reserve two to kick start the pipeline, and then it is steady state
    cb.reserve_back(2);
    while (chunks_read < slice_count) {
        fill_entry();

        curr_trid = (curr_trid == max_trid) ? 1 : curr_trid + 1;
        if (txns_in_flight) {
            // push_back() will unblock the writer to send Fabric data in opposite dir
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = wait_trid});
            cb.push_back(1);
            wait_trid = (wait_trid == max_trid) ? 1 : (wait_trid + 1);

            send_entry();

            // Reserve for next block.
            // Reserve back is not incremental, so to reserve one more, we need to reserve 2.
            // This accounts for the one we already have reserved (for in-flight read).
            cb.reserve_back(2);
        }
        txns_in_flight = true;
    }
    // Drain in-flight reads
    while (wait_trid != curr_trid) {
        // push_back() will unblock the writer to send Fabric data in opposite dir
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = wait_trid});
        cb.push_back(1);
        wait_trid = (wait_trid == max_trid) ? 1 : (wait_trid + 1);

        send_entry();
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
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), barrier_wait_value);
    // Subtract, don't clear: a peer running ahead may already have posted credits for its next invocation
    noc_semaphore_inc(get_noc_addr(barrier_sem), uint32_t{0} - barrier_wait_value);
    noc.async_atomic_barrier();

    if constexpr (enable_fabric) {
        close_connections(fabric_connection);
    }
    noc.async_write_barrier();
}
