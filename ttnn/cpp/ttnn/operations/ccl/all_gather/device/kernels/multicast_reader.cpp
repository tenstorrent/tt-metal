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
    constexpr uint32_t split_factor = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(3);
    constexpr uint32_t num_devices = get_compile_time_arg_val(4);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_depth = get_compile_time_arg_val(6);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t packet_size = get_compile_time_arg_val(8);
    constexpr bool load_balance_across_alt_routes = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(10);
    constexpr bool do_init_barrier = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t run_cap_bytes = get_compile_time_arg_val(12);  // longest run the walk may emit; 0 = no cap
    constexpr auto input_tensor_args = TensorAccessorArgs<13>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr bool enable_fabric = (num_connections > 0);
    constexpr uint32_t chunks_per_cb_entry = cb_page_size / output_chunk_size;
    constexpr uint32_t xfer_max = chunks_per_transfer(packet_size, output_chunk_size, run_cap_bytes);
    // A chunk bigger than a burst cannot be one NOC command, so it takes the generic path.
    constexpr bool one_command = output_chunk_size <= NOC_MAX_BURST_SIZE;

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t slice_first_chunk = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t device_idx = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_wait_value = get_arg_val<uint32_t>(arg_idx++);
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
    if constexpr (do_init_barrier) {
        if constexpr (enable_fabric) {
            uint64_t barrier_sem_noc_addr_in_pkt =
                safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
            fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_connection,
                sem_route_id,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), barrier_wait_value);
        // Subtract, don't clear: a peer running ahead may already have posted credits for its next invocation
        noc_semaphore_inc(get_noc_addr(barrier_sem), uint32_t{0} - barrier_wait_value);
    }

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
    auto l1_end_addr = l1_base_addr + (cb_depth * cb_page_size);
    auto l1_write_addr = l1_base_addr;
    auto l1_read_addr = l1_base_addr;

    const auto plan = walk_plan<output_chunks_per_page, output_chunk_size, xfer_max>(output_tensor_accessor);
    // Our input yields runs only when it strides the way the walk does.
    const auto in_src = run_source(
        input_tensor_accessor.get_aligned_page_size() == split_factor * output_chunk_size,
        split_factor,
        input_tensor_accessor.contiguous_page_stride(),
        plan.stride);
    const uint32_t input_end_chunk = slice_first_chunk + slice_chunks;

    StripeMap<output_chunks_per_stripe, num_devices> map;
    map.init(device_idx);
    // Same walk, one CB entry apart: reads stay in flight while the entry before them is sent.
    TiledWalk read_walk;
    read_walk.init(slice_first_chunk, slice_chunks, 0, plan.stride, plan.xfer);
    TiledWalk send_walk = read_walk;
    uint32_t chunks_read = 0;
    uint32_t chunks_sent = 0;

    auto input_addr = [&](uint32_t chunk) {
        return input_tensor_accessor.get_noc_addr(
            page_of<split_factor>(chunk), byte_off_of<split_factor, output_chunk_size>(chunk), noc.get_noc_id());
    };

    auto read_run = [&](uint64_t src, uint32_t chunks) __attribute__((always_inline)) {
        if constexpr (one_command) {
            noc.async_read<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
                tensor_accessor::Page(src, 0),
                CoreLocalMem<uint32_t>(l1_write_addr),
                chunks * output_chunk_size,
                {},
                {},
                {.trid = curr_trid});
        } else {
            noc.async_read<NocOptions::TXN_ID>(
                tensor_accessor::Page(src, 0),
                CoreLocalMem<uint32_t>(l1_write_addr),
                output_chunk_size,
                {},
                {},
                {.trid = curr_trid});
        }
    };

    // Read input tensor and fill CB page
    auto fill_entry = [&]() __attribute__((always_inline)) {
        const uint32_t batch = std::min(chunks_per_cb_entry, slice_chunks - chunks_read);
        for (uint32_t left = batch; left > 0;) {
            const uint32_t chunk = read_walk.chunk();
            const uint32_t run =
                next_run<split_factor>(read_walk, input_tensor_accessor, in_src, chunk, input_end_chunk, left);
            const uint64_t src = input_addr(chunk);
            ASSERT(run_is_linear(read_walk, run, output_chunk_size, src, input_addr));
            read_run(src, run);
            l1_write_addr += run * output_chunk_size;
            left -= run;
            read_walk.advance(run);
        }
        chunks_read += batch;
        if (l1_write_addr == l1_end_addr) {
            l1_write_addr = l1_base_addr;
        }
    };

    // Send Fabric data in our dir
    auto send_entry = [&]() __attribute__((always_inline)) {
        if constexpr (enable_fabric) {
            auto run_addr = [&](uint32_t chunk) {
                const uint32_t global = map.at(chunk).global;
                return output_tensor_accessor.get_noc_addr(
                    page_of<output_chunks_per_page>(global),
                    byte_off_of<output_chunks_per_page, output_chunk_size>(global),
                    noc.get_noc_id());
            };
            const uint32_t batch = std::min(chunks_per_cb_entry, slice_chunks - chunks_sent);
            for (uint32_t left = batch; left > 0;) {
                const auto pos = map.at(send_walk.chunk());
                const uint32_t page = page_of<output_chunks_per_page>(pos.global);
                const uint32_t off = byte_off_of<output_chunks_per_page, output_chunk_size>(pos.global);
                const uint32_t run = next_run<output_chunks_per_page>(
                    send_walk, output_tensor_accessor, plan.out, pos.global, pos.row_end, left);
                ASSERT(run_is_linear(send_walk, run, output_chunk_size, run_addr(send_walk.chunk()), run_addr));
                fabric.queue_segment(
                    l1_read_addr,
                    tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page, off),
                    run * output_chunk_size);
                l1_read_addr += run * output_chunk_size;
                left -= run;
                send_walk.advance(run);
            }
            chunks_sent += batch;
            fabric.flush_packet_and_wait();
            if (l1_read_addr == l1_end_addr) {
                l1_read_addr = l1_base_addr;
            }
        }
    };

    // We reserve two to kick start the pipeline, and then it is steady state
    cb.reserve_back(2);
    while (chunks_read < slice_chunks) {
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
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
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
