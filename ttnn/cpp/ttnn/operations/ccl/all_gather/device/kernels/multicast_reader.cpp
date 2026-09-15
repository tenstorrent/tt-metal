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

#include "chunk_walk.hpp"
#include "concat.hpp"
#include "multicast_common.hpp"

using address_t = uint32_t;

// Multicast reader: fills the CB from our own input and sends it on the forward directions (E-line +
// S-rect) itself, one entry behind the reads. Also owns both barriers.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////

    // --- moving chunks ---
    constexpr uint32_t chunk_size = get_compile_time_arg_val(0);
    constexpr uint32_t in_chunks_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t out_chunks_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t payload = get_compile_time_arg_val(3);        // bytes a packet may carry
    constexpr uint32_t asked_run_max = get_compile_time_arg_val(4);  // chunks; 0 = the whole payload
    constexpr uint32_t entry_chunks = get_compile_time_arg_val(5);
    // --- all_gather ---
    constexpr uint32_t stripe = get_compile_time_arg_val(6);
    constexpr uint32_t num_devices = get_compile_time_arg_val(7);
    // --- this kernel ---
    constexpr uint32_t cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_depth = get_compile_time_arg_val(9);
    constexpr bool alternate_routes = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(11);
    constexpr bool do_init_barrier = get_compile_time_arg_val(12) != 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<13>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr bool enable_fabric = (num_connections > 0);
    constexpr uint32_t entry_bytes = entry_chunks * chunk_size;
    constexpr uint32_t payload_chunks = payload / chunk_size > 0 ? payload / chunk_size : 1;
    constexpr uint32_t run_max_want = run_max_capped(asked_run_max, payload_chunks, chunk_size);

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
    CircularBuffer cb(cb_id);

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

    // Allocate headers and set state for data sends
    MulticastSender<alternate_routes> fabric(fabric_connection, num_connections, ranges, ranges_alt);
    Packer<chunk_size, payload, MulticastSender<alternate_routes>> packer(noc, fabric);

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
    // SETUP
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
    auto l1_end_addr = l1_base_addr + (cb_depth * entry_bytes);
    auto l1_write_addr = l1_base_addr;
    auto l1_read_addr = l1_base_addr;

    // The walk order follows the output. Hop 0 reads the input in that same order, and the input yields
    // runs only where it happens to stride the same way.
    const bool out_packed = packed_pages(output_tensor_accessor, out_chunks_per_page, chunk_size);
    const uint32_t bank_step = bank_step_of(output_tensor_accessor, out_chunks_per_page);
    const uint32_t run_max = out_packed ? run_max_want : 1u;
    const bool out_in_page = join_in_page(out_packed, out_chunks_per_page, bank_step);
    const bool out_across_pages =
        join_pages(out_packed, out_chunks_per_page, output_tensor_accessor.contiguous_page_stride(), bank_step);

    const bool in_packed = packed_pages(input_tensor_accessor, in_chunks_per_page, chunk_size);
    const bool in_in_page = join_in_page(in_packed, in_chunks_per_page, bank_step);
    const bool in_across_pages =
        join_pages(in_packed, in_chunks_per_page, input_tensor_accessor.contiguous_page_stride(), bank_step);

    const uint32_t input_end_chunk = slice_first_chunk + slice_chunks;
    const uint32_t stripe_base = device_idx * stripe;

    // Same walk, one CB entry apart: reads stay in flight while the entry before them is sent.
    Walk read_walk;
    read_walk.init(slice_first_chunk, slice_chunks, 0, bank_step, run_max);
    Walk send_walk = read_walk;
    uint32_t chunks_read = 0;
    uint32_t chunks_sent = 0;

    auto out_addr = [&](uint32_t out) {
        return output_tensor_accessor.get_noc_addr(
            page_of<out_chunks_per_page>(out), byte_off<out_chunks_per_page, chunk_size>(out), noc.get_noc_id());
    };
    // Address of one of our chunks. Named, not inline: an ASSERT argument is unevaluated, and a
    // lambda cannot appear there.
    auto run_addr = [&](uint32_t ours) { return out_addr(out_chunk<stripe, num_devices>(ours, stripe_base)); };

    auto in_addr = [&](uint32_t chunk) {
        return input_tensor_accessor.get_noc_addr(
            page_of<in_chunks_per_page>(chunk), byte_off<in_chunks_per_page, chunk_size>(chunk), noc.get_noc_id());
    };

    auto read_run = [&](uint64_t src, uint32_t chunks) __attribute__((always_inline)) {
        if constexpr (chunk_fits_command(chunk_size)) {
            noc.async_read<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
                tensor_accessor::Page(src, 0),
                CoreLocalMem<uint32_t>(l1_write_addr),
                chunks * chunk_size,
                {},
                {},
                {.trid = curr_trid});
        } else {
            noc.async_read<NocOptions::TXN_ID>(
                tensor_accessor::Page(src, 0),
                CoreLocalMem<uint32_t>(l1_write_addr),
                chunk_size,
                {},
                {},
                {.trid = curr_trid});
        }
    };

    // Read input tensor and fill one CB entry
    auto fill_entry = [&]() __attribute__((always_inline)) {
        const uint32_t entry = std::min(entry_chunks, slice_chunks - chunks_read);
        for (uint32_t left = entry; left > 0;) {
            const uint32_t ours = read_walk.chunk();
            const uint32_t run = run_length<in_chunks_per_page>(
                input_tensor_accessor,
                in_in_page,
                in_across_pages,
                ours,
                input_end_chunk,
                std::min(left, read_walk.lane_room()));
            const uint64_t src = in_addr(ours);
            ASSERT(run_is_linear(read_walk, run, chunk_size, src, in_addr));
            read_run(src, run);
            l1_write_addr += run * chunk_size;
            left -= run;
            read_walk.advance(run);
        }
        chunks_read += entry;
        if (l1_write_addr == l1_end_addr) {
            l1_write_addr = l1_base_addr;
        }
    };

    // Send one CB entry on our routes
    auto send_entry = [&]() __attribute__((always_inline)) {
        if constexpr (enable_fabric) {
            const uint32_t entry = std::min(entry_chunks, slice_chunks - chunks_sent);
            for (uint32_t left = entry; left > 0;) {
                const uint32_t ours = send_walk.chunk();
                const uint32_t out = out_chunk<stripe, num_devices>(ours, stripe_base);  // where it lands
                const uint32_t room = row_room<stripe>(ours);                            // a run stops at the row edge
                const uint32_t run = run_length<out_chunks_per_page>(
                    output_tensor_accessor,
                    out_in_page,
                    out_across_pages,
                    out,
                    out + room,
                    std::min(left, send_walk.lane_room()));
                ASSERT(run_is_linear(send_walk, run, chunk_size, run_addr(ours), run_addr));
                packer.add_run(
                    l1_read_addr,
                    tt::tt_fabric::addrgen_detail::get_noc_address(
                        output_tensor_accessor,
                        page_of<out_chunks_per_page>(out),
                        byte_off<out_chunks_per_page, chunk_size>(out)),
                    run * chunk_size);
                l1_read_addr += run * chunk_size;
                left -= run;
                send_walk.advance(run);
            }
            chunks_sent += entry;
            packer.flush();
            if (l1_read_addr == l1_end_addr) {
                l1_read_addr = l1_base_addr;
            }
        }
    };

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

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
