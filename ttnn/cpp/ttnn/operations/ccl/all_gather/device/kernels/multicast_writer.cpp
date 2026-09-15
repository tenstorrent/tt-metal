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

// Multicast writer: CB consumer, covers the backward directions (W-line + N-rect). Sends our stripe to
// every device on those routes, and writes it into our own output on the way.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////

    // --- moving chunks ---
    constexpr uint32_t chunk_size = get_compile_time_arg_val(0);
    constexpr uint32_t out_chunks_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t payload = get_compile_time_arg_val(2);        // bytes a packet may carry
    constexpr uint32_t asked_run_max = get_compile_time_arg_val(3);  // chunks; 0 = the whole payload
    constexpr uint32_t entry_chunks = get_compile_time_arg_val(4);
    // --- all_gather ---
    constexpr uint32_t stripe = get_compile_time_arg_val(5);
    constexpr uint32_t num_devices = get_compile_time_arg_val(6);
    // --- this kernel ---
    constexpr uint32_t cb_id = get_compile_time_arg_val(7);
    constexpr bool alternate_routes = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(9);
    constexpr bool do_init_barrier = get_compile_time_arg_val(10) != 0;
    constexpr auto output_tensor_args = TensorAccessorArgs<11>();

    constexpr bool enable_fabric = (num_connections > 0);
    constexpr uint32_t payload_chunks = payload / chunk_size > 0 ? payload / chunk_size : 1;
    constexpr uint32_t run_max_want = run_max_capped(asked_run_max, payload_chunks, chunk_size);
    // A run is emitted as one scatter segment starting at its source offset within the packet, so every
    // chunk size has to keep source and destination NoC-write aligned.
    static_assert(chunk_size % 16 == 0, "chunk size must be a multiple of the NoC write alignment");

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
    if constexpr (do_init_barrier && enable_fabric) {
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
    }

    ///////////////////////////////////////////////////
    // SETUP
    ///////////////////////////////////////////////////

    const bool packed = packed_pages(output_tensor_accessor, out_chunks_per_page, chunk_size);
    const uint32_t bank_step = bank_step_of(output_tensor_accessor, out_chunks_per_page);
    const bool out_in_page = join_in_page(packed, out_chunks_per_page, bank_step);
    const bool out_across_pages =
        join_pages(packed, out_chunks_per_page, output_tensor_accessor.contiguous_page_stride(), bank_step);
    const uint32_t run_max = packed ? run_max_want : 1u;
    const uint32_t stripe_base = device_idx * stripe;

    Walk walk;
    walk.init(slice_first_chunk, slice_chunks, 0, bank_step, run_max);

    auto out_addr = [&](uint32_t out) {
        return output_tensor_accessor.get_noc_addr(
            page_of<out_chunks_per_page>(out), byte_off<out_chunks_per_page, chunk_size>(out), noc.get_noc_id());
    };
    // Address of one of our chunks. Named, not inline: an ASSERT argument is unevaluated, and a
    // lambda cannot appear there.
    auto run_addr = [&](uint32_t ours) { return out_addr(out_chunk<stripe, num_devices>(ours, stripe_base)); };

    auto local_write = [&](uint32_t l1_read_addr, uint64_t dst, uint32_t chunks) {
        // Posted write on a separate VC so it doesn't contend with the fabric writes on the same NOC.
        if constexpr (chunk_fits_command(chunk_size)) {
            noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                CoreLocalMem<uint32_t>(l1_read_addr),
                tensor_accessor::Page(dst, 0),
                chunks * chunk_size,
                {},
                {},
                {.vc = NOC_UNICAST_WRITE_VC + 1});
        } else {
            noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
                CoreLocalMem<uint32_t>(l1_read_addr),
                tensor_accessor::Page(dst, 0),
                chunk_size,
                {},
                {},
                {.vc = NOC_UNICAST_WRITE_VC + 1});
        }
    };

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    for (uint32_t chunks_sent = 0; chunks_sent < slice_chunks;) {
        const uint32_t entry = std::min(entry_chunks, slice_chunks - chunks_sent);
        cb.wait_front(1);
        uint32_t l1_read_addr = cb.get_read_ptr();

        for (uint32_t left = entry; left > 0;) {
            const uint32_t ours = walk.chunk();
            const uint32_t out = out_chunk<stripe, num_devices>(ours, stripe_base);  // all_gather: where it lands
            const uint32_t room = row_room<stripe>(ours);  // all_gather: a run stops at the row edge
            const uint32_t run = run_length<out_chunks_per_page>(
                output_tensor_accessor,
                out_in_page,
                out_across_pages,
                out,
                out + room,
                std::min(left, walk.lane_room()));
            const uint64_t dst = out_addr(out);
            ASSERT(run_is_linear(walk, run, chunk_size, dst, run_addr));
            if constexpr (enable_fabric) {
                packer.add_run(
                    l1_read_addr,
                    tt::tt_fabric::addrgen_detail::get_noc_address(
                        output_tensor_accessor,
                        page_of<out_chunks_per_page>(out),
                        byte_off<out_chunks_per_page, chunk_size>(out)),
                    run * chunk_size);
            }
            local_write(l1_read_addr, dst, run);
            l1_read_addr += run * chunk_size;
            left -= run;
            walk.advance(run);
        }

        noc.async_writes_flushed<NocOptions::POSTED>();  // wait for local writes
        if constexpr (enable_fabric) {
            packer.flush();  // wait for Fabric writes
        }
        cb.pop_front(1);
        chunks_sent += entry;
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
