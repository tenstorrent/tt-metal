// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>
#include <array>
#include <utility>

#include "multicast_common.hpp"

using address_t = uint32_t;

void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(3);
    constexpr uint32_t output_page_stripe_jump = get_compile_time_arg_val(4);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_depth = get_compile_time_arg_val(6);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t packet_size = get_compile_time_arg_val(8);
    constexpr bool load_balance_across_alt_routes = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(10);
    constexpr bool do_init_barrier = get_compile_time_arg_val(11) != 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<12>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr bool enable_fabric = (num_connections > 0);
    constexpr uint32_t output_page_size = output_chunks_per_page * output_chunk_size;
    constexpr uint32_t inputs_per_cb_page = cb_page_size / input_page_size;
    constexpr uint32_t outputs_per_cb_page = cb_page_size / output_chunk_size;

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_chunk_in_stripe_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_byte_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_byte_offset_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_output_chunks = get_arg_val<uint32_t>(arg_idx++);
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
    size_t arg_for_fab = arg_idx;

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    DataflowBuffer cb(cb0_id);

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    if constexpr (enable_fabric) {
        open_connections(fabric_connection, num_connections, arg_for_fab);
    }

    // Build ranges + ranges_alt arrays.
    // Connection order matches host: line (E) first, then rect (S) — only active ones are present,
    // indexed 0..num_connections-1.
    FabricRange ranges[2] = {};      // [0] = E-line, [1] = S-rect (Fabric_2D only)
    FabricRange ranges_alt[2] = {};  // [0] = E-line, [1] = S-rect (Fabric_2D only)
#ifdef FABRIC_2D
    {
        uint32_t idx = 0;
        if (line_hops > 0) {
            ranges[idx] = FabricRange{line_hops, 0, 0, 0};
            ranges_alt[idx] = FabricRange{line_hops_alt, 0, 0, 0};
            ++idx;
        }
        if (rect_spine_hops > 0) {
            ranges[idx] = FabricRange{rect_e_hops, rect_w_hops, 0, rect_spine_hops};
            ranges_alt[idx] = FabricRange{rect_e_hops_alt, rect_w_hops_alt, 0, rect_spine_hops_alt};
            ++idx;
        }
    }
#else
    // 1D: exactly one of (line_hops, rect_spine_hops) is nonzero — that's the active axis.
    ranges[0] = (line_hops != 0) ? line_hops : rect_spine_hops;
    ranges_alt[0] = (line_hops != 0) ? line_hops_alt : rect_spine_hops_alt;
#endif

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
        // Atomic decrement (add -value), not reset to 0, so any increments from other phases are preserved.
        noc_semaphore_inc(
            safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem),
            (uint32_t)(-(int32_t)barrier_wait_value));
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

    // "iterator" for input_tensor
    auto input_page_id = input_page_id_start;
    auto valid_input_page_id = [&]() __attribute__((always_inline)) { return input_page_id < input_page_id_end; };
    auto next_input_page_id = [&]() __attribute__((always_inline)) { return input_page_id++; };

    // "iterator" for output_tensor:
    //   byte_offset++ within an output page -> chunk++ -> stripe+=jump
    // (see the "Page indexing" glossary in all_gather_multicast_factory.cpp)
    // Returns {output_page_id, byte_offset} for the current chunk, then advances.
    // Supports any gather dim, any N-D shape, any shard spec.
    uint32_t output_page_id = output_page_id_start;
    uint32_t output_page_byte_off = output_page_byte_offset_start;
    uint32_t output_chunks_sent = 0;
    uint32_t output_chunk_in_stripe = output_chunk_in_stripe_start;
    auto valid_output_chunk = [&]() __attribute__((always_inline)) { return output_chunks_sent < num_output_chunks; };
    auto next_output_chunk = [&]() __attribute__((always_inline)) {
        std::pair<uint32_t, uint32_t> loc{output_page_id, output_page_byte_off};
        output_chunks_sent++;
        if (++output_chunk_in_stripe == output_chunks_per_stripe) {
            output_chunk_in_stripe = 0;
            output_page_id += output_page_stripe_jump;
            output_page_byte_off = output_page_byte_offset;
        } else {
            output_page_byte_off += output_chunk_size;
            if (output_page_byte_off == output_page_size) {
                output_page_byte_off = 0;
                output_page_id++;
            }
        }
        return loc;
    };

    // We reserve two to kick start the pipeline, and then it is steady state
    cb.reserve_back(2);
    while (valid_input_page_id()) {
        // Read input tensor and fill CB page
        for (uint32_t i = 0; i < inputs_per_cb_page && valid_input_page_id(); ++i) {
            auto page_id = next_input_page_id();
            noc.async_read<NocOptions::TXN_ID>(
                input_tensor_accessor,
                CoreLocalMem<uint32_t>(l1_write_addr),
                input_page_size,
                {.page_id = page_id},
                {},
                {.trid = curr_trid});
            l1_write_addr += input_page_size;
        }
        if (l1_write_addr == l1_end_addr) {
            l1_write_addr = l1_base_addr;
        }

        curr_trid = (curr_trid == max_trid) ? 1 : curr_trid + 1;
        if (txns_in_flight) {
            // push_back() will unblock the writer to send Fabric data in opposite dir
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = wait_trid});
            cb.push_back(1);
            wait_trid = (wait_trid == max_trid) ? 1 : (wait_trid + 1);

            if constexpr (enable_fabric) {
                // Send Fabric data in our dir
                for (uint32_t i = 0; i < outputs_per_cb_page && valid_output_chunk(); ++i) {
                    auto [page_id, page_byte_offset] = next_output_chunk();
                    auto fabric_tensor_page_addr = tt::tt_fabric::addrgen_detail::get_noc_address(
                        output_tensor_accessor, page_id, page_byte_offset);
                    fabric.async_write(l1_read_addr, fabric_tensor_page_addr);
                    l1_read_addr += output_chunk_size;
                }
                fabric.async_writes_flushed();
                if (l1_read_addr == l1_end_addr) {
                    l1_read_addr = l1_base_addr;
                }
            }

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

        if constexpr (enable_fabric) {
            // Send Fabric data in our dir
            for (uint32_t i = 0; i < outputs_per_cb_page && valid_output_chunk(); ++i) {
                auto [page_id, page_byte_offset] = next_output_chunk();
                auto fabric_tensor_page_addr =
                    tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page_id, page_byte_offset);
                fabric.async_write(l1_read_addr, fabric_tensor_page_addr);
                l1_read_addr += output_chunk_size;
            }
            fabric.async_writes_flushed();
            if (l1_read_addr == l1_end_addr) {
                l1_read_addr = l1_base_addr;
            }
        }
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
    // Atomic decrement (add -value), not reset to 0, so any increments from other phases are preserved.
    noc_semaphore_inc(
        safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem),
        (uint32_t)(-(int32_t)barrier_wait_value));

    if constexpr (enable_fabric) {
        close_connections(fabric_connection);
    }
    noc.async_write_barrier();
}
