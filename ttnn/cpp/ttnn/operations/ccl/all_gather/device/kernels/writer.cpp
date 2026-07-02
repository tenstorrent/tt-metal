// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>
#include <utility>

#include "common.hpp"

using address_t = uint32_t;

// Store-and-forward writer: CB consumer, owns all fabric. It only ever sends (never waits on a semaphore --
// its sole backpressure is wait_front). Each relay iteration drains the CB and unicasts the stripe one hop to
// the neighbor's output (same address); iteration 0 also materializes this device's own slice into local
// output. Signals downstream via data_valid after each stripe; sends its one-shot "alive" ready inc up front.
void kernel_main() {
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(2);
    constexpr uint32_t num_devices = get_compile_time_arg_val(3);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t packet_size = get_compile_time_arg_val(6);
    constexpr bool do_init_barrier = get_compile_time_arg_val(7) != 0;
    constexpr auto output_tensor_args = TensorAccessorArgs<8>();

    constexpr uint32_t outputs_per_cb_page = cb_page_size / output_chunk_size;

    size_t arg_idx = 0;
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t initial_stripe = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stripe_step = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_iters = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t do_local_write = get_arg_val<uint32_t>(arg_idx++);
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const address_t ready_sem = get_arg_val<uint32_t>(arg_idx++);  // used only if do_init_barrier
    const uint8_t own_noc_x = get_arg_val<uint32_t>(arg_idx++);  // data_valid target: neighbor's mirror core
    const uint8_t own_noc_y = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const uint8_t paired_noc_x = get_arg_val<uint32_t>(arg_idx++);  // ready target (init only)
    [[maybe_unused]] const uint8_t paired_noc_y = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    // A direction with no neighbor (a line endpoint) relays nothing; no fabric connection was appended.
    if (num_iters == 0) {
        return;
    }

    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, 1, arg_for_fab);
    FabricWriter<output_chunk_size, packet_size> fabric(noc, fabric_connection);

    // One 1-hop atomic-inc route reused for both the "alive" ready inc and the per-stripe data_valid signal.
    uint8_t sem_route_id = PacketHeaderPool::allocate_header_n(1);
    uint8_t num_hops[1] = {1};
    fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection, sem_route_id, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});

    // Init handshake (send only): tell the neighbor's opposite-direction reader we're alive, so it lets its
    // paired writer start writing into our output. Our own reader does the matching wait.
    if constexpr (do_init_barrier) {
        uint64_t paired_ready_addr = safe_get_noc_addr(paired_noc_x, paired_noc_y, ready_sem, 0);
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection, sem_route_id, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{paired_ready_addr, 0});
    }

    const uint64_t downstream_data_valid_addr = safe_get_noc_addr(own_noc_x, own_noc_y, data_valid_sem, 0);

    OutputStripeIterator<output_chunks_per_stripe, output_chunks_per_page, output_chunk_size, num_devices> it;

    uint32_t stripe = initial_stripe;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const bool last = (iter == num_iters - 1);
        it.init(stripe, last ? final_start : slice_start, last ? final_count : slice_count);
        const bool local_copy = (iter == 0) && (do_local_write != 0);

        while (it.valid()) {
            cb.wait_front(1);
            uint32_t l1_read_addr = cb.get_read_ptr();
            for (uint32_t i = 0; i < outputs_per_cb_page && it.valid(); ++i) {
                auto [page_id, byte_off] = it.next();
                uint64_t neighbor_addr =
                    tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page_id, byte_off);
                fabric.async_write(l1_read_addr, neighbor_addr);
                if (local_copy) {
                    // Own shard -> own output stripe (same address). Posted write on a separate VC so it
                    // doesn't contend with the fabric writes on the same NOC.
                    noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
                        CoreLocalMem<uint32_t>(l1_read_addr),
                        output_tensor_accessor,
                        output_chunk_size,
                        {},
                        {.page_id = page_id, .offset_bytes = byte_off},
                        {.vc = NOC_UNICAST_WRITE_VC + 1});
                }
                l1_read_addr += output_chunk_size;
            }
            if (local_copy) {
                noc.async_writes_flushed<NocOptions::POSTED>();
            }
            fabric.async_writes_flushed();
            cb.pop_front(1);
        }

        // Tell the downstream neighbor this stripe has landed. Ordered after the data on this connection, so
        // its reader can safely read the stripe once it observes the increment.
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{downstream_data_valid_addr, 0});

        stripe = (stripe + stripe_step) % num_devices;
    }

    close_connections(fabric_connection);
    noc.async_write_barrier();
}
