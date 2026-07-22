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

#include "unicast_common.hpp"

using address_t = uint32_t;

// Store-and-forward writer: CB consumer, owns all fabric. It only ever sends (never waits on a semaphore --
// its sole backpressure is wait_front). Each iteration drains the CB and unicasts the stripe one hop to the
// neighbor's output (same address); iteration 0 also writes this device's local data into local output.
// Maintains the downstream reader's data_valid (= chunks delivered; see the note in unicast_common.hpp), and
// sends its one-shot "alive" barrier inc up front.
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
    constexpr bool do_init_barrier = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t data_valid_granularity = get_compile_time_arg_val(8);
    constexpr auto output_tensor_args = TensorAccessorArgs<9>();

#ifdef USE_WORKER_MUX
    // Fabric-mux geometry, appended by ccl::fabric_mux_connection_ct_args (after the tensor-accessor args).
    constexpr uint32_t mux_ct_base = output_tensor_args.next_compile_time_args_offset();
    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(mux_ct_base + 0);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(mux_ct_base + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(mux_ct_base + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(mux_ct_base + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(mux_ct_base + 4);
#endif

    constexpr uint32_t outputs_per_cb_page = cb_page_size / output_chunk_size;

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
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
    [[maybe_unused]] const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);  // used only if do_init_barrier
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const uint8_t barrier_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // neighbor opposite-dir core
    [[maybe_unused]] const uint8_t barrier_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t data_valid_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // mirror core (data_valid_sem target)
    const uint8_t data_valid_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_granular_sends = get_arg_val<uint32_t>(arg_idx++);  // leading sends the downstream relays
    [[maybe_unused]] size_t arg_for_fab = arg_idx;  // fabric connection args start here (non-mux path)

    // A direction with no neighbor (a line endpoint) relays nothing; no fabric/mux connection was appended.
    if (num_iters == 0) {
        return;
    }

#ifdef USE_WORKER_MUX
    // Fabric-mux connection RT args, appended by ccl::fabric_mux_connection_rt_args (17 args). Only appended
    // for an active direction, so we always have a live connection here (num_iters > 0 implies a neighbor).
    [[maybe_unused]] const bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++) != 0;
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
#endif

    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

#ifdef USE_WORKER_MUX
    // Multiple workers per direction share one fabric link through a fabric mux. Connect to our channel on the
    // mux instead of opening a direct connection to the neighbor's ERISC.
    using SenderT = tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>;
    SenderT mux_connection = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address,
        fabric_mux_connection_info_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_flow_control_address,
        fabric_mux_buffer_index_address,
        local_flow_control_address,
        local_teardown_address,
        local_buffer_index_address);
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
    tt::tt_fabric::fabric_client_connect(mux_connection);
    SenderT* sender = &mux_connection;
#else
    // Single worker per direction: connect directly to the neighbor's ERISC.
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, 1, arg_for_fab);
    using SenderT = tt::tt_fabric::WorkerToFabricEdmSender;
    SenderT* sender = &fabric_connection.get(0).sender;
#endif

    FabricWriter<output_chunk_size, packet_size, SenderT> fabric(noc, sender);

    // One 1-hop atomic-inc header for both the "alive" barrier inc and the data_valid signals; destination and
    // value (chunks) are set per send. Flush keeps a data_valid inc ordered after the payload it announces.
    auto sem_packet_header = PacketHeaderPool::allocate_header(1);
    fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        sem_packet_header, /*num_hops=*/1, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});
    auto atomic_inc = [&](uint64_t addr, uint32_t val) {
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
            sender, sem_packet_header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{addr, val});
    };

    // Init handshake (send only): tell the neighbor's opposite-direction reader we're alive, so it lets its
    // paired writer start writing into our output. Our own reader does the matching wait.
    if constexpr (do_init_barrier) {
        atomic_inc(safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem, 0), 1);
    }

    const uint64_t downstream_data_valid_addr =
        safe_get_noc_addr(data_valid_sem_noc_x, data_valid_sem_noc_y, data_valid_sem, 0);
    auto signal = [&](uint32_t chunks) { atomic_inc(downstream_data_valid_addr, chunks); };

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    OutputStripeIterator<output_chunks_per_stripe, output_chunks_per_page, output_chunk_size, num_devices> it;

    uint32_t stripe = initial_stripe;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const bool last = (iter == num_iters - 1);
        const uint32_t start = last ? final_start : slice_start;
        const uint32_t count = last ? final_count : slice_count;
        const bool granular = (iter < num_granular_sends);  // downstream relays this stripe -> signal fine-grained
        const bool local_copy = (iter == 0) && (do_local_write != 0);
        it.init(stripe, start, count);

        uint32_t pending_chunks = 0, pending_pages = 0;
        for (uint32_t chunks_sent = 0; chunks_sent < count;) {
            const uint32_t batch = std::min(outputs_per_cb_page, count - chunks_sent);
            cb.wait_front(1);
            uint32_t l1_read_addr = cb.get_read_ptr();
            for (uint32_t i = 0; i < batch; ++i) {
                auto [page_id, byte_off] = it.next();
                uint64_t neighbor_addr =
                    tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page_id, byte_off);
                fabric.async_write(l1_read_addr, neighbor_addr);

                if (local_copy) {
                    // Local data -> our output stripe (same address). Posted write on a separate VC so it
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

            pending_chunks += batch;
            if (granular && ++pending_pages == data_valid_granularity) {
                signal(pending_chunks);
                pending_chunks = 0;
                pending_pages = 0;
            }
            chunks_sent += batch;
        }
        // Trailing chunks of a relayed stripe, or the whole of a sink stripe (granular == false).
        if (pending_chunks > 0) {
            signal(pending_chunks);
        }
        stripe = (stripe + stripe_step) % num_devices;
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Commit our own NOC writes (the iter-0 local copy, plus the packet writes into the mux buffer) before
    // teardown.
    noc_async_write_barrier();
    noc_async_atomic_barrier();

#ifdef USE_WORKER_MUX
    // The mux forwards our buffered packets asynchronously, and close()/graceful-termination do NOT wait for
    // in-flight packets to drain. Spin until the mux has forwarded everything (all channel slots freed).
    // get_num_free_write_slots() invalidates the L1 cache internally.
    while (mux_connection.get_num_free_write_slots() != fabric_mux_num_buffers_per_channel) {
    }

    // Disconnect from the mux. Worker 0 (termination master) waits for every peer to disconnect, then tells the
    // mux to gracefully terminate (drain, then exit); the rest just signal the master.
    tt::tt_fabric::fabric_client_disconnect(mux_connection);
    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t master_sync_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(master_sync_addr, 1);
        noc_async_atomic_barrier();
    }
#else
    close_connections(fabric_connection);
#endif
    noc.async_write_barrier();
}
