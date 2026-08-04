// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

#include "unicast_common.hpp"

using address_t = uint32_t;

// Store-and-forward writer: CB consumer, owns all fabric. It only ever sends (never waits on a semaphore --
// its sole backpressure is wait_front). Each iteration drains the CB and unicasts the stripe one hop to the
// neighbor's output (same address); iteration 0 also writes this device's local data into local output.
// Maintains the downstream reader's data_valid (= chunks delivered; see the note in unicast_common.hpp), and
// signals downstream readiness only after the corresponding payload is visible.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t packet_size = get_compile_time_arg_val(5);
    constexpr uint32_t slice_step = get_compile_time_arg_val(6);
    // See unicast_reader.cpp: the maximum per-rank output slot remains fixed for
    // runtime-controlled gathers, so its stripe width can be baked.
    constexpr uint32_t static_output_chunks_per_stripe = get_compile_time_arg_val(7);
    constexpr bool linearized_mesh_ring = get_compile_time_arg_val(8) != 0;
    constexpr auto snake_orientation =
        static_cast<ttnn::operations::experimental::high_bw_all_gather::snake_ring::Orientation>(
            get_compile_time_arg_val(9));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(10);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(11);
    constexpr auto output_tensor_args = TensorAccessorArgs<12>();

    // The direct-EDM/mux path and mux geometry follow the tensor-accessor arguments.
    constexpr uint32_t mux_ct_base = output_tensor_args.next_compile_time_args_offset();
    constexpr bool use_worker_mux = get_compile_time_arg_val(mux_ct_base + 0) != 0;
    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(mux_ct_base + 1);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(mux_ct_base + 2);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(mux_ct_base + 3);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(mux_ct_base + 4);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(mux_ct_base + 5);

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
    const address_t ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t ready_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // neighbor opposite-direction core
    const uint8_t ready_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t data_valid_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // mirror core (data_valid_sem target)
    const uint8_t data_valid_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_granular_sends = get_arg_val<uint32_t>(arg_idx++);  // leading sends the downstream relays
    const uint32_t data_valid_granularity = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const uint8_t neighbor_dev_id = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const uint16_t neighbor_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_chunks_per_stripe = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] size_t arg_for_fab = arg_idx;  // fabric connection args start here (non-mux path)

    // A direction with no neighbor (a line endpoint) relays nothing; no fabric/mux connection was appended.
    if (num_iters == 0) {
        return;
    }

    // Fabric-mux connection RT args are present only for the mux constexpr branch.
    [[maybe_unused]] bool mux_connection_valid = false;
    bool is_termination_master = false;
    uint8_t fabric_mux_x = 0, fabric_mux_y = 0, fabric_mux_channel_id = 0;
    size_t fabric_mux_channel_base_address = 0, fabric_mux_connection_info_address = 0;
    size_t fabric_mux_connection_handshake_address = 0, fabric_mux_flow_control_address = 0;
    size_t fabric_mux_buffer_index_address = 0;
    uint32_t termination_sync_semaphore_id = 0;
    uint32_t local_fabric_mux_status_address = 0, local_flow_control_address = 0;
    uint32_t local_teardown_address = 0, local_buffer_index_address = 0;
    uint8_t termination_master_noc_x = 0, termination_master_noc_y = 0;
    if constexpr (use_worker_mux) {
        mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
        is_termination_master = get_arg_val<uint32_t>(arg_idx++) != 0;
        fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
        fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
        fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
        fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
        fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
        fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
        fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
        fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
        termination_sync_semaphore_id = get_arg_val<uint32_t>(arg_idx++);
        local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
        termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
    }

    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

    auto run_writer = [&](auto* sender) {
        using SenderT = std::remove_pointer_t<decltype(sender)>;

        constexpr bool coalesce_contiguous_pages = slice_step > 1 && outputs_per_cb_page > 1;
        static_assert(!coalesce_contiguous_pages || cb_page_size <= packet_size);
        FabricWriter<output_chunk_size, packet_size, coalesce_contiguous_pages, SenderT> fabric(
            noc, sender, neighbor_dev_id, neighbor_mesh_id);

        // One 1-hop atomic-inc header for data_valid signals. Flush keeps each
        // increment ordered after the payload it announces.
        auto sem_packet_header = PacketHeaderPool::allocate_header(1);
#ifdef FABRIC_2D
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            sem_packet_header,
            neighbor_dev_id,
            neighbor_mesh_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});
#else
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            sem_packet_header, /*num_hops=*/1, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});
#endif
        auto atomic_inc = [&](uint64_t addr, uint32_t val) {
            fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
                UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
                sender, sem_packet_header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{addr, val});
        };

        // Announce that this device has reached the collective program. Its
        // earlier command-queue entries, including output initialization, are
        // complete, so the neighbor can safely write into this output. Target
        // the opposite-direction reader paired with the writer returning here.
        atomic_inc(safe_get_noc_addr(ready_sem_noc_x, ready_sem_noc_y, ready_sem, 0), 1);

        const uint64_t downstream_data_valid_addr =
            safe_get_noc_addr(data_valid_sem_noc_x, data_valid_sem_noc_y, data_valid_sem, 0);
        auto signal = [&](uint32_t chunks) { atomic_inc(downstream_data_valid_addr, chunks); };

        OutputStripeIterator<
            output_chunks_per_page,
            output_chunk_size,
            num_devices,
            slice_step,
            static_output_chunks_per_stripe,
            linearized_mesh_ring,
            snake_orientation,
            mesh_rows,
            mesh_cols>
            it;

        uint32_t stripe = initial_stripe;
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            const bool last = (iter == num_iters - 1);
            const uint32_t start = last ? final_start : slice_start;
            const uint32_t count = last ? final_count : slice_count;
            const bool granular = (iter < num_granular_sends);  // downstream relays this stripe -> signal fine-grained
            const bool local_copy = (iter == 0) && (do_local_write != 0);
            it.init(stripe, start, count, output_chunks_per_stripe);

            uint32_t pending_chunks = 0, pending_pages = 0;
            for (uint32_t chunks_sent = 0; chunks_sent < count;) {
                const uint32_t batch = std::min(outputs_per_cb_page, count - chunks_sent);
                const bool signal_after_page =
                    (granular && pending_pages + 1 == data_valid_granularity) || (chunks_sent + batch == count);
                const uint32_t chunks_to_signal = pending_chunks + batch;
                cb.wait_front(1);
                uint32_t l1_read_addr = cb.get_read_ptr();
                bool signal_fused = false;
                if constexpr (coalesce_contiguous_pages) {
                    // Bank-owned logical pages are physically contiguous. Send the complete CB batch as one Fabric
                    // packet and use the same contiguous address for the optional local output copy.
                    auto [first_page_id, first_byte_off] = it.next();
                    const uint64_t neighbor_addr = tt::tt_fabric::addrgen_detail::get_noc_address(
                        output_tensor_accessor, first_page_id, first_byte_off);
                    for (uint32_t i = 1; i < batch; ++i) {
                        (void)it.next();
                    }

                    const uint32_t batch_size = batch * output_chunk_size;
                    if (signal_after_page) {
                        fabric.async_write_contiguous_and_signal(
                            l1_read_addr, neighbor_addr, batch_size, downstream_data_valid_addr, chunks_to_signal);
                        signal_fused = true;
                    } else {
                        fabric.async_write_contiguous(l1_read_addr, neighbor_addr, batch_size);
                    }

                    if (local_copy) {
                        const uint64_t local_output_noc_addr =
                            output_tensor_accessor.get_noc_addr(first_page_id, first_byte_off, noc.get_noc_id());
                        noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
                            CoreLocalMem<uint32_t>(l1_read_addr),
                            tensor_accessor::Page(local_output_noc_addr, 0),
                            batch_size,
                            {},
                            {},
                            {.vc = NOC_UNICAST_WRITE_VC + 1});
                    }
                } else {
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
                }
                if (local_copy) {
                    noc.async_writes_flushed<NocOptions::POSTED>();
                }
                fabric.async_writes_flushed();
                cb.pop_front(1);

                if (signal_after_page) {
                    if (!signal_fused) {
                        signal(chunks_to_signal);
                    }
                    pending_chunks = 0;
                    pending_pages = 0;
                } else {
                    pending_chunks += batch;
                    if (granular) {
                        ++pending_pages;
                    }
                }
                chunks_sent += batch;
            }
            stripe = (stripe + stripe_step) % num_devices;
        }

        ///////////////////////////////////////////////////
        // CLEANUP
        ///////////////////////////////////////////////////

        // Commit our own NOC writes (the iter-0 local copy, plus the packet writes into the mux buffer) before
        // connection-specific teardown.
        noc.async_write_barrier();
        noc.async_atomic_barrier();
    };

    if constexpr (use_worker_mux) {
        // Multiple workers per direction share one fabric link through a fabric mux.
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
        run_writer(&mux_connection);

        // The mux forwards packets asynchronously; wait until all channel slots are free before disconnecting.
        while (mux_connection.get_num_free_write_slots() != fabric_mux_num_buffers_per_channel) {
        }
        tt::tt_fabric::fabric_client_disconnect(mux_connection);

        Semaphore<> termination_sync(termination_sync_semaphore_id);
        if (is_termination_master) {
            termination_sync.wait(num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            termination_sync.up(noc, termination_master_noc_x, termination_master_noc_y, 1);
            noc.async_atomic_barrier();
        }
    } else {
        // Single worker per direction connects directly to the neighbor's ERISC.
        tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
        open_connections(fabric_connection, 1, arg_for_fab);
        run_writer(&fabric_connection.get(0).sender);
        close_connections(fabric_connection);
    }
    noc.async_write_barrier();
}
