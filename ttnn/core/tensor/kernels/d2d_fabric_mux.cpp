// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Lease-aware FabricMux kernel for D2DStreamService.
// Keeps the exact inner forwarding loop of tt_fabric_mux.cpp for full wire throughput,
// and checks lease connect/disconnect boundaries once per outer burst.

// clang-format off
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"

#include <array>
#include <cstddef>
// clang-format on

constexpr size_t NUM_FULL_SIZE_CHANNELS = get_compile_time_arg_val(0);
constexpr uint8_t NUM_BUFFERS_FULL_SIZE_CHANNEL = get_compile_time_arg_val(1);
constexpr size_t BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL = get_compile_time_arg_val(2);
constexpr size_t NUM_HEADER_ONLY_CHANNELS = get_compile_time_arg_val(3);
constexpr uint8_t NUM_BUFFERS_HEADER_ONLY_CHANNEL = get_compile_time_arg_val(4);

constexpr size_t status_address = get_compile_time_arg_val(5);
constexpr size_t termination_signal_address = get_compile_time_arg_val(6);
constexpr size_t connection_info_base_address = get_compile_time_arg_val(7);
constexpr size_t connection_handshake_base_address = get_compile_time_arg_val(8);
constexpr size_t sender_flow_control_base_address = get_compile_time_arg_val(9);
constexpr size_t channels_base_l1_address = get_compile_time_arg_val(10);
constexpr size_t local_fabric_router_status_address = get_compile_time_arg_val(11);
constexpr size_t fabric_router_status_address = get_compile_time_arg_val(12);

constexpr uint8_t NUM_EDM_BUFFERS = get_compile_time_arg_val(13);
constexpr size_t NUM_FULL_SIZE_CHANNELS_ITERS = get_compile_time_arg_val(14);
constexpr size_t NUM_ITERS_BETWEEN_TEARDOWN_CHECKS = get_compile_time_arg_val(15);

constexpr ProgrammableCoreType CORE_TYPE = static_cast<ProgrammableCoreType>(get_compile_time_arg_val(16));
constexpr bool wait_for_fabric_endpoint = get_compile_time_arg_val(17) == 1;
constexpr bool in_program_local_l1 = get_compile_time_arg_val(18) == 1;
constexpr uint32_t program_local_l1_cb_index = get_compile_time_arg_val(19);
constexpr uint32_t l1_base_semaphore_id = get_compile_time_arg_val(20);
constexpr size_t CHANNEL_STREAM_IDS_START_IDX = 21;

constexpr bool lease_upstream_edm = get_named_compile_time_arg_val("lease_upstream_edm") == 1;

constexpr size_t NOC_ALIGN_PADDING_BYTES = 12;
constexpr bool ENABLE_RISC_CPU_DATA_CACHE = true;

namespace tt::tt_fabric {
using FabricMuxToEdmSender = WorkerToFabricEdmSenderImpl<false, NUM_EDM_BUFFERS>;
}  // namespace tt::tt_fabric

template <uint8_t NUM_BUFFERS>
void setup_channel(
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>* channel_ptr,
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>* worker_interface_ptr,
    bool& channel_connection_established,
    size_t buffer_size_bytes,
    size_t& channel_base_address,
    size_t& connection_info_address,
    size_t& connection_handshake_address,
    size_t& sender_flow_control_address,
    StreamId my_channel_free_slots_stream_id) {
    new (channel_ptr) tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>(
        channel_base_address, buffer_size_bytes, sizeof(PACKET_HEADER_TYPE));
    channel_base_address += NUM_BUFFERS * buffer_size_bytes;
    init_ptr_val(my_channel_free_slots_stream_id, NUM_BUFFERS);

    auto connection_worker_info_ptr =
        reinterpret_cast<volatile tt::tt_fabric::FabricMuxChannelClientLocationInfo*>(connection_info_address);
    connection_info_address += sizeof(tt::tt_fabric::FabricMuxChannelClientLocationInfo);

    new (worker_interface_ptr) tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>(
        connection_worker_info_ptr,
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(sender_flow_control_address),
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_handshake_address),
        0 /* unused */,
        tt::tt_fabric::MUX_TO_WORKER_INTERFACE_STARTING_READ_COUNTER_VALUE);
    sender_flow_control_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
    connection_handshake_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;

    channel_connection_established = false;
}

template <uint8_t NUM_BUFFERS>
void forward_data_hot(
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>& channel,
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
    tt::tt_fabric::FabricMuxToEdmSender& fabric_connection,
    bool& edm_open,
    bool& channel_connection_established,
    StreamId my_channel_free_slots_stream_id,
    [[maybe_unused]] uint8_t channel_id) {
    constexpr bool use_worker_credit_addr = CORE_TYPE == ProgrammableCoreType::IDLE_ETH;
    const bool has_unsent_payload = get_ptr_val(my_channel_free_slots_stream_id.get()) != NUM_BUFFERS;
    if (has_unsent_payload) {
        if constexpr (lease_upstream_edm) {
            if (!edm_open) {
                fabric_connection.open<use_worker_credit_addr>();
                edm_open = true;
            }
        }
        if (!channel_connection_established) {
            invalidate_l1_cache();
            uint32_t cached = *worker_interface.connection_live_semaphore;
            if (tt::tt_fabric::connect_is_requested(cached)) {
                channel_connection_established = true;
                tt::tt_fabric::
                    establish_worker_connection<tt::tt_fabric::USE_DYNAMIC_CREDIT_ADDR, ENABLE_RISC_CPU_DATA_CACHE>(
                        worker_interface);
            }
        }

        size_t buffer_address = channel.get_buffer_address(worker_interface.local_write_counter.get_buffer_index());
        invalidate_l1_cache();
        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(buffer_address);

        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)packet_header, packet_header->get_payload_size_including_header());

        worker_interface.local_write_counter.increment();
        worker_interface.local_read_counter.increment();
        increment_local_update_ptr_val(my_channel_free_slots_stream_id.get(), 1);

        noc_async_writes_flushed();
        if (channel_connection_established) {
            worker_interface.notify_worker_of_read_counter_update();
        }
    }
}

void kernel_main() {
    set_l1_data_cache<true>();
    size_t rt_args_idx = 0;

    size_t l1_base = 0;
    if constexpr (in_program_local_l1) {
        l1_base = get_write_ptr(program_local_l1_cb_index);
    }

    auto status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + status_address);
    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::STARTED;

    auto num_regions_to_clear = get_arg_val<uint32_t>(rt_args_idx++);
    for (uint32_t i = 0; i < num_regions_to_clear; i++) {
        auto address = get_arg_val<uint32_t>(rt_args_idx++);
        auto size = get_arg_val<uint32_t>(rt_args_idx++);
        zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(l1_base + address), size);
    }

    auto fabric_connection = tt::tt_fabric::FabricMuxToEdmSender::build_from_args<CORE_TYPE>(rt_args_idx);

    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_FULL_SIZE_CHANNEL>, NUM_FULL_SIZE_CHANNELS>
        full_size_channels;
    std::array<
        tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS_FULL_SIZE_CHANNEL>,
        NUM_FULL_SIZE_CHANNELS>
        full_size_channel_worker_interfaces;
    std::array<bool, NUM_FULL_SIZE_CHANNELS> full_size_channel_connection_established;

    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_HEADER_ONLY_CHANNEL>, NUM_HEADER_ONLY_CHANNELS>
        header_only_channels;
    std::array<
        tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS_HEADER_ONLY_CHANNEL>,
        NUM_HEADER_ONLY_CHANNELS>
        header_only_channel_worker_interfaces;
    std::array<bool, NUM_HEADER_ONLY_CHANNELS> header_only_channel_connection_established;

    constexpr size_t NUM_TOTAL_CHANNELS = NUM_FULL_SIZE_CHANNELS + NUM_HEADER_ONLY_CHANNELS;
    constexpr std::array<uint32_t, NUM_TOTAL_CHANNELS> channel_stream_ids =
        fill_array_with_next_n_args<uint32_t, CHANNEL_STREAM_IDS_START_IDX, NUM_TOTAL_CHANNELS>();

    size_t channel_base_address = l1_base + channels_base_l1_address;
    size_t connection_info_address = l1_base + connection_info_base_address;
    size_t connection_handshake_address = l1_base + connection_handshake_base_address;
    size_t sender_flow_control_address = l1_base + sender_flow_control_base_address;

    for (uint8_t i = 0; i < NUM_FULL_SIZE_CHANNELS; i++) {
        setup_channel<NUM_BUFFERS_FULL_SIZE_CHANNEL>(
            &full_size_channels[i],
            &full_size_channel_worker_interfaces[i],
            full_size_channel_connection_established[i],
            BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL,
            channel_base_address,
            connection_info_address,
            connection_handshake_address,
            sender_flow_control_address,
            StreamId{channel_stream_ids[i]});
    }

    for (uint8_t i = 0; i < NUM_HEADER_ONLY_CHANNELS; i++) {
        setup_channel<NUM_BUFFERS_HEADER_ONLY_CHANNEL>(
            &header_only_channels[i],
            &header_only_channel_worker_interfaces[i],
            header_only_channel_connection_established[i],
            sizeof(PACKET_HEADER_TYPE),
            channel_base_address,
            connection_info_address,
            connection_handshake_address,
            sender_flow_control_address,
            StreamId{channel_stream_ids[i + NUM_FULL_SIZE_CHANNELS]});
    }

    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(l1_base + termination_signal_address);

    if constexpr (wait_for_fabric_endpoint) {
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_connection.edm_noc_x,
            fabric_connection.edm_noc_y,
            fabric_router_status_address,
            l1_base + local_fabric_router_status_address);
    }

    constexpr bool use_worker_allocated_credit_address = CORE_TYPE == ProgrammableCoreType::IDLE_ETH;
    bool edm_open = false;
    if constexpr (!lease_upstream_edm) {
        fabric_connection.open<use_worker_allocated_credit_address>();
        edm_open = true;
    }

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC;
    if constexpr (in_program_local_l1) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(l1_base_semaphore_id)) = l1_base;
    }

    auto channel_active_or_pending =
        [&]<uint8_t NUM_BUFFERS>(
            bool established,
            uint32_t stream_id,
            const tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& iface) -> bool {
        return established || get_ptr_val(stream_id) != NUM_BUFFERS ||
               tt::tt_fabric::connect_is_requested(*iface.connection_live_semaphore);
    };

    auto any_other_channel_active_or_pending = [&](size_t closing_global_idx) -> bool {
        invalidate_l1_cache();
        for (uint8_t c = 0; c < NUM_FULL_SIZE_CHANNELS; c++) {
            if (c != closing_global_idx && channel_active_or_pending(
                                               full_size_channel_connection_established[c],
                                               channel_stream_ids[c],
                                               full_size_channel_worker_interfaces[c])) {
                return true;
            }
        }
        for (uint8_t c = 0; c < NUM_HEADER_ONLY_CHANNELS; c++) {
            const size_t gidx = c + NUM_FULL_SIZE_CHANNELS;
            if (gidx != closing_global_idx && channel_active_or_pending(
                                                  header_only_channel_connection_established[c],
                                                  channel_stream_ids[gidx],
                                                  header_only_channel_worker_interfaces[c])) {
                return true;
            }
        }
        return false;
    };

    auto maintain_channel = [&]<uint8_t NUM_BUFFERS>(
                                size_t global_idx,
                                uint32_t stream_id,
                                bool& established,
                                tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& iface) {
        if (!established) {
            if (tt::tt_fabric::connect_is_requested(*iface.connection_live_semaphore)) {
                if constexpr (lease_upstream_edm) {
                    if (!edm_open) {
                        fabric_connection.open<use_worker_allocated_credit_address>();
                        edm_open = true;
                    }
                }
                established = true;
                tt::tt_fabric::
                    establish_worker_connection<tt::tt_fabric::USE_DYNAMIC_CREDIT_ADDR, ENABLE_RISC_CPU_DATA_CACHE>(
                        iface);
            }
        } else if (get_ptr_val(stream_id) == NUM_BUFFERS && iface.has_worker_teardown_request()) {
            established = false;
            if constexpr (lease_upstream_edm) {
                if (edm_open && !any_other_channel_active_or_pending(global_idx)) {
                    noc_async_full_barrier();
                    fabric_connection.close();
                    edm_open = false;
                }
            }
            iface.template teardown_worker_connection<true, ENABLE_RISC_CPU_DATA_CACHE>();
        }
    };

    while (!got_immediate_termination_signal<true>(termination_signal_ptr)) {
        if (got_graceful_termination_signal(termination_signal_ptr)) {
            bool all_channels_drained = true;
            for (uint8_t c = 0; c < NUM_FULL_SIZE_CHANNELS; c++) {
                all_channels_drained &= get_ptr_val(channel_stream_ids[c]) == NUM_BUFFERS_FULL_SIZE_CHANNEL;
            }
            for (uint8_t c = 0; c < NUM_HEADER_ONLY_CHANNELS; c++) {
                all_channels_drained &=
                    get_ptr_val(channel_stream_ids[c + NUM_FULL_SIZE_CHANNELS]) == NUM_BUFFERS_HEADER_ONLY_CHANNEL;
            }
            if (all_channels_drained) {
                break;
            }
        }

        // Tight packet-forwarding burst (zero L1 cache flushes on idle channels).
        for (size_t i = 0; i < NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
            for (size_t iter = 0; iter < NUM_FULL_SIZE_CHANNELS_ITERS; iter++) {
                for (uint8_t c = 0; c < NUM_FULL_SIZE_CHANNELS; c++) {
                    forward_data_hot<NUM_BUFFERS_FULL_SIZE_CHANNEL>(
                        full_size_channels[c],
                        full_size_channel_worker_interfaces[c],
                        fabric_connection,
                        edm_open,
                        full_size_channel_connection_established[c],
                        StreamId{channel_stream_ids[c]},
                        c);
                }
            }

            for (uint8_t c = 0; c < NUM_HEADER_ONLY_CHANNELS; c++) {
                const uint32_t sid = channel_stream_ids[c + NUM_FULL_SIZE_CHANNELS];
                forward_data_hot<NUM_BUFFERS_HEADER_ONLY_CHANNEL>(
                    header_only_channels[c],
                    header_only_channel_worker_interfaces[c],
                    fabric_connection,
                    edm_open,
                    header_only_channel_connection_established[c],
                    StreamId{sid},
                    c + NUM_FULL_SIZE_CHANNELS);
            }
        }

        // Connection / teardown maintenance once per burst.
        invalidate_l1_cache();
        for (uint8_t c = 0; c < NUM_FULL_SIZE_CHANNELS; c++) {
            maintain_channel(
                c,
                channel_stream_ids[c],
                full_size_channel_connection_established[c],
                full_size_channel_worker_interfaces[c]);
        }
        for (uint8_t c = 0; c < NUM_HEADER_ONLY_CHANNELS; c++) {
            const size_t gidx = c + NUM_FULL_SIZE_CHANNELS;
            maintain_channel(
                gidx,
                channel_stream_ids[gidx],
                header_only_channel_connection_established[c],
                header_only_channel_worker_interfaces[c]);
        }
    }

    if (edm_open) {
        fabric_connection.close();
    }
    noc_async_full_barrier();
    noc_clear_packet_tags(noc_index);

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
    set_l1_data_cache<false>();
}
