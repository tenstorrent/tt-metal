// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program-local fabric mux. Port of tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp: forward every client channel's
// packets over one fabric (EDM) connection until the clients' termination master signals it. Its compile-time args are
// the FabricMuxConfig kernel args, placed at positional offset A::cta_mux; its runtime args are the config's
// kernel_runtime_args block.

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"
#include <cstddef>
#include <array>
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::fabric {

template <typename A>
struct Mux {
    static constexpr uint32_t C = A::cta_mux;
    static constexpr size_t NUM_FULL_SIZE_CHANNELS = get_compile_time_arg_val(C + 0);
    static constexpr uint8_t NUM_BUFFERS_FULL_SIZE_CHANNEL = get_compile_time_arg_val(C + 1);
    static constexpr size_t BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL = get_compile_time_arg_val(C + 2);
    static constexpr size_t NUM_HEADER_ONLY_CHANNELS = get_compile_time_arg_val(C + 3);
    static constexpr uint8_t NUM_BUFFERS_HEADER_ONLY_CHANNEL = get_compile_time_arg_val(C + 4);
    static constexpr size_t status_address = get_compile_time_arg_val(C + 5);
    static constexpr size_t termination_signal_address = get_compile_time_arg_val(C + 6);
    static constexpr size_t connection_info_base_address = get_compile_time_arg_val(C + 7);
    static constexpr size_t connection_handshake_base_address = get_compile_time_arg_val(C + 8);
    static constexpr size_t sender_flow_control_base_address = get_compile_time_arg_val(C + 9);
    static constexpr size_t channels_base_l1_address = get_compile_time_arg_val(C + 10);
    static constexpr size_t local_fabric_router_status_address = get_compile_time_arg_val(C + 11);
    static constexpr size_t fabric_router_status_address = get_compile_time_arg_val(C + 12);
    static constexpr uint8_t NUM_EDM_BUFFERS = get_compile_time_arg_val(C + 13);
    static constexpr size_t NUM_FULL_SIZE_CHANNELS_ITERS = get_compile_time_arg_val(C + 14);
    static constexpr size_t NUM_ITERS_BETWEEN_TEARDOWN_CHECKS = get_compile_time_arg_val(C + 15);
    static constexpr ProgrammableCoreType CORE_TYPE =
        static_cast<ProgrammableCoreType>(get_compile_time_arg_val(C + 16));
    static constexpr bool wait_for_fabric_endpoint = get_compile_time_arg_val(C + 17) == 1;
    static constexpr bool in_program_local_l1 = get_compile_time_arg_val(C + 18) == 1;
    static constexpr uint32_t program_local_l1_cb_index = get_compile_time_arg_val(C + 19);
    static constexpr uint32_t l1_base_semaphore_id = get_compile_time_arg_val(C + 20);
    static constexpr size_t CHANNEL_STREAM_IDS_START_IDX = C + 21;
    static constexpr size_t NOC_ALIGN_PADDING_BYTES = 12;
    static constexpr bool ENABLE_RISC_CPU_DATA_CACHE = true;
    using FabricMuxToEdmSender = tt::tt_fabric::WorkerToFabricEdmSenderImpl<false, NUM_EDM_BUFFERS>;

    template <uint8_t NUM_BUFFERS>
    static void setup_channel(
        tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>* channel_ptr,
        tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>* worker_interface_ptr,
        bool& channel_connection_established,
        uint8_t channel_id,
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
            0,
            tt::tt_fabric::MUX_TO_WORKER_INTERFACE_STARTING_READ_COUNTER_VALUE);
        sender_flow_control_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
        connection_handshake_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
        channel_connection_established = false;
    }

    template <uint8_t NUM_BUFFERS>
    static void forward_data(
        tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>& channel,
        tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
        FabricMuxToEdmSender& fabric_connection,
        bool& channel_connection_established,
        StreamId my_channel_free_slots_stream_id,
        uint8_t channel_id) {
        bool has_unsent_payload = get_ptr_val(my_channel_free_slots_stream_id.get()) != NUM_BUFFERS;
        if (has_unsent_payload) {
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
        tt::tt_fabric::check_worker_connections<tt::tt_fabric::USE_DYNAMIC_CREDIT_ADDR, true>(
            worker_interface, channel_connection_established, my_channel_free_slots_stream_id.get());
    }

    void init() {}

    void run() {
        set_l1_data_cache<true>();
        size_t rt_args_idx = rt_index<A>(0);
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
        auto fabric_connection = FabricMuxToEdmSender::template build_from_args<CORE_TYPE>(rt_args_idx);

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
                i,
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
                i,
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
        fabric_connection.template open<use_worker_allocated_credit_address>();
        status_ptr[0] = tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC;
        if constexpr (in_program_local_l1) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(l1_base_semaphore_id)) = l1_base;
        }

        while (!got_immediate_termination_signal<true>(termination_signal_ptr)) {
            bool got_graceful_termination = got_graceful_termination_signal(termination_signal_ptr);
            if (got_graceful_termination) {
                bool all_channels_drained = true;
                for (uint8_t channel_id = 0; channel_id < NUM_FULL_SIZE_CHANNELS; channel_id++) {
                    all_channels_drained &= get_ptr_val(channel_id) == NUM_BUFFERS_FULL_SIZE_CHANNEL;
                }
                for (uint8_t channel_id = 0; channel_id < NUM_HEADER_ONLY_CHANNELS; channel_id++) {
                    all_channels_drained &=
                        get_ptr_val(channel_id + NUM_FULL_SIZE_CHANNELS) == NUM_BUFFERS_HEADER_ONLY_CHANNEL;
                }
                if (all_channels_drained) {
                    break;
                }
            }
            for (size_t i = 0; i < NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
                for (size_t iter = 0; iter < NUM_FULL_SIZE_CHANNELS_ITERS; iter++) {
                    for (uint8_t channel_id = 0; channel_id < NUM_FULL_SIZE_CHANNELS; channel_id++) {
                        forward_data<NUM_BUFFERS_FULL_SIZE_CHANNEL>(
                            full_size_channels[channel_id],
                            full_size_channel_worker_interfaces[channel_id],
                            fabric_connection,
                            full_size_channel_connection_established[channel_id],
                            StreamId{channel_stream_ids[channel_id]},
                            channel_id);
                    }
                }
                for (uint8_t channel_id = 0; channel_id < NUM_HEADER_ONLY_CHANNELS; channel_id++) {
                    forward_data<NUM_BUFFERS_HEADER_ONLY_CHANNEL>(
                        header_only_channels[channel_id],
                        header_only_channel_worker_interfaces[channel_id],
                        fabric_connection,
                        header_only_channel_connection_established[channel_id],
                        StreamId{channel_stream_ids[channel_id + NUM_FULL_SIZE_CHANNELS]},
                        channel_id + NUM_FULL_SIZE_CHANNELS);
                }
            }
        }
        fabric_connection.close();
        noc_async_full_barrier();
        noc_clear_packet_tags(noc_index);
        status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
        set_l1_data_cache<false>();
    }

    void teardown() {}
};

}  // namespace lego::fabric
