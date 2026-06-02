// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Worker-side fabric-mux connection helpers for the all_gather_rms_norm writer.
//
// A fabric mux core opens the fabric EDM connection and multiplexes many worker cores onto it, so
// that more compute cores than there are fabric sender channels can each send off-device. Worker
// cores connect to the mux as clients (one channel per direction) and then issue the SAME fabric
// send API as a direct connection (the send primitives in linear/api.h are templated on the sender
// type and accept WorkerToFabricMuxSender).
//
// MuxConnection / parse_mux_connection_args / build_and_connect / close_mux are adapted from
// experimental/ccl/all_gather_minimal_matmul_async/device/kernels/matmul_dataflow_common.hpp. The
// matmul version gates connection_valid on a core-order index; here every worker connects, so the
// gate is dropped.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"

template <uint32_t NumBuffersPerChannel, uint32_t ChannelBufferSizeBytes>
struct MuxConnection {
    bool connection_valid;
    bool is_termination_master;
    uint8_t fabric_mux_x;
    uint8_t fabric_mux_y;
    size_t fabric_mux_channel_base_address;
    size_t fabric_mux_connection_info_address;
    size_t fabric_mux_connection_handshake_address;
    size_t fabric_mux_flow_control_address;
    size_t fabric_mux_buffer_index_address;
    uint8_t fabric_mux_channel_id;

    uint32_t termination_sync_address;
    uint32_t local_fabric_mux_status_address;
    uint32_t local_flow_control_address;
    uint32_t local_teardown_address;
    uint32_t local_buffer_index_address;

    uint32_t termination_master_noc_x;
    uint32_t termination_master_noc_y;

    tt::tt_fabric::WorkerToFabricMuxSender<NumBuffersPerChannel> connection;

    FORCE_INLINE tt::tt_fabric::WorkerToFabricMuxSender<NumBuffersPerChannel>* build_and_connect(
        size_t fabric_mux_status_address) {
        if (!connection_valid) {
            return nullptr;
        }

        connection = tt::tt_fabric::build_connection_to_fabric_endpoint<NumBuffersPerChannel>(
            fabric_mux_x,
            fabric_mux_y,
            fabric_mux_channel_id,
            NumBuffersPerChannel,
            ChannelBufferSizeBytes,
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

        tt::tt_fabric::fabric_client_connect(connection);

        return &connection;
    }
};

// Parse one direction's mux connection runtime args (17 values, matching the host emit order).
template <uint32_t NumBuffersPerChannel, uint32_t ChannelBufferSizeBytes>
FORCE_INLINE MuxConnection<NumBuffersPerChannel, ChannelBufferSizeBytes> parse_mux_connection_args(uint32_t& argidx) {
    MuxConnection<NumBuffersPerChannel, ChannelBufferSizeBytes> mux;

    mux.connection_valid = (get_arg_val<uint32_t>(argidx++) == 1);
    mux.is_termination_master = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_x = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_y = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_channel_base_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_connection_info_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_flow_control_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_buffer_index_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_channel_id = get_arg_val<uint32_t>(argidx++);

    mux.termination_sync_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    mux.local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    mux.local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    mux.local_teardown_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    mux.local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(argidx++));

    mux.termination_master_noc_x = get_arg_val<uint32_t>(argidx++);
    mux.termination_master_noc_y = get_arg_val<uint32_t>(argidx++);

    return mux;
}

// Disconnect from the mux; the elected termination master waits for all other clients then signals
// the mux to terminate, every other client increments the master's termination-sync semaphore.
template <typename ConnectionHandleType>
FORCE_INLINE void close_mux(
    ConnectionHandleType mux_connection_handle,
    bool is_termination_master,
    uint32_t termination_sync_address,
    uint32_t num_mux_clients,
    const uint8_t fabric_mux_x,
    const uint8_t fabric_mux_y,
    size_t fabric_mux_termination_signal_address,
    uint32_t termination_master_noc_x,
    uint32_t termination_master_noc_y) {
    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);
    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
}
