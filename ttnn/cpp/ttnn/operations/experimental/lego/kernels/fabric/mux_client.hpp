// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Client side of a program-local fabric mux, as all_gather_async's minimal_default_writer.cpp uses it under
// USE_WORKER_MUX: connect to the mux channel, send through it like through a fabric connection, then disconnect, and
// let the termination master stop the mux once every client of that mux has disconnected.
//
// CT (A::): mux_num_buffers mux_buffer_size mux_l1_base_sem mux_termination_offset num_mux_clients
// RT (17 words from `first`): the ccl::fabric_mux_connection_rt_args block (valid, is_termination_master, mux x, y,
//   channel base, connection info, handshake, flow control, buffer index offsets, channel id, termination sync sem,
//   local mux status / flow control / teardown / buffer index sems, termination master x, y)

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::fabric {

template <typename A>
struct MuxClient {
    using Sender = tt::tt_fabric::WorkerToFabricMuxSender<A::mux_num_buffers>;
    Sender connection;
    bool valid = false, is_termination_master = false;
    uint8_t mux_x = 0, mux_y = 0, channel_id = 0;
    size_t channel_base, connection_info, handshake, flow_control, buffer_index;
    uint32_t termination_sync_id, status_addr, flow_control_addr, teardown_addr, buffer_index_addr;
    uint32_t master_x, master_y;
    uint32_t l1_base = 0;

    void read_args(uint32_t first) {
        uint32_t i = first;
        valid = rt<A>(i++) == 1;
        is_termination_master = rt<A>(i++);
        mux_x = rt<A>(i++);
        mux_y = rt<A>(i++);
        channel_base = rt<A>(i++);
        connection_info = rt<A>(i++);
        handshake = rt<A>(i++);
        flow_control = rt<A>(i++);
        buffer_index = rt<A>(i++);
        channel_id = rt<A>(i++);
        termination_sync_id = rt<A>(i++);
        status_addr = get_semaphore(rt<A>(i++));
        flow_control_addr = get_semaphore(rt<A>(i++));
        teardown_addr = get_semaphore(rt<A>(i++));
        buffer_index_addr = get_semaphore(rt<A>(i++));
        master_x = rt<A>(i++);
        master_y = rt<A>(i++);
    }

    Sender* open() {
        if (!valid) {
            return nullptr;
        }
        connection = tt::tt_fabric::build_connection_to_program_local_fabric_mux<A::mux_num_buffers>(
            mux_x,
            mux_y,
            channel_id,
            A::mux_num_buffers,
            A::mux_buffer_size,
            A::mux_l1_base_sem,
            channel_base,
            connection_info,
            handshake,
            flow_control,
            buffer_index,
            flow_control_addr,
            teardown_addr,
            buffer_index_addr,
            status_addr,
            l1_base);
        tt::tt_fabric::fabric_client_connect(connection);
        return &connection;
    }

    void close() {
        if (!valid) {
            return;
        }
        Noc noc_obj;
        tt::tt_fabric::fabric_client_disconnect(connection);
        if (is_termination_master) {
            Semaphore<> termination_sync(termination_sync_id);
            termination_sync.wait(A::num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(mux_x, mux_y, l1_base + A::mux_termination_offset);
        } else {
            uint64_t dest = safe_get_noc_addr(master_x, master_y, get_semaphore(termination_sync_id), 0);
            noc_semaphore_inc(dest, 1);
            noc_obj.async_atomic_barrier();
        }
    }
};

}  // namespace lego::fabric
