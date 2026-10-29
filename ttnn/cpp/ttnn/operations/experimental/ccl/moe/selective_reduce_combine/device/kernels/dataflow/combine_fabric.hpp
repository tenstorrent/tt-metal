// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The combine's fabric side on the CCL path, shared by this op's writer and moe_compute's expert rows combine: one
// Fabric Mux V2 connection per open direction, the init barrier, unicast sends, and the teardown (mux drain, link sync,
// final barrier, close). The host's append_combine_fabric sets the named compile-time args read here, the defines
// DEST_CHIP_ID, DEST_MESH_ID, DIRECTIONS and REPLICATE_GROUP_AXIS, and each worker's runtime-arg block.
#pragma once

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

namespace combine_fabric {

using ttnn::operations::ccl::common::ReplicateGroup;

constexpr uint32_t packet_header_cb_id = get_named_compile_time_arg_val("packet_header_cb_id");
constexpr uint32_t num_workers_per_link = get_named_compile_time_arg_val("num_workers_per_link");
// the workers' bounding box, virtual coordinates
constexpr uint32_t noc_x_start = get_named_compile_time_arg_val("noc_x_start");
constexpr uint32_t noc_y_start = get_named_compile_time_arg_val("noc_y_start");
constexpr uint32_t noc_x_end = get_named_compile_time_arg_val("noc_x_end");
constexpr uint32_t noc_y_end = get_named_compile_time_arg_val("noc_y_end");
constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
constexpr uint32_t src_chip_id = get_named_compile_time_arg_val("src_chip_id");
constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");
constexpr uint32_t fabric_max_packet_size_bytes = get_named_compile_time_arg_val("fabric_max_packet_size_bytes");
constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
constexpr auto topology = tt::tt_fabric::Topology(get_named_compile_time_arg_val("topology"));
constexpr uint32_t alignment = get_named_compile_time_arg_val("alignment");

constexpr ReplicateGroup replicate_axis = ReplicateGroup(REPLICATE_GROUP_AXIS);
static_assert(replicate_axis != ReplicateGroup::NONE, "the combine reduces along one mesh axis");
constexpr uint32_t replicate_factor = replicate_axis == ReplicateGroup::COLS ? mesh_cols : mesh_rows;
constexpr uint8_t replicate_group_devices = num_devices / replicate_factor;
constexpr uint8_t num_directions = 4;

// Linearized mesh coordinate of the device that owns global token t: token t is on device t / tokens_per_device of
// this device's replicate group.
template <uint32_t tokens_per_device>
FORCE_INLINE uint32_t device_of_token(uint32_t t) {
    const uint32_t device_in_group = t / tokens_per_device;
    if constexpr (replicate_axis == ReplicateGroup::ROWS) {
        return (linearized_mesh_coord / mesh_cols) * mesh_cols + device_in_group;
    } else {
        return device_in_group * mesh_cols + linearized_mesh_coord % mesh_cols;
    }
}

using MuxSender = tt::tt_fabric::FabricMuxV2Sender<>;

// One worker's fabric state. Runtime args from `arg` on: is_link_sync_core, link sync semaphore id, the link sync
// core's virtual x and y, then FabricMuxV2Sender::build_from_args' args per open direction (the host's neighbor order).
class Connections {
public:
    explicit Connections(size_t& arg) :
        is_link_sync_core_(get_arg_val<uint32_t>(arg++)),
        link_sync_semaphore_addr_(get_semaphore(get_arg_val<uint32_t>(arg++))),
        link_sync_core_x_(get_arg_val<uint32_t>(arg++)),
        link_sync_core_y_(get_arg_val<uint32_t>(arg++)) {
        for (uint8_t d = 0; d < num_directions; ++d) {
            if (directions_[d]) {
                mux_[d] = MuxSender::build_from_args(arg);
            }
        }
    }

    // Packet headers, then a blocking open of every connection (waits for each mux to be READY). Records each
    // channel's free-slot count right after the open, all slots free, for drain().
    void open() {
        CircularBuffer cb_packet_header(packet_header_cb_id);
        for (uint8_t i = 0; i < 3; ++i) {
            cb_packet_header.reserve_back(1);
            headers_[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_packet_header.get_write_ptr());
            cb_packet_header.push_back(1);
        }
        for (uint8_t d = 0; d < num_directions; ++d) {
            if (directions_[d]) {
                mux_[d].open();
                capacity_[d] = mux_[d].get_num_free_write_slots();
            }
        }
    }

    // The init sync core announces this device to every other device of its replicate group.
    void send_init(uint32_t init_semaphore_addr) {
        ttnn::operations::ccl::common::send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            replicate_axis,
            num_devices>(mux_, headers_[1], dest_chip_ids_, dest_mesh_ids_, get_noc_addr(init_semaphore_addr));
    }

    // Every worker waits until the other devices of the group are ready; the init sync core then releases the
    // other `workers - 1` workers of the bounding box.
    void wait_init(const Noc& noc, uint32_t init_semaphore_addr, bool is_init_sync_core, uint32_t workers) {
        auto* init_semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_addr);
        // Device 2.0 migration: legacy primitive retained: init_semaphore_addr is a GlobalSemaphore address.
        noc_semaphore_wait(init_semaphore, replicate_group_devices - 1);
        if (is_init_sync_core) {
            // this kernel uses NOC1: start and end swap
            const uint64_t multicast_addr =
                get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, init_semaphore_addr, 1);
            noc_semaphore_set_multicast(init_semaphore_addr, multicast_addr, workers - 1, /*linked=*/false, /*noc=*/1);
            noc.async_writes_flushed();
        }
        noc_semaphore_set(init_semaphore, 0);
    }

    // size_bytes from L1 at payload_addr to page `page` (+ offset_bytes) of the output on device dest_device.
    template <typename OutputAccessor>
    void send(
        const OutputAccessor& output,
        uint32_t dest_device,
        uint32_t payload_addr,
        uint32_t page,
        uint32_t size_bytes,
        uint32_t offset_bytes) {
        ttnn::operations::ccl::common::fabric_send_chip_unicast_noc_unicast_1d<
            linearized_mesh_coord,
            topology,
            mesh_rows,
            mesh_cols,
            fabric_max_packet_size_bytes>(
            output, mux_, headers_[0], dest_device, payload_addr, page, size_bytes, alignment, offset_bytes);
    }

    // After the worker's last send and its NoC barriers: waits until the mux forwarded every packet of this worker,
    // then the link's barrier. The link sync core waits for the link's other workers, increments every device's
    // final semaphore, closes, and waits for the other devices' increments; the others close and report to it.
    void finish(const Noc& noc, uint32_t final_semaphore_addr) {
        for (uint8_t d = 0; d < num_directions; ++d) {
            if (directions_[d]) {
                while (mux_[d].get_num_free_write_slots() != capacity_[d]) {
                }
            }
        }
        if (is_link_sync_core_) {
            auto* link_sync_semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(link_sync_semaphore_addr_);
            noc_semaphore_wait(link_sync_semaphore, num_workers_per_link - 1);
            noc_semaphore_set(link_sync_semaphore, 0);
            ttnn::operations::ccl::common::fabric_multicast_bidirectional_atomic_inc_1d<
                linearized_mesh_coord,
                topology,
                mesh_rows,
                mesh_cols,
                replicate_axis,
                /*DoubleAntipodalAtomicInc=*/(topology == tt::tt_fabric::Topology::Ring)>(
                mux_, headers_[1], headers_[2], get_noc_addr(final_semaphore_addr, /*noc=*/1));
            noc.async_write_barrier();
            noc.async_atomic_barrier();
            // the barrier packets sit in the mux channel; the mux forwards them before it terminates
            close();
            // a ring delivers the antipodal device's increment from both directions
            constexpr uint32_t expected_increments =
                topology == tt::tt_fabric::Topology::Linear ? replicate_group_devices - 1 : replicate_group_devices;
            auto* final_semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_semaphore_addr);
            noc_semaphore_wait(final_semaphore, expected_increments);
            noc_semaphore_set(final_semaphore, 0);
        } else {
            close();
            // Device 2.0 migration: legacy primitive retained: the link sync semaphore is a get_semaphore(id) address
            // on a remote core.
            noc_semaphore_inc(
                safe_get_noc_addr(link_sync_core_x_, link_sync_core_y_, link_sync_semaphore_addr_, /*noc=*/1),
                1,
                /*noc=*/1);
            noc.async_write_barrier();
            noc.async_atomic_barrier();
        }
    }

private:
    // close request and the manager's teardown ack per connection; the mux terminates after its last client closes
    void close() {
        for (uint8_t d = 0; d < num_directions; ++d) {
            if (directions_[d]) {
                mux_[d].close();
            }
        }
    }

    static constexpr uint8_t dest_chip_ids_[num_devices] = DEST_CHIP_ID;
    static constexpr uint8_t dest_mesh_ids_[num_devices] = DEST_MESH_ID;
    const std::array<bool, num_directions> directions_ = DIRECTIONS;
    const bool is_link_sync_core_;
    const uint32_t link_sync_semaphore_addr_, link_sync_core_x_, link_sync_core_y_;
    std::array<MuxSender, num_directions> mux_;
    std::array<uint32_t, num_directions> capacity_{};
    volatile PACKET_HEADER_TYPE* headers_[3];
};

}  // namespace combine_fabric
