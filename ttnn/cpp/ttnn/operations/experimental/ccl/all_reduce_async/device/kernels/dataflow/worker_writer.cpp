// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using namespace dataflow_kernel_lib::ccl;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);
constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<8>();
constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<8 + ccl_routing_utils::num_line_multicast_args>();

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    uint32_t reduction_input_cb_id = get_arg_val<address_t>(arg_idx++);
    address_t reduction_input_addr = get_write_ptr(reduction_input_cb_id);

    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_mcast_cores = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t reduction_semaphore_send_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_mcast_ranges = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t link = get_arg_val<uint32_t>(arg_idx++);

    Noc noc_obj;
    // The packet headers now come from PacketHeaderPool (owned by the stream below), so the op's
    // reserved_packet_header_cb_id / num_packet_headers_storable compile-time args are dead. They
    // are left in place so the host factory's arg indices are untouched; removing the CB and those
    // two args is a follow-up host-side cleanup.
    CircularBuffer cb0(cb0_id);

    // Set up for mcasting to reduction workers
    Semaphore<> reduction_semaphore_send(reduction_semaphore_send_id);
    reduction_semaphore_send.set(VALID);

    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    tt_l1_ptr uint32_t* mcast_dest_noc_start_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_start_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;

    // Build the DUPLEX egress: this worker drives both the forward and the backward direction of one
    // fabric connection. The build only STARTS the open (open_finish carries a barrier, so it is
    // deferred as late as possible — sender.open() below does it), exactly as before.
    FabricDuplexSender<> sender(arg_idx, /*alignment=*/1);

    // Open the duplex egress, binding BOTH directions' MULTICAST routes at once — the route-pair type
    // selects the chip-level cast, so this is a Cast::Multicast stream and every issue below fans out
    // to whichever directions this worker actually has wired (an end-of-line worker has one).
    auto stream = sender.open(forward_multicast_route_info, backward_multicast_route_info);
    // Arm once, issue many. Both channels draw their own per-direction pooled headers. The armed size
    // is the maximum packet; each issue below carries its own size because the last packet of a shard
    // is short.
    const uint32_t max_payload_size_bytes = packet_size_in_pages * tensor0_page_size;
    auto writer = stream.arm_write(max_payload_size_bytes);
    auto fused = stream.arm_fused_write_inc(max_payload_size_bytes, /*val=*/1, /*flush=*/false);

    // 1. mcast via fabric to remote tensor addresses
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    uint32_t writer_chip_offset = my_chip_id * num_tiles_per_core * tensor0_page_size;

    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
        num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
        cb0.wait_front(num_tiles_to_read_this_core);
        size_t l1_read_addr = cb0.get_read_ptr();

        uint64_t noc0_dest_noc_addr =
            safe_get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], reduction_input_addr + writer_chip_offset);

        uint64_t sema_noc_addr = safe_get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], out_ready_sem_bank_addr);

        // Within-shard offset
        noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;

        const uint32_t payload_size_bytes = num_tiles_to_read_this_core * tensor0_page_size;

        // Last packet of this shard (or of this worker's whole range): fold the receiver's semaphore
        // bump into the same packet, so the reduction worker learns the slice landed without a second
        // packet. Otherwise it is a plain payload write. Both mirror the payload locally as well as
        // forwarding it, and both fan out over every connected direction.
        if (shard_tile_id + num_tiles_to_read_this_core >= num_tiles_per_core ||
            tiles_read + num_tiles_to_read_this_core >= num_tiles_to_read) {
            fused.write_fused_with_local_copy(noc0_dest_noc_addr, l1_read_addr, sema_noc_addr, payload_size_bytes);
            // The fused issue deliberately does not flush (it pairs with this op's semaphore
            // protocol), so the flush barrier stays here exactly as before.
            noc_obj.async_writes_flushed();
        } else {
            writer.write_with_local_copy(noc0_dest_noc_addr, l1_read_addr, payload_size_bytes);
        }

        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id += num_tiles_to_read_this_core;
        if (shard_tile_id >= num_tiles_per_core) {
            shard_tile_id = 0;
            core_id++;
        }
        cb0.pop_front(num_tiles_to_read_this_core);
    }

    // 2. local semaphore increment
    // Device 2.0 migration: legacy primitive retained: out_ready_sem_bank_addr is the address of a GlobalSemaphore.
    // Semaphore<> binds to per-program ids via get_semaphore<>(id), so it cannot wrap a GlobalSemaphore.
    for (uint32_t i = 0; i < core_id; i++) {
        noc_semaphore_inc(safe_get_noc_addr(core_noc_x[i], core_noc_y[i], out_ready_sem_bank_addr), 1);
    }

    // close() drains (write + atomic barriers) then closes both directions — the close_start /
    // close_finish pair and their is_logically_connected gating are handled inside. The stream dtor
    // would also close; this is idempotent.
    stream.close();

    noc_obj.async_write_barrier();
}
