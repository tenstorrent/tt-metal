// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_reduce — writer (BRISC). Two sequential phases.
//
//   Phase 1  Chip-level MULTICAST this device's whole shard to every peer on the
//            line, driving BOTH fabric directions from this ONE core through the
//            CCL dataflow helper's DUPLEX tier:
//
//              FabricDuplexSender -> open(fwd_mcast, bwd_mcast) -> Cast::Multicast
//              stream -> arm_write / arm_fused_write_inc -> issue -> close()
//
//            A MULTICAST route PAIR selects Cast::Multicast at compile time, and
//            every issue fans out to each CONNECTED direction (an end-of-line
//            worker has exactly one — DuplexConn::has(dir) suppresses both the arm
//            and the issue for the missing side, which is also what keeps a
//            zero-range multicast header off the wire on Linear).
//
//            Pages 0..P-2 are plain multicast payload writes; page P-1 is a FUSED
//            write + atomic-inc (val=1, flush=true). Because the fused packet is
//            multicast, EVERY chip in that direction's range performs both the
//            payload write and the increment in one delivery, so each device
//            receives exactly N-1 increments — one per peer. flush=true is
//            load-bearing: the payload lands in DRAM while the semaphore lives in
//            L1, so without it the inc could overtake the write and the peer's
//            reader would read stale DRAM (wrong values, no hang). It makes the
//            receiving fabric endpoint flush its NoC write pipeline before the inc,
//            which also covers the earlier in-order packets on that channel.
//
//            The op owns only cb_wait_front / noc_async_writes_flushed /
//            cb_pop_front around each issue — the helper owns the connection
//            lifecycle, direction set, packet headers, route programming and the
//            set_state/with_state dance. noc_async_writes_flushed() before the pop
//            is required: close()/drain() are write+atomic barriers only and do NOT
//            guarantee the fabric sender has finished reading the CB slot.
//
//   Phase 2  Drain the reduced tiles the compute kernel produced to output DRAM.
//            Strictly after phase 1, which depends only on reader phase 1, so the
//            dependency chain is acyclic: reader1 -> writer1 -> (fabric) -> reader2
//            -> reader3 -> compute -> writer2.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib::ccl;

constexpr uint32_t cb_broadcast_pages = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
constexpr uint32_t fabric_alignment = get_compile_time_arg_val(2);
constexpr uint32_t my_chip_id = get_compile_time_arg_val(3);
constexpr uint32_t num_devices = get_compile_time_arg_val(4);

// Pure-multicast route blocks, one per fabric direction (6 uint32 each). The
// designated initializers .dst_mesh_id / .dst_chip_id are the first members of the
// two anonymous unions, i.e. start_distance_in_hops / range_hops on the 1-D path.
constexpr uint32_t kFwdRouteIdx = 5;
constexpr uint32_t kBwdRouteIdx = kFwdRouteIdx + ccl_routing_utils::num_line_multicast_args;
constexpr ccl_routing_utils::line_multicast_route_info_t forward_mcast_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<kFwdRouteIdx>();
constexpr ccl_routing_utils::line_multicast_route_info_t backward_mcast_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<kBwdRouteIdx>();

constexpr auto gathered_args = TensorAccessorArgs<kBwdRouteIdx + ccl_routing_utils::num_line_multicast_args>();
constexpr auto output_args = TensorAccessorArgs<gathered_args.next_compile_time_args_offset()>();

void kernel_main() {
    static_assert(num_devices >= 2, "all_reduce needs at least 2 devices on the line");

    uint32_t ai = 0;
    const uint32_t gathered_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_shard = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t recv_sem_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t peer_noc_x = get_arg_val<uint32_t>(ai++);
    const uint32_t peer_noc_y = get_arg_val<uint32_t>(ai++);

    // The fabric connection arg block starts here:
    //   [has_forward][<forward conn args>?][has_backward][<backward conn args>?]
    // FabricDuplexSender takes it BY REFERENCE and advances past the whole block.
    size_t conn_arg_idx = ai;

    const auto gathered = TensorAccessor(gathered_args, gathered_addr, page_size);
    const auto output = TensorAccessor(output_args, output_addr, page_size);
    const uint32_t P = pages_per_shard;

    // A multicast sender cannot know which receiver it is talking to, so every
    // sender writes into slot == its OWN id and all senders agree on that mapping.
    const uint32_t slot_base = my_chip_id * P;
    // Same logical core (0, 0) on every chip => identical NoC coords, and the
    // GlobalSemaphore is at the same L1 address on every chip, so one noc0-encoded
    // address is the receive counter of every peer the multicast reaches.
    const uint64_t peer_sem_noc_addr = safe_get_noc_addr(peer_noc_x, peer_noc_y, recv_sem_addr, 0);

    // ---- Phase 1: duplex multicast broadcast of the local shard ----
    {
        // Declare the sender FIRST: the stream borrows its connection, so the
        // sender must outlive it.
        FabricDuplexSender<> sender(conn_arg_idx, fabric_alignment);
        auto stream = sender.open(forward_mcast_route, backward_mcast_route);
        // Arm once, issue many. Every packet is exactly one tile page, so the armed
        // size is the invariant on-wire size for both channels.
        auto payload = stream.arm_write(page_size);
        auto fused = stream.arm_fused_write_inc(page_size, /*val=*/1, /*flush=*/true);

        for (uint32_t p = 0; p < P; ++p) {
            cb_wait_front(cb_broadcast_pages, 1);
            const uint32_t l1 = get_read_ptr(cb_broadcast_pages);
            if (p + 1 == P) {
                // The FUSED channel has no page/addrgen overload, so resolve the
                // page here exactly as write_page() does internally
                // (ccl_helpers_dataflow.inl:463-467). noc0-encoded destination:
                // the gathered buffer is ONE mesh allocation, so this address
                // resolves to the correct DRAM bank on every peer.
                const uint64_t dst_noc_addr =
                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(gathered, slot_base + p, 0);
                fused.write_fused(dst_noc_addr, l1, peer_sem_noc_addr);
            } else {
                // Helper-owned page resolution (same convenience overload
                // all_gather's writer uses): armed size + noc0 conversion.
                payload.write_page(l1, slot_base + p, gathered);
            }
            // The fabric sender must have read the page out of the CB slot before
            // the reader may refill it.
            noc_async_writes_flushed();
            cb_pop_front(cb_broadcast_pages, 1);
        }

        stream.close();  // drains (write + atomic barriers) then closes; idempotent
    }

    // ---- Phase 2: drain the reduced tiles to output DRAM ----
    for (uint32_t p = 0; p < P; ++p) {
        cb_wait_front(cb_output_tiles, 1);
        const uint32_t l1 = get_read_ptr(cb_output_tiles);
        noc_async_write(l1, output.get_noc_addr(p), page_size);
        noc_async_writes_flushed();
        cb_pop_front(cb_output_tiles, 1);
    }
    noc_async_write_barrier();
}
