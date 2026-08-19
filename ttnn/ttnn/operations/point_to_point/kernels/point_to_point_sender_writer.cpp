// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — sender writer (BRISC). Phases 1, 3 and 4.
//
//   1. Wait for the receiver's "ready" fabric inc, then re-arm the semaphore
//      BEFORE our own outgoing inc (the cache-reuse footgun,
//      ccl_helpers_dataflow.hpp:111-113).
//   3. Open the stream, arm the write + inc channels, frame cb_shard_pages into
//      packets and issue total_packets fabric writes into the RECEIVER's
//      intermediate DRAM.
//   4. Fabric-inc the receiver's semaphore ("done"), then close() — which drains
//      the write + atomic barriers before tearing down, so the inc can never
//      overtake or be lost.
//
// Fabric egress goes entirely through the safety-by-construction CCL helper
// (FabricStreamSender -> open(route) -> arm_unicast_write / arm_inc -> issue ->
// close). The op owns only what the helper banner says it does not:
//   * the page<->packet coalescing / segmentation (ccl_helpers_dataflow.hpp:130-133
//     names it first in the "does NOT own" list) — via tt_memmove;
//   * the local semaphore wait + reset halves of the cross-device sync (:104-113);
//   * address generation — the intermediate TensorAccessor is consumed by
//     write_page, never re-wrapped (:130-140).
//
// MANDATORY (op_design.md Key Risk #1): the intermediate TensorAccessor is built
// with exactly TWO arguments so its per-bank stride is the CT-baked
// buffer.aligned_page_size(). packet_size is used ONLY as the armed fabric
// payload size. Both endpoints build the accessor from the same CT args of the
// same mesh tensor, so "page k" means the same bytes on both chips.
//
// Two disjoint framing regimes, both implemented:
//   A (page_segments == 1): several shard pages ride in one packet at
//     aligned_page_size stride. total_packets = ceil(num_pages / pages_per_packet).
//   B (page_segments  > 1): one shard page is split across page_segments packets
//     of packet_size bytes. total_packets = page_segments * num_pages.
//
// Regime A's LAST packet always carries a full packet_size payload even when it
// holds fewer than pages_per_packet live pages; the trailing bytes are stale
// staging content that lands in the intermediate's tail and is never read back.
// This is intentional — the armed payload size is a per-stream invariant
// (ccl_helpers_dataflow.hpp:486) and the intermediate is sized for it.

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;
using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t cb_shard_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_packet_staging = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr uint32_t page_segments = get_compile_time_arg_val(3);
    constexpr auto intermediate_args = TensorAccessorArgs<4>();

    uint32_t ai = 0;
    const uint32_t intermediate_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t num_pages = get_arg_val<uint32_t>(ai++);
    const uint32_t total_packets = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t packet_size = get_arg_val<uint32_t>(ai++);
    const uint32_t pages_per_packet = get_arg_val<uint32_t>(ai++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t dst_num_hops = get_arg_val<uint32_t>(ai++);

    // Fabric connection arg block (laid out by _append_fabric_rt_args, mirroring
    // ttnn::ccl::dataflow::append_ccl_fabric_rt_args) is LAST; its leading
    // has_forward flag also encodes the send direction. The FabricStreamSender
    // consumes the whole block, advancing conn_arg_idx.
    size_t conn_arg_idx = ai;
    const bool dst_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
    FabricStreamSender<> sender(conn_arg_idx, dst_is_forward, alignment);

    // 2-argument ctor: stride == buffer.aligned_page_size() (CT-baked). NOT packet_size.
    const auto intermediate = TensorAccessor(intermediate_args, intermediate_addr);

    const uint32_t aligned_page_size = round_up(page_size, alignment);

    // Reserve-once scratch: claim the L1 region, then address it raw. Producer and
    // consumer are the same kernel, so there is no CB handshake to balance (0
    // pushes / 0 waits). One slot suffices — write_page copies the payload into the
    // fabric channel buffer under flow control before returning
    // (ccl_helpers_dataflow.inl:59-63), so the staging buffer is immediately free.
    cb_reserve_back(cb_packet_staging, 1);
    const uint32_t staging = get_write_ptr(cb_packet_staging);

    // --- Phase 1: wait for the receiver's "ready", then re-arm BEFORE our own inc.
    // Gating on the receiver's own ready-inc (which it can only issue after its
    // previous program on that device retired) is what makes invocation k+1's
    // "done" un-eraseable by invocation k's reset.
    auto sem_ptr = dataflow_kernel_lib::addr_to_l1_ptr(sem_addr);
    noc_semaphore_wait_min(sem_ptr, 1);
    noc_semaphore_set(sem_ptr, 0);

    // --- Phase 3: open the stream (route bound ONCE), arm both channels.
    auto stream = sender.open(unicast_route(dst_num_hops));
    auto writer = stream.arm_unicast_write(packet_size);  // invariant on-wire payload size
    auto done = stream.arm_inc(1);                        // invariant "done" inc value

    if constexpr (page_segments == 1) {
        // Regime A — coalesce: pack up to pages_per_packet pages into one packet.
        for (uint32_t pkt = 0; pkt < total_packets; ++pkt) {
            const uint32_t base = pkt * pages_per_packet;
            const uint32_t n = std::min(pages_per_packet, num_pages - base);
            for (uint32_t k = 0; k < n; ++k) {
                cb_wait_front(cb_shard_pages, 1);
                tt_memmove<false, false, false, 0>(
                    staging + k * aligned_page_size, get_read_ptr(cb_shard_pages), page_size);
                cb_pop_front(cb_shard_pages, 1);
            }
            writer.write_page(staging, pkt, intermediate);
        }
    } else {
        // Regime B — segment: split each page across page_segments packets.
        uint32_t pkt = 0;
        for (uint32_t p = 0; p < num_pages; ++p) {
            cb_wait_front(cb_shard_pages, 1);
            const uint32_t src = get_read_ptr(cb_shard_pages);
            for (uint32_t s = 0; s < page_segments; ++s) {
                const uint32_t off = s * packet_size;
                const uint32_t bytes = std::min(page_size - off, packet_size);
                tt_memmove<false, false, false, 0>(staging, src + off, bytes);
                writer.write_page(staging, pkt, intermediate);
                ++pkt;
            }
            cb_pop_front(cb_shard_pages, 1);
        }
    }

    // --- Phase 4: signal the receiver "done", then drain + close. All payload
    // writes and this inc are issued on the SAME connection with the same route,
    // so they are delivered in issue order: sem >= 1 implies every byte landed.
    done.inc(get_noc_addr(sem_addr));
    stream.close();  // drains write + atomic barriers, then closes (dtor is idempotent)
}
