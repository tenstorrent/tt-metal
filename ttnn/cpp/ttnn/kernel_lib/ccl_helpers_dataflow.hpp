// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file ccl_helpers_dataflow.hpp
 * @brief Multi-device CCL (fabric) dataflow-kernel helpers.
 *
 * The multi-device analog of the single-device dataflow-helper library (#45698,
 * @c reduce/dfb/tilize_helpers_dataflow). It gives op authors an intent-level surface
 * for the footgun-heavy fabric egress plumbing — connection lifecycle + direction,
 * packet-header allocation, 1-D unicast route programming, flow-controlled fabric
 * writes, and cross-device atomic-inc — so they express "send these pages from A to B"
 * / "signal that peer" instead of re-deriving connection build modes, header framing,
 * routing, and the send/flush handshake.
 *
 * This is PURE DATA MOVEMENT. No compute/unpack/math/pack appears here. Reduction
 * collectives (all_reduce, reduce_scatter) are out of scope.
 *
 * @par SCOPE & EXTENSION — read this first.
 *   The shipped, verified surface is the 1-D UNICAST pattern exercised end-to-end by
 *   point_to_point (the only migrated pure-DM CCL op): open a one-direction fabric
 *   egress, program a line-unicast route by hop distance, write pages with flow
 *   control, atomic-inc a remote semaphore for the handshake/completion signal, close.
 *   It is built on the LINEAR (1-D) fabric API (@c tt_metal/fabric/hw/inc/linear/api.h),
 *   which the TT-Fabric spec guarantees builds and runs UNCHANGED on a 2-D (mesh)
 *   fabric — so a 1-D-API CCL kernel is forward-compatible to mesh hardware.
 *
 *   Richer collectives (e.g. all_gather) layer line-MULTICAST routes and 4-chunk
 *   SCATTER writes on top of this SAME substrate, slotting in behind the same
 *   FabricStreamSender call sites: a multicast route setter over
 *   @c ccl_routing_utils::fabric_set_line_multicast_route, and a scatter write over
 *   @c minimal_ccl_common scatter writes with stateful (set_state/with_state) headers.
 *   They are deliberately NOT shipped here: the only op that exercises them is
 *   all_gather, which is @c @skip_for_blackhole and therefore unverifiable on the
 *   available hardware. Add them together WITH the all_gather migration and its test —
 *   not as unverified surface ahead of a caller.
 *
 * @par Recv-side coordination is op-owned (intentionally NOT wrapped).
 *   The receive INGRESS is a local NoC read the op already owns. Cross-device
 *   synchronization is a remote atomic-inc (@c FabricStreamSender::inc_remote) paired
 *   with a local @c noc_semaphore_wait_min(sem, threshold) — a plain threshold, no ring
 *   vocabulary: 1 = 2-party handshake, ring_size-1 = N-party barrier, sem_target =
 *   incremental counting. These are stock dataflow-API calls, so the op calls them
 *   directly rather than through a renamed wrapper.
 *   @warning CACHE-REUSE FOOTGUN: programs are cached and GlobalSemaphores reused, so
 *     each side must @c noc_semaphore_set(sem, 0) to re-arm for the next run — a SENDER
 *     resets BEFORE its own outgoing inc; a RECEIVER resets after its wait. Missing
 *     reset = first run green, second hangs or corrupts.
 *
 * It WRAPS, and does not reinvent, the existing fragmented fabric layer:
 *   - @c FabricConnectionManager (connection + per-direction @c WorkerToFabricEdmSender)
 *   - @c PacketHeaderPool (the idiomatic fabric-L1 packet-header allocator)
 *   - @c ccl_routing_utils (line-unicast route programming)
 *   - @c perform_payload_send and friends from @c minimal_ccl_common
 *
 * @par What the helper does NOT own (the op composes it):
 *   ring slice-walk (chip_id +/- k mod ring_size), store-and-forward relay,
 *   page<->packet coalescing/segmentation, concat-by-gather_dim output addressing,
 *   split-forwarding, address generation (TensorAccessor/ShardedAddrGen is consumed,
 *   never re-wrapped), and the all_gather fuse_op/OpSignaler matmul-fusion hooks.
 */

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

namespace dataflow_kernel_lib::ccl {

/**
 * @brief A single open fabric egress endpoint in one direction.
 *
 * Owns connection build + deferred open + close, direction selection, packet-header
 * allocation (lazy, per actual use, from @c PacketHeaderPool), 1-D unicast route
 * programming, flow-controlled unicast payload writes, and cross-device atomic-inc.
 *
 * @par Lifecycle (deferred-open mirrors point_to_point):
 *   1. construct  -> builds the connection (BUILD_AND_OPEN_CONNECTION_START_ONLY)
 *   2. [optional] noc_semaphore_wait_min(...) on a pre-open semaphore
 *   3. open()     -> open_finish() + bind the forward/backward direction
 *   4. set_route_unicast(num_hops) once (or per hop for a ring), then
 *      write_page() / inc_remote()
 *   5. close()
 *
 * Headers are allocated lazily on first use: the payload header on the first
 * set_route/write, the semaphore header on the first inc_remote. A sender that only
 * signals (e.g. the receiver's "ready" inc) therefore allocates exactly one header,
 * matching the hand-written kernel's header count.
 *
 * @note There is intentionally no symmetric "FabricStreamReceiver": the receive
 *   INGRESS is a local NoC read the op owns; a receiver's only fabric activity is a
 *   brief egress (its ack/ready inc) plus a local semaphore wait — the egress is this
 *   class, the wait is a stock @c noc_semaphore_wait_min (see the file banner).
 *
 * @par Example (point_to_point sender, abbreviated):
 * @code
 * using namespace dataflow_kernel_lib::ccl;
 * size_t conn_arg_idx = NUM_OP_RT_ARGS;                  // start of the fabric arg block
 * bool is_forward = get_arg_val<uint32_t>(conn_arg_idx); // peek has_forward (== direction here)
 * FabricStreamSender tx(conn_arg_idx, is_forward, alignment);
 *
 * noc_semaphore_wait_min(recv_sem, 1);   // wait for the receiver's "ready"
 * noc_semaphore_set(recv_sem, 0);        // reset BEFORE our own inc (cache-reuse safe)
 * tx.open();
 * tx.set_route_unicast(num_hops);
 * for (...) tx.write_page(packet_l1_addr, payload_size, packet_idx, dst);  // op owns coalescing
 * tx.inc_remote(get_noc_addr(recv_sem), 1);   // "done"
 * tx.close();
 * @endcode
 */
class FabricStreamSender {
public:
    /**
     * @brief Build the connection (deferred open) from runtime args.
     * @param conn_arg_idx  Index of the fabric arg block produced by
     *        ttnn::ccl::dataflow::append_ccl_fabric_rt_args; ADVANCED past the block.
     * @param is_forward    Send on the forward (true) or backward (false) connection.
     * @param alignment     L1 alignment used to size the on-wire payload (bytes).
     */
    FORCE_INLINE FabricStreamSender(size_t& conn_arg_idx, bool is_forward, uint32_t alignment);

    /// Finish opening the connection and bind the forward/backward direction.
    FORCE_INLINE void open();

    /// Close the connection.
    FORCE_INLINE void close();

    /**
     * @brief Program a 1-D unicast route (distance in hops) — point_to_point.
     * Routes through ccl_routing_utils::fabric_set_line_unicast_route (its
     * LowLatencyPacketHeader branch IS the raw fabric_set_unicast_route<false>).
     * Stored and reused by write_page and inc_remote; re-call to change hops (ring).
     */
    FORCE_INLINE void set_route_unicast(uint32_t num_hops);

    /**
     * @brief Unicast-write `size_bytes` from local L1 `src_l1_addr` to page `page_idx`
     *        of `dst`, then push the packet over the fabric (flow-controlled).
     * @tparam AddrGen  TensorAccessor / ShardedAddrGen (consumed, not re-wrapped).
     */
    template <class AddrGen>
    FORCE_INLINE void write_page(uint32_t src_l1_addr, uint32_t size_bytes, uint32_t page_idx, const AddrGen& dst);

    /**
     * @brief Atomic-increment a remote semaphore over the fabric (ready/done/counting).
     * Programs the stored route on a dedicated semaphore header, then flushes.
     */
    FORCE_INLINE void inc_remote(uint64_t remote_sem_noc_addr, uint32_t val = 1);

private:
    FORCE_INLINE void ensure_payload_header();

    FabricConnectionManager conn_;
    tt::tt_fabric::WorkerToFabricEdmSender* dir_ = nullptr;  // bound in open()
    volatile PACKET_HEADER_TYPE* payload_hdr_ = nullptr;     // lazy: first set_route/write
    volatile PACKET_HEADER_TYPE* sem_hdr_ = nullptr;         // lazy: first inc_remote
    uint32_t alignment_ = 0;
    bool is_forward_ = true;
    bool route_set_ = false;
    ccl_routing_utils::line_unicast_route_info_t unicast_info_{};
};

}  // namespace dataflow_kernel_lib::ccl

#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.inl"
