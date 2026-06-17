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
 * packet-header allocation, 1-D unicast route programming, the stateful
 * set_state/with_state @c UpdateMask dance, flow-controlled fabric writes, and
 * cross-device atomic-inc — so they express "arm a write channel, send these pages from
 * A to B" / "signal that peer" instead of re-deriving connection build modes, header
 * framing, routing, mask selection, and the send/flush handshake.
 *
 * This is PURE DATA MOVEMENT. No compute/unpack/math/pack appears here. Reduction
 * collectives (all_reduce, reduce_scatter) are out of scope.
 *
 * @par The armed-channel model — "arm once -> issue many".
 *   A fabric egress is a stateful packet header: you program its INVARIANT fields once
 *   (route + payload size, or route + inc value), then issue many packets that update
 *   only the VARIABLE field (the destination NOC address). This mirrors the underlying
 *   fabric @c *_set_state / @c *_with_state API and is the all_gather throughput path.
 *   FabricStreamSender exposes it as typed @c arm_* + issue pairs and OWNS the
 *   @c UpdateMask selection for each phase — the op never names an @c UpdateMask. A wrong
 *   mask silently corrupts the packet, so hiding it is the central footgun the helper
 *   removes. Call @c set_route_unicast before @c arm_*; @c arm_* before its issues.
 *
 * @par SCOPE & EXTENSION — read this first.
 *   The shipped, verified surface is the 1-D UNICAST pattern exercised end-to-end by
 *   point_to_point (the only migrated pure-DM CCL op): open a one-direction fabric
 *   egress, program a line-unicast route by hop distance, arm a unicast-write channel
 *   (fixed payload size) and a unicast atomic-inc channel (fixed value), issue page
 *   writes + the handshake/completion inc, close. It is built on the LINEAR (1-D) fabric
 *   API (@c tt_metal/fabric/hw/inc/linear/api.h), which the TT-Fabric spec guarantees
 *   builds and runs UNCHANGED on a 2-D (mesh) fabric — so a 1-D-API CCL kernel is
 *   forward-compatible to mesh hardware.
 *
 *   Richer collectives (e.g. all_gather) layer line-MULTICAST routes, real 4-chunk
 *   SCATTER writes, a multicast atomic-inc barrier, and a final fabric drain on top of
 *   this SAME armed-channel substrate, behind the same FabricStreamSender call sites:
 *   @c set_route_multicast, @c arm_scatter_write/@c write_scatter,
 *   @c arm_multicast_inc/@c multicast_inc, and @c drain. These ARE shipped and are
 *   exercised + PCC-verified by the migrated @c all_gather_async writer on a Wormhole
 *   multi-chip simulator (bit-identical to the pre-migration kernel). The local barrier
 *   wait/reset and the counting-semaphore threshold stay op-owned (see below). Worker-mux
 *   is the one fabric path NOT yet wrapped — a ConnPolicy{Direct,Mux} follow-on; the
 *   migrated writer keeps its mux path raw behind @c \#ifdef USE_WORKER_MUX.
 *
 * @par Recv-side coordination is op-owned (intentionally NOT wrapped).
 *   The receive INGRESS is a local NoC read the op already owns. Cross-device
 *   synchronization is a remote atomic-inc (@c FabricStreamSender::inc) paired with a
 *   local @c noc_semaphore_wait_min(sem, threshold) — a plain threshold, no ring
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
 *   - the @c tt::tt_fabric::linear::experimental stateful set_state/with_state fabric API
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
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

namespace dataflow_kernel_lib::ccl {

/**
 * @brief Direct fabric-connection policy (default). Wraps one FabricConnectionManager and
 *        binds a single forward/backward direction. FabricStreamSender's arm/send methods
 *        are agnostic to the policy — they call conn_.sender(); a Mux policy (MuxConn<N>,
 *        for worker-mux link sharing) slots in behind the same open()/close()/sender()
 *        interface without touching those methods.
 */
class DirectConn {
public:
    using SenderT = tt::tt_fabric::WorkerToFabricEdmSender;
    /// Build the connection (deferred open) from the fabric runtime-arg block; advances conn_arg_idx.
    FORCE_INLINE DirectConn(size_t& conn_arg_idx, bool is_forward) :
        conn_(FabricConnectionManager::build_from_args<
              FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx)),
        is_forward_(is_forward) {}
    /// Finish opening + bind the forward/backward direction.
    FORCE_INLINE void open() {
        conn_.open_finish();
        dir_ = is_forward_ ? &conn_.get_forward_connection() : &conn_.get_backward_connection();
    }
    FORCE_INLINE void close() { conn_.close(); }
    FORCE_INLINE SenderT* sender() { return dir_; }

private:
    FabricConnectionManager conn_;
    SenderT* dir_ = nullptr;
    bool is_forward_ = true;
};

/**
 * @brief Worker-mux fabric-connection policy. Many workers share one fabric link through a
 *        WorkerToFabricMuxSender<NumBuffers>, instead of DirectConn's 1:1 link<->worker bind.
 *        Slots in behind the same open()/close()/sender() interface as DirectConn, so
 *        FabricStreamSender's arm/send methods are unchanged.
 *
 * The ctor reads the mux runtime-arg block (advancing arg_idx), builds the connection, and
 * waits for the mux endpoint to be ready. A worker with no link in its direction has
 * valid==false: it builds nothing and sender() returns nullptr — the op gates sends on valid
 * targets, so it is never issued to. close() runs the mux teardown handshake: every client
 * disconnects, non-masters inc the termination-master's sync semaphore, and the master waits
 * for all clients before signalling the mux endpoint to terminate.
 *
 * @note The factory enables worker-mux (the USE_WORKER_MUX compile define + this arg block)
 *   when a link is shared by more than one worker; the compile-time mux params (buffer size,
 *   status/termination addresses, client count) are passed to the ctor from compile-time args.
 * @tparam NumBuffers  fabric_mux_num_buffers_per_channel (compile-time).
 */
template <uint8_t NumBuffers>
class MuxConn {
public:
    using SenderT = tt::tt_fabric::WorkerToFabricMuxSender<NumBuffers>;
    /**
     * @brief Read the mux runtime-arg block from arg_idx (advancing it), build the connection,
     *        and wait for the mux endpoint to become ready.
     * @param arg_idx                  Cursor at the start of the mux RT-arg block; ADVANCED past it.
     * @param channel_buffer_size_bytes  fabric_mux_channel_buffer_size_bytes (compile-time arg).
     * @param status_address           fabric_mux_status_address (compile-time arg).
     * @param termination_signal_address fabric_mux_termination_signal_address (compile-time arg).
     * @param num_mux_clients          number of workers sharing this mux (compile-time arg).
     */
    FORCE_INLINE MuxConn(
        size_t& arg_idx,
        size_t channel_buffer_size_bytes,
        size_t status_address,
        size_t termination_signal_address,
        uint32_t num_mux_clients) :
        termination_signal_address_(termination_signal_address), num_mux_clients_(num_mux_clients) {
        valid_ = get_arg_val<uint32_t>(arg_idx++) == 1;
        is_termination_master_ = get_arg_val<uint32_t>(arg_idx++);
        mux_x_ = get_arg_val<uint32_t>(arg_idx++);
        mux_y_ = get_arg_val<uint32_t>(arg_idx++);
        const size_t channel_base_address = get_arg_val<uint32_t>(arg_idx++);
        const size_t connection_info_address = get_arg_val<uint32_t>(arg_idx++);
        const size_t connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
        const size_t flow_control_address = get_arg_val<uint32_t>(arg_idx++);
        const size_t buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
        const uint8_t channel_id = get_arg_val<uint32_t>(arg_idx++);
        termination_sync_address_ = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        const uint32_t local_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        termination_master_noc_x_ = get_arg_val<uint32_t>(arg_idx++);
        termination_master_noc_y_ = get_arg_val<uint32_t>(arg_idx++);
        if (valid_) {
            mux_ = tt::tt_fabric::build_connection_to_fabric_endpoint<NumBuffers>(
                mux_x_,
                mux_y_,
                channel_id,
                NumBuffers,
                channel_buffer_size_bytes,
                channel_base_address,
                connection_info_address,
                connection_handshake_address,
                flow_control_address,
                buffer_index_address,
                local_flow_control_address,
                local_teardown_address,
                local_buffer_index_address);
            // The mux endpoint is a separate kernel; block until it is ready to accept connections.
            tt::tt_fabric::wait_for_fabric_endpoint_ready(mux_x_, mux_y_, status_address, local_status_address);
        }
    }
    /// Connect to the mux endpoint (no-op for a worker with no link in its direction).
    FORCE_INLINE void open() {
        if (valid_) {
            tt::tt_fabric::fabric_client_connect(mux_);
        }
    }
    /// Disconnect, then the mux termination handshake (master waits for all clients then signals
    /// the mux to terminate; non-masters inc the master's sync semaphore). No-op if not valid.
    FORCE_INLINE void close() {
        if (!valid_) {
            return;
        }
        tt::tt_fabric::fabric_client_disconnect(mux_);
        if (is_termination_master_) {
            auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address_);
            noc_semaphore_wait(termination_sync_ptr, num_mux_clients_ - 1);
            tt::tt_fabric::fabric_endpoint_terminate(mux_x_, mux_y_, termination_signal_address_);
        } else {
            const uint64_t dest_addr =
                safe_get_noc_addr(termination_master_noc_x_, termination_master_noc_y_, termination_sync_address_, 0);
            noc_semaphore_inc(dest_addr, 1);
            noc_async_atomic_barrier();
        }
    }
    FORCE_INLINE SenderT* sender() { return valid_ ? &mux_ : nullptr; }

private:
    SenderT mux_;
    bool valid_ = false;
    bool is_termination_master_ = false;
    uint8_t mux_x_ = 0;
    uint8_t mux_y_ = 0;
    size_t termination_signal_address_ = 0;
    uint32_t termination_sync_address_ = 0;
    uint32_t termination_master_noc_x_ = 0;
    uint32_t termination_master_noc_y_ = 0;
    uint32_t num_mux_clients_ = 0;
};

/**
 * @brief A single open fabric egress endpoint in one direction, exposing armed channels.
 *
 * Owns connection build + deferred open + close, direction selection, packet-header
 * allocation (lazy, per armed channel, from @c PacketHeaderPool), 1-D unicast route
 * programming, the stateful @c UpdateMask selection, flow-controlled unicast payload
 * writes, and cross-device atomic-inc.
 *
 * @par Lifecycle (deferred-open mirrors point_to_point):
 *   1. construct  -> builds the connection (BUILD_AND_OPEN_CONNECTION_START_ONLY)
 *   2. [optional] noc_semaphore_wait_min(...) on a pre-open semaphore
 *   3. open()                       -> open_finish() + bind the forward/backward direction
 *   4. set_route_unicast(num_hops)
 *   5. arm_unicast_write(page_size) / arm_inc(val)  -- program invariant header state once
 *   6. write()/write_page() / inc()                 -- issue many, varying only the dst addr
 *   7. close()
 *
 * Each armed channel lazily allocates its own header from @c PacketHeaderPool on the arm
 * call: the payload header on @c arm_unicast_write, the semaphore header on @c arm_inc. A
 * sender that only signals (e.g. the receiver's "ready" inc) therefore arms one channel
 * and allocates exactly one header, matching the hand-written kernel's header count.
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
 * tx.arm_unicast_write(payload_size);    // invariant: payload size + route, once
 * tx.arm_inc(1);                         // invariant: inc value + route, once
 * for (...) tx.write_page(packet_l1_addr, packet_idx, dst);  // op owns coalescing; varies dst addr
 * tx.inc(get_noc_addr(recv_sem));        // "done"
 * tx.close();
 * @endcode
 */
template <typename ConnT = DirectConn>
class FabricStreamSender {
public:
    /**
     * @brief Convenience ctor for the default DirectConn policy: build the connection
     *        (deferred open) from runtime args. Advances conn_arg_idx past the fabric block.
     * @param conn_arg_idx  Index of the fabric arg block produced by
     *        ttnn::ccl::dataflow::append_ccl_fabric_rt_args; ADVANCED past the block.
     * @param is_forward    Send on the forward (true) or backward (false) connection.
     * @param alignment     L1 alignment used to size the on-wire payload (bytes).
     */
    FORCE_INLINE FabricStreamSender(size_t& conn_arg_idx, bool is_forward, uint32_t alignment);

    /**
     * @brief Construct from a pre-built connection policy (e.g. MuxConn<N>) + alignment.
     *        The policy already read its own args; open()/close()/sender() route through it.
     */
    FORCE_INLINE FabricStreamSender(ConnT conn, uint32_t alignment);

    /// Finish opening the connection and bind the forward/backward direction.
    FORCE_INLINE void open();

    /// Close the connection.
    FORCE_INLINE void close();

    /**
     * @brief Program a 1-D unicast route (distance in hops) — point_to_point.
     * Stored and applied to each subsequent armed channel's header via
     * ccl_routing_utils::fabric_set_line_unicast_route. Call BEFORE arm_*.
     */
    FORCE_INLINE void set_route_unicast(uint32_t num_hops);

    /**
     * @brief Program a 1-D unicast route from a precomputed route info — the form
     * all_gather uses (it reads the route from compile-time args). Equivalent to the
     * num_hops overload. Call BEFORE arm_*.
     */
    FORCE_INLINE void set_route_unicast(const ccl_routing_utils::line_unicast_route_info_t& info);

    // --- Armed unicast-write channel -------------------------------------------------
    /**
     * @brief Arm the unicast-write channel: program the invariant route + on-wire payload
     *        size onto a dedicated packet header once (set_state). Helper owns the
     *        @c UpdateMask. Call after set_route_unicast, then issue write()/write_page().
     */
    FORCE_INLINE void arm_unicast_write(uint32_t page_size_bytes);

    /**
     * @brief Issue one armed unicast write of the armed payload size from local L1
     *        @c src_l1_addr to @c dst_noc_addr (with_state, varying only the dst addr).
     *        Requires a prior arm_unicast_write + open.
     */
    FORCE_INLINE void write(uint64_t dst_noc_addr, uint32_t src_l1_addr);

    /**
     * @brief Convenience over write(): compute the destination NOC address for page
     *        @c page_idx of @c dst (a consumed TensorAccessor/ShardedAddrGen) and issue
     *        an armed unicast write. Keeps the addrgen-friendly entry point.
     * @tparam AddrGen  TensorAccessor / ShardedAddrGen (consumed, not re-wrapped).
     */
    template <class AddrGen>
    FORCE_INLINE void write_page(uint32_t src_l1_addr, uint32_t page_idx, const AddrGen& dst);

    // --- Armed unicast atomic-inc channel --------------------------------------------
    /**
     * @brief Arm the atomic-inc channel: program the invariant route + increment value
     *        (+ flush) onto a dedicated header once (set_state). Helper owns the
     *        @c UpdateMask. Call after set_route_unicast, then issue inc().
     */
    FORCE_INLINE void arm_inc(uint32_t val = 1);

    /**
     * @brief Atomic-increment a remote semaphore over the fabric by the armed value
     *        (ready / done / counting), varying only the semaphore address (with_state).
     *        Requires a prior arm_inc + open.
     */
    FORCE_INLINE void inc(uint64_t remote_sem_noc_addr);

    // --- Multicast route (for the N-party barrier; e.g. all_gather) -----------------
    /**
     * @brief Program a 1-D line-MULTICAST route (start distance + range, in hops).
     * Stored and applied to the multicast atomic-inc channel by arm_multicast_inc. Route
     * info comes from the host (ttnn::ccl::get_forward_backward_line_mcast_*). Call before
     * arm_multicast_inc.
     */
    FORCE_INLINE void set_route_multicast(const ccl_routing_utils::line_multicast_route_info_t& info);

    // --- Armed scatter-write channel (<=4 chunks/packet) ----------------------------
    /**
     * @brief Arm the scatter-write channel: program the invariant per-chunk sizes + chunk
     *        count + on-wire payload size onto a dedicated header once (set_state). Helper
     *        owns the ChunkSizes|PayloadSize mask. Call after set_route_unicast; then issue
     *        write_scatter().
     * @param chunk_size_bytes  Per-chunk (per-tile) payload size.
     * @param num_chunks        Chunks per packet (2..4; the NocUnicastScatterCommandHeader limit).
     */
    FORCE_INLINE void arm_scatter_write(uint32_t chunk_size_bytes, uint32_t num_chunks);

    /**
     * @brief Issue one armed scatter write: pack up to 4 destination NOC addresses into one
     *        packet from local L1 @c src_l1_addr (with_state, DstAddrs|ChunkSizes|PayloadSize).
     *        @c num_chunks must match the arm. Requires a prior arm_scatter_write + open.
     */
    FORCE_INLINE void write_scatter(const uint64_t* dst_noc_addrs, uint32_t num_chunks, uint32_t src_l1_addr);

    // --- Armed multicast atomic-inc channel (the N-party barrier) --------------------
    /**
     * @brief Arm the multicast atomic-inc channel: program the invariant increment value
     *        (+ flush) + the multicast route onto the inc header once (set_state, Val|Flush).
     *        Call after set_route_multicast; then issue multicast_inc().
     * @note Reuses the SAME Pool header as arm_inc — matching all_gather, which reuses one
     *       sem-inc header for the barrier-multicast phase then re-arms it for the per-chunk
     *       unicast incs. Arm/issue the barrier (multicast) phase fully before re-arming
     *       with arm_inc for the unicast counting phase.
     */
    FORCE_INLINE void arm_multicast_inc(uint32_t val = 1);

    /**
     * @brief Multicast atomic-increment @c remote_sem_noc_addr to all peers on the armed
     *        multicast route by the armed value (with_state, DstAddr). The matching local
     *        barrier wait/reset (noc_semaphore_wait_min(sem, ring_size-1) + set 0) stays
     *        op-owned — see the file banner. Requires a prior arm_multicast_inc + open.
     */
    FORCE_INLINE void multicast_inc(uint64_t remote_sem_noc_addr);

    // --- Final fabric drain ----------------------------------------------------------
    /**
     * @brief Drain outstanding local NoC writes + fabric atomic-incs before close
     *        (noc_async_write_barrier + noc_async_atomic_barrier). all_gather ends with this;
     *        p2p doesn't need it (close() drains its single trailing inc).
     */
    FORCE_INLINE void drain();

private:
    ConnT conn_;                                          // connection policy (Direct/Mux); owns open/close/sender()
    volatile PACKET_HEADER_TYPE* payload_hdr_ = nullptr;  // armed by arm_unicast_write
    volatile PACKET_HEADER_TYPE* scatter_hdr_ = nullptr;  // armed by arm_scatter_write
    volatile PACKET_HEADER_TYPE* sem_hdr_ = nullptr;      // armed by arm_inc / arm_multicast_inc (shared)
    uint32_t alignment_ = 0;
    uint32_t scatter_chunk_size_ = 0;  // per-chunk size armed by arm_scatter_write
    ccl_routing_utils::line_unicast_route_info_t unicast_info_{};
    ccl_routing_utils::line_multicast_route_info_t multicast_info_{};
};

}  // namespace dataflow_kernel_lib::ccl

#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.inl"
