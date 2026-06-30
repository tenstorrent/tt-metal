// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file ccl_helpers_dataflow.hpp
 * @brief Multi-device CCL (fabric) dataflow-kernel helpers — a safety-by-construction API.
 *
 * The multi-device analog of the single-device dataflow-helper library (#45698,
 * @c reduce/dfb/tilize_helpers_dataflow). It gives op authors an intent-level surface for the
 * footgun-heavy fabric egress plumbing — connection lifecycle + direction, packet-header
 * allocation, 1-D route programming, the stateful set_state/with_state @c UpdateMask dance,
 * flow-controlled fabric writes, and cross-device atomic-inc.
 *
 * This is PURE DATA MOVEMENT. No compute/unpack/math/pack appears here. Reduction collectives
 * (all_reduce, reduce_scatter) are out of scope.
 *
 * @par Safety by construction — the call order IS the type progression.
 *   The legal fabric-egress sequence is open(route) -> arm -> issue -> close. Rather than
 *   document that order and trust callers, this API makes each stage a distinct type that
 *   exposes ONLY the operations legal at that stage, so a mis-ordered sequence fails to compile:
 *
 *     FabricStreamSender<ConnT>      // UNOPENED: open(route) (the stream), or signal() (one-shot).
 *          | open(route)  -> open_finish() + bind the direction + BIND THE STREAM'S ROUTE
 *          v
 *     FabricStream<ConnT>            // OPENED: arm_*(...), drain(), close().
 *          | arm_unicast_write(page_size)             -> UnicastWriteChannel
 *          | arm_scatter_write(chunk, n)              -> ScatterWriteChannel
 *          | arm_inc(val)                             -> AtomicIncChannel
 *          | arm_multicast_inc(mcast_route, val)      -> MulticastIncChannel
 *          v
 *     <channel handle>               // ARMED: the issue methods, and nothing else.
 *          write()/write_page() | write_scatter() | inc() | multicast_inc()
 *
 *   What this rules out at compile time:
 *     1. arm or issue before open() — arm_* live only on FabricStream, which only open() yields.
 *     2. arm without a route — the route is bound ONCE at open(route) and reused by every arm_*,
 *        so an unrouted send cannot be written and a stream's channels cannot disagree on the
 *        route. A wrong/absent route silently corrupts the packet, so binding it un-omittably at
 *        the stream is the central footgun this API removes. (arm_multicast_inc takes its own
 *        multicast route, since that is a different cast mode than the stream's unicast route.)
 *     3. issue before arm — write()/inc()/etc. exist only on the handle arm_* returns; you
 *        cannot name an issue without first holding an armed channel.
 *     4. forgot close()/drain() — close() DRAINS (write + atomic barriers) then closes; it is
 *        idempotent and the FabricStream destructor closes if you did not. drain() stays callable
 *        for an explicit mid-stream flush, but the teardown drain is automatic.
 *
 * @par One-shot convenience — FabricStreamSender::signal().
 *   The common "send exactly one atomic-inc over the fabric, then tear down" (a ready/done
 *   handshake) is a single call: @c signal(route_or_hops, remote_sem_noc_addr) opens, arms the
 *   inc, issues it, and closes — the whole open/arm/issue/close sequence collapsed. Use the staged
 *   open()->arm_*->issue path when a stream issues MANY packets across a loop.
 *
 * @par The armed-channel model — "arm once -> issue many".
 *   A fabric egress is a stateful packet header: arm_* programs its INVARIANT fields once
 *   (the stream's route + payload size, or route + inc value) via set_state and OWNS the
 *   @c UpdateMask; the returned channel issues many packets that update only the VARIABLE field
 *   (the destination NOC address) via with_state. The op never names an @c UpdateMask.
 *
 * @par SCOPE & EXTENSION.
 *   Shipped + verified: the 1-D UNICAST pattern (point_to_point) and the line-MULTICAST barrier
 *   + 4-chunk SCATTER + final drain layered on the same channels (all_gather_async, PCC-verified
 *   on a Wormhole multi-chip simulator). Built on the LINEAR (1-D) fabric API
 *   (@c tt_metal/fabric/hw/inc/linear/api.h), which the TT-Fabric spec guarantees runs UNCHANGED
 *   on a 2-D (mesh) fabric. Worker-mux is wrapped via the ConnT policy (MuxConn<N>); see below.
 *
 * @par Cross-device coordination is split (intentionally).
 *   The SENDING half of a cross-device sync — a remote atomic-inc — is owned here
 *   (AtomicIncChannel::inc / MulticastIncChannel::multicast_inc). The WAITING half is a plain
 *   local @c noc_semaphore_wait_min(sem, threshold) the op calls directly (1 = handshake,
 *   ring_size-1 = N-party barrier, sem_target = counting) — a stock dataflow call, not renamed.
 *   The receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver.
 *   @warning CACHE-REUSE FOOTGUN: programs are cached and GlobalSemaphores reused, so each side
 *     must @c noc_semaphore_set(sem, 0) to re-arm — a SENDER resets BEFORE its outgoing inc, a
 *     RECEIVER after its wait. Missing reset = first run green, second hangs or corrupts.
 *   @note Each arm_* draws its OWN pooled header (unicast-write, scatter, unicast-inc, and
 *     multicast-inc are independent), so any mix of channels may be live at once with no ordering
 *     constraint between them. The pool holds several headers per RISC (8 on Wormhole/Blackhole)
 *     and is reset every kernel launch; a stream arms at most four, well within budget.
 *
 * It WRAPS, and does not reinvent, the existing fragmented fabric layer:
 *   - @c FabricConnectionManager (connection + per-direction @c WorkerToFabricEdmSender)
 *   - @c PacketHeaderPool (the idiomatic fabric-L1 packet-header allocator)
 *   - @c ccl_routing_utils (line-unicast / line-multicast route programming)
 *   - the @c tt::tt_fabric::linear::experimental stateful set_state/with_state fabric API
 *
 * @par What the helper does NOT own (the op composes it):
 *   ring slice-walk (chip_id +/- k mod ring_size), store-and-forward relay, page<->packet
 *   coalescing/segmentation, concat-by-gather_dim output addressing, split-forwarding, address
 *   generation (TensorAccessor/ShardedAddrGen is consumed, never re-wrapped), the local barrier
 *   wait/reset, and the all_gather fuse_op/OpSignaler matmul-fusion hooks.
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
 *        binds a single forward/backward direction. The arm/send methods are agnostic to the
 *        policy — they call conn_.sender(); a Mux policy (MuxConn<N>, for worker-mux link
 *        sharing) slots in behind the same open()/close()/sender() interface.
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
 *        Slots in behind the same open()/close()/sender() interface as DirectConn, so the
 *        FabricStream's arm/send methods are unchanged.
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

// Forward declarations: FabricStream constructs the channel handles (their ctors are private,
// FabricStream is their friend); FabricStreamSender constructs FabricStream.
template <typename ConnT>
class FabricStream;
template <typename ConnT>
class FabricStreamSender;

/// Build a 1-D unicast route info from a hop distance — the point_to_point convenience form.
/// (all_gather reads its route info from compile-time args and passes it directly.) 1-D linear
/// routing is intra-mesh and hop-distance based; dst_mesh_id is unused on the LowLatency path.
FORCE_INLINE ccl_routing_utils::line_unicast_route_info_t unicast_route(uint32_t num_hops) {
    ccl_routing_utils::line_unicast_route_info_t info{};
    info.dst_mesh_id = 0;
    info.distance_in_hops = static_cast<uint16_t>(num_hops);
    return info;
}

// ============================================================================================
// Armed channel handles — each is produced by a FabricStream::arm_* call and exposes ONLY the
// issues for its send type. Holding one is the compile-time proof that arm (and therefore open
// + route) happened. Each borrows the connection (owned by the FabricStreamSender) and the
// pooled header (owned by the FabricStream); both outlive every channel by construction.
// ============================================================================================

/// Armed unicast-write channel: issue armed-size payload writes, varying only the dst address.
template <typename ConnT>
class UnicastWriteChannel {
public:
    /// Issue one armed unicast write of the armed payload size from local L1 @c src_l1_addr to
    /// @c dst_noc_addr (with_state — varies only the dst address).
    FORCE_INLINE void write(uint64_t dst_noc_addr, uint32_t src_l1_addr);
    /// Convenience over write(): compute the dst NOC address for page @c page_idx of @c dst (a
    /// consumed TensorAccessor/ShardedAddrGen) and issue an armed unicast write.
    template <class AddrGen>
    FORCE_INLINE void write_page(uint32_t src_l1_addr, uint32_t page_idx, const AddrGen& dst);

private:
    friend class FabricStream<ConnT>;
    FORCE_INLINE UnicastWriteChannel(ConnT* conn, volatile PACKET_HEADER_TYPE* hdr) : conn_(conn), hdr_(hdr) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_;
};

/// Armed scatter-write channel: issue <=4-destination packets (the NocUnicastScatter limit).
template <typename ConnT>
class ScatterWriteChannel {
public:
    /// Issue one armed scatter write: pack up to 4 destination NOC addresses into one packet from
    /// local L1 @c src_l1_addr (with_state — DstAddrs|ChunkSizes|PayloadSize, since the last packet
    /// of a run may carry fewer chunks than the armed maximum). @c num_chunks must be <= the arm.
    FORCE_INLINE void write_scatter(const uint64_t* dst_noc_addrs, uint32_t num_chunks, uint32_t src_l1_addr);

private:
    friend class FabricStream<ConnT>;
    FORCE_INLINE ScatterWriteChannel(ConnT* conn, volatile PACKET_HEADER_TYPE* hdr, uint32_t chunk_size_bytes) :
        conn_(conn), hdr_(hdr), chunk_size_bytes_(chunk_size_bytes) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_;
    uint32_t chunk_size_bytes_;
};

/// Armed unicast atomic-inc channel: increment a remote semaphore by the armed value over fabric.
template <typename ConnT>
class AtomicIncChannel {
public:
    /// Atomic-increment a remote semaphore over the fabric by the armed value (ready / done /
    /// counting), varying only the semaphore address (with_state).
    FORCE_INLINE void inc(uint64_t remote_sem_noc_addr);

private:
    friend class FabricStream<ConnT>;
    FORCE_INLINE AtomicIncChannel(ConnT* conn, volatile PACKET_HEADER_TYPE* hdr) : conn_(conn), hdr_(hdr) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_;
};

/// Armed multicast atomic-inc channel (the N-party barrier): increment a semaphore on all peers
/// on the armed multicast route by the armed value. The matching local barrier wait/reset
/// (noc_semaphore_wait_min(sem, ring_size-1) + set 0) stays op-owned. Draws its own pooled header,
/// independent of the unicast AtomicIncChannel — the two may be live at once in any order.
template <typename ConnT>
class MulticastIncChannel {
public:
    /// Multicast atomic-increment @c remote_sem_noc_addr to all peers on the armed route by the
    /// armed value (with_state — varies only the dst address).
    FORCE_INLINE void multicast_inc(uint64_t remote_sem_noc_addr);

private:
    friend class FabricStream<ConnT>;
    FORCE_INLINE MulticastIncChannel(ConnT* conn, volatile PACKET_HEADER_TYPE* hdr) : conn_(conn), hdr_(hdr) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_;
};

// ============================================================================================
// FabricStream — the OPENED egress. Owns the (lazy) pooled headers and the alignment; hands out
// armed channels. Borrows the connection from the FabricStreamSender that produced it (so the
// sender must outlive the stream — declare the sender first). RAII-closes on destruction.
// ============================================================================================
template <typename ConnT = DirectConn>
class FabricStream {
public:
    FabricStream(const FabricStream&) = delete;
    FabricStream& operator=(const FabricStream&) = delete;
    /// Move ctor: open() returns a FabricStream by value. C++17 guaranteed copy elision usually
    /// constructs it in place, but provide a move that transfers `closed_` so the moved-from
    /// stream never double-closes the (now transferred) connection.
    FORCE_INLINE FabricStream(FabricStream&& o) :
        conn_(o.conn_),
        alignment_(o.alignment_),
        route_(o.route_),
        payload_hdr_(o.payload_hdr_),
        scatter_hdr_(o.scatter_hdr_),
        sem_hdr_(o.sem_hdr_),
        mcast_hdr_(o.mcast_hdr_),
        closed_(o.closed_) {
        o.closed_ = true;
    }
    FabricStream& operator=(FabricStream&&) = delete;
    FORCE_INLINE ~FabricStream() { close(); }  // RAII backstop; idempotent with explicit close()

    // --- Armed unicast-write channel -------------------------------------------------
    /// Arm the unicast-write channel: program the stream's route + on-wire payload size onto a
    /// pooled header once (set_state). Helper owns the @c UpdateMask. Returns the channel to write.
    FORCE_INLINE UnicastWriteChannel<ConnT> arm_unicast_write(uint32_t page_size_bytes);

    // --- Armed scatter-write channel (<=4 chunks/packet) ----------------------------
    /// Arm the scatter-write channel: program the stream's route + per-chunk sizes + chunk count +
    /// payload size onto a pooled header once (set_state, ChunkSizes|PayloadSize). Returns it.
    /// @param chunk_size_bytes  Per-chunk (per-tile) payload size.
    /// @param num_chunks        Chunks per packet (2..4).
    FORCE_INLINE ScatterWriteChannel<ConnT> arm_scatter_write(uint32_t chunk_size_bytes, uint32_t num_chunks);

    // --- Armed unicast atomic-inc channel --------------------------------------------
    /// Arm the unicast atomic-inc channel: program the stream's route + increment value (+ flush)
    /// onto a pooled header once (set_state, Val|Flush). Returns the channel to issue inc()s.
    FORCE_INLINE AtomicIncChannel<ConnT> arm_inc(uint32_t val = 1);

    // --- Armed multicast atomic-inc channel (the N-party barrier) --------------------
    /// Arm the multicast atomic-inc channel: program a MULTICAST route (its own, distinct from the
    /// stream's unicast route) + increment value (+ flush) onto a dedicated pooled header once
    /// (set_state, Val|Flush). Returns the channel. Independent of arm_inc's header.
    FORCE_INLINE MulticastIncChannel<ConnT> arm_multicast_inc(
        const ccl_routing_utils::line_multicast_route_info_t& route, uint32_t val = 1);

    // --- Lifecycle -------------------------------------------------------------------
    /// Drain outstanding local NoC writes + fabric atomic-incs (noc_async_write_barrier +
    /// noc_async_atomic_barrier). Optional — close() drains automatically; call this only for an
    /// explicit mid-stream flush before more issues.
    FORCE_INLINE void drain();
    /// Drain, then close the connection. Idempotent — safe to call explicitly and again from the
    /// destructor (the RAII backstop).
    FORCE_INLINE void close();

private:
    friend class FabricStreamSender<ConnT>;
    FORCE_INLINE FabricStream(
        ConnT* conn, uint32_t alignment, const ccl_routing_utils::line_unicast_route_info_t& route) :
        conn_(conn), alignment_(alignment), route_(route) {}

    ConnT* conn_;                                         // borrowed from the FabricStreamSender
    uint32_t alignment_;                                  // L1 alignment for on-wire payload sizing
    ccl_routing_utils::line_unicast_route_info_t route_;  // bound at open(); reused by every unicast arm_*
    volatile PACKET_HEADER_TYPE* payload_hdr_ = nullptr;  // lazily allocated by arm_unicast_write
    volatile PACKET_HEADER_TYPE* scatter_hdr_ = nullptr;  // lazily allocated by arm_scatter_write
    volatile PACKET_HEADER_TYPE* sem_hdr_ = nullptr;      // lazily allocated by arm_inc
    volatile PACKET_HEADER_TYPE* mcast_hdr_ = nullptr;    // lazily allocated by arm_multicast_inc (independent)
    bool closed_ = false;
};

// ============================================================================================
// FabricStreamSender — the UNOPENED egress. Owns the connection policy. open(route) finishes the
// connection and yields the FabricStream; signal() is the one-shot "send one inc then close"
// shortcut. Construct it, optionally do a pre-open noc_semaphore_wait_min, then open() or signal().
// ============================================================================================
template <typename ConnT = DirectConn>
class FabricStreamSender {
public:
    /**
     * @brief Convenience ctor for the default DirectConn policy: build the connection (deferred
     *        open) from runtime args. Advances conn_arg_idx past the fabric block.
     * @param conn_arg_idx  Index of the fabric arg block produced by
     *        ttnn::ccl::dataflow::append_ccl_fabric_rt_args; ADVANCED past the block.
     * @param is_forward    Send on the forward (true) or backward (false) connection.
     * @param alignment     L1 alignment used to size the on-wire payload (bytes).
     */
    FORCE_INLINE FabricStreamSender(size_t& conn_arg_idx, bool is_forward, uint32_t alignment) :
        conn_(conn_arg_idx, is_forward), alignment_(alignment) {}

    /// Construct from a pre-built connection policy (e.g. MuxConn<N>, which read its own args).
    FORCE_INLINE FabricStreamSender(ConnT conn, uint32_t alignment) : conn_(conn), alignment_(alignment) {}

    FabricStreamSender(const FabricStreamSender&) = delete;
    FabricStreamSender& operator=(const FabricStreamSender&) = delete;

    /// Finish opening the connection + bind the direction, bind the stream's unicast @c route, and
    /// yield the opened FabricStream. Every unicast arm_* reuses this route. The returned stream
    /// borrows this sender's connection, so this sender must outlive it.
    FORCE_INLINE FabricStream<ConnT> open(const ccl_routing_utils::line_unicast_route_info_t& route) {
        conn_.open();
        return FabricStream<ConnT>(&conn_, alignment_, route);
    }

    /// One-shot: send exactly one fabric atomic-inc of @c val to @c remote_sem_noc_addr along
    /// @c route, then tear down. Collapses open() -> arm_inc() -> inc() -> close() for the common
    /// ready/done handshake. Terminal — do not also call open() on this sender afterwards.
    FORCE_INLINE void signal(
        const ccl_routing_utils::line_unicast_route_info_t& route, uint64_t remote_sem_noc_addr, uint32_t val = 1);
    /// signal() convenience taking a hop distance instead of a route info.
    FORCE_INLINE void signal(uint32_t num_hops, uint64_t remote_sem_noc_addr, uint32_t val = 1) {
        signal(unicast_route(num_hops), remote_sem_noc_addr, val);
    }

private:
    ConnT conn_;
    uint32_t alignment_;
};

}  // namespace dataflow_kernel_lib::ccl

#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.inl"
