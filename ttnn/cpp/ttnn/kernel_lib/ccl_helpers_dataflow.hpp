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
 * This is PURE DATA MOVEMENT. No compute/unpack/math/pack appears here — but it is NOT restricted
 * to pure-data-movement OPS. Reduction and fused collectives (all_reduce, reduce_scatter,
 * rms_allgather, the fused all_gather+matmul family) reach the fabric through exactly the same
 * egress plumbing; their COMPUTE kernels are untouched by this header, and their DATAFLOW kernels
 * are first-class consumers. What those ops add over point_to_point / all_gather is three egress
 * shapes, all covered below: DUPLEX egress (one connection driving forward AND backward from a
 * single core), FUSED write+atomic-inc (payload plus the receiver's semaphore bump in ONE packet),
 * and chip-level MULTICAST of a payload.
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
 *          | arm_fused_write_inc(page_size, val)      -> FusedWriteIncChannel
 *          | arm_multicast_write(mcast_route, size)   -> MulticastWriteChannel
 *          v
 *     <channel handle>               // ARMED: the issue methods, and nothing else.
 *          write()/write_page() | write_scatter() | inc() | multicast_inc() | write_fused()
 *
 *   The DUPLEX tier — what the reduction collectives need — is the SAME progression over a doubled
 *   egress. One FabricConnectionManager owns both directions; every issue fans out to each
 *   CONNECTED direction, each with its own route and its own pooled header:
 *
 *     FabricDuplexSender<ConnT>      // UNOPENED: open(fwd_route, bwd_route).
 *          | open(fwd, bwd)  -> open_finish() + BIND BOTH DIRECTIONS' ROUTES. The route-pair TYPE
 *          |                    picks the chip-level cast: a unicast pair yields a
 *          |                    Cast::Unicast stream, a multicast pair a Cast::Multicast one.
 *          v
 *     FabricDuplexStream<Cast, ConnT>  // OPENED: arm_*(...), drain(), close().
 *          | arm_write(page_size)                     -> DuplexWriteChannel
 *          | arm_fused_write_inc(page_size, val)      -> DuplexFusedWriteIncChannel
 *          | arm_scatter_write(chunk, n)              -> DuplexScatterWriteChannel
 *          | arm_inc(val)                             -> DuplexIncChannel
 *          v
 *     <duplex channel handle>        // ARMED: each issue fans out to every connected direction.
 *          write() | write_with_local_copy() | write_fused() | write_scatter() | inc()
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
 *   The DUPLEX + FUSED + MULTICAST-payload tier extends the same model to the reduction and fused
 *   collectives, replacing the hand-rolled free functions in
 *   @c ccl/common/kernels/minimal_ccl_common.hpp (@c write_and_advance_local_read_address_for_fabric_write,
 *   @c fused_write_atomic_and_advance_local_read_address_for_fabric_write,
 *   @c scatter_write_and_advance_local_read_address_for_fabric_write and their addrgen overloads).
 *   Those free functions take raw header pointers and a @c FabricConnectionManager& and re-derive the
 *   has_forward/has_backward gating at EVERY call site, which is precisely the footgun surface this
 *   API removes: here the direction set is resolved once at open() and a mis-ordered or unrouted
 *   send does not compile. They also use the NON-stateful @c to_noc_* header writers (a full header
 *   re-program per packet); the duplex channels use set_state/with_state like the rest of this API.
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
 *     and is reset every kernel launch; a unidirectional stream arms at most six, well within
 *     budget. A DUPLEX channel draws TWO headers (one per direction), so a duplex stream arming
 *     all FOUR of its channels draws eight — the whole per-RISC budget on Wormhole/Blackhole. A duplex
 *     op therefore should not arm channels it never issues on, and one that also needs unidirectional
 *     channels must count them. Ops that previously carved
 *     packet headers out of their OWN reserved CB (the reduction collectives' historical
 *     @c reserved_packet_header_cb_id) no longer need to: the pool owns header storage, and that CB
 *     plus its @c num_packet_headers_storable arg become dead once the op is migrated.
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
 *   wait/reset, and the all_gather fuse_op/OpSignaler matmul-fusion hooks. For the reduction
 *   collectives it additionally does NOT own: the REDUCTION itself (that is the op's compute
 *   kernel — this header never touches unpack/math/pack), which slice a worker reduces vs forwards,
 *   the reduction-worker mcast/semaphore fan-out on the LOCAL chip (a stock local NoC mcast), and
 *   the L1 read cursor. @c write_with_local_copy mirrors one payload to the local chip because the
 *   local and fabric destinations are the same logical address and splitting them invites drift,
 *   but the op still advances its own cursor — consistent with the op owning coalescing.
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
    FORCE_INLINE void open() { open_finish(); }
    /// No-op: the ctor's BUILD_AND_OPEN_CONNECTION_START_ONLY already started this handshake, so a
    /// DirectConn is always mid-open by construction. Present so FabricStreamSender's split-open path
    /// works uniformly across connection policies (MuxConn::open_start() does real work).
    FORCE_INLINE void open_start() {}
    /// Complete the handshake + bind the direction.
    FORCE_INLINE void open_finish() {
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

    /**
     * @brief Split open — start the mux connection handshake WITHOUT waiting for it.
     *
     * @c open() is the combined blocking form. The mux handshake has real latency, and a worker that
     * has independent setup to do (address-generator construction, a local semaphore reset, reading
     * the rest of its runtime args) can overlap that work with the handshake by calling
     * @c open_start(), doing the work, then @c open_finish() before the first issue.
     * @c line_reduce_scatter_minimal_async_writer hand-rolled exactly this with
     * @c fabric_client_connect_start / @c _finish, which is why it could not adopt the helper while
     * only the combined form existed.
     *
     * @warning @c open_finish() MUST be called before any issue on a stream opened this way. Nothing
     *   in the type system enforces the pairing — unlike the sender->stream->channel progression, both
     *   halves live on the connection policy, below the stage types. Prefer @c open() unless the
     *   overlap is measurable.
     */
    FORCE_INLINE void open_start() {
        if (valid_) {
            tt::tt_fabric::fabric_client_connect_start(mux_);
        }
    }
    /// Complete a handshake begun by @c open_start(). No-op for a worker with no link.
    FORCE_INLINE void open_finish() {
        if (valid_) {
            tt::tt_fabric::fabric_client_connect_finish(mux_);
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

/**
 * @brief Armed FUSED unicast-write + atomic-inc channel — the reduction collectives' workhorse.
 *
 * One packet carries the payload AND bumps a semaphore on the receiving chip, so a reduction worker
 * never needs a second inc packet to announce "your input slice has landed". The armed invariants
 * are the inc value, the flush flag and the payload size; each issue varies the payload destination
 * and the semaphore address.
 *
 * @note The receiver-side wait stays op-owned (a local @c noc_semaphore_wait_min), exactly as for
 *   the non-fused AtomicIncChannel — the split documented in the banner is unchanged.
 */
template <typename ConnT>
class FusedWriteIncChannel {
public:
    /// Issue one armed fused write+inc: land the armed payload size at @c dst_noc_addr and atomically
    /// bump @c remote_sem_noc_addr by the armed value (with_state — varies dst + semaphore address).
    FORCE_INLINE void write_fused(uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr);

private:
    friend class FabricStream<ConnT>;
    FORCE_INLINE FusedWriteIncChannel(ConnT* conn, volatile PACKET_HEADER_TYPE* hdr) : conn_(conn), hdr_(hdr) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_;
};

/**
 * @brief Armed chip-MULTICAST payload-write channel: one payload write delivered to every chip on
 *        the armed multicast route (as opposed to MulticastIncChannel, which multicasts an
 *        atomic-inc). The NOC send type is still a unicast write — it is the CHIP-level routing
 *        that is multicast, which is why this takes its own multicast route rather than the
 *        stream's unicast one (same rule as arm_multicast_inc).
 */
template <typename ConnT>
class MulticastWriteChannel {
public:
    /// Issue one armed multicast payload write of the armed size to @c dst_noc_addr on every chip
    /// along the armed route (with_state — varies only the destination address).
    FORCE_INLINE void write(uint64_t dst_noc_addr, uint32_t src_l1_addr);

private:
    friend class FabricStream<ConnT>;
    FORCE_INLINE MulticastWriteChannel(ConnT* conn, volatile PACKET_HEADER_TYPE* hdr) : conn_(conn), hdr_(hdr) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_;
};

/// Armed chip-MULTICAST fused write + atomic-inc channel: the MulticastWriteChannel payload fan-out
/// with the receiver's semaphore bump folded into the same packet (see FusedWriteIncChannel).
template <typename ConnT>
class MulticastFusedWriteIncChannel {
public:
    /// Issue one armed multicast fused write+inc to every chip on the armed route.
    FORCE_INLINE void write_fused(uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr);

private:
    friend class FabricStream<ConnT>;
    FORCE_INLINE MulticastFusedWriteIncChannel(ConnT* conn, volatile PACKET_HEADER_TYPE* hdr) :
        conn_(conn), hdr_(hdr) {}
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
        fused_hdr_(o.fused_hdr_),
        mcast_write_hdr_(o.mcast_write_hdr_),
        mcast_fused_hdr_(o.mcast_fused_hdr_),
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

    // --- Armed fused write + atomic-inc channel --------------------------------------
    /// Arm the fused write+inc channel: program the stream's route + the invariant inc value, flush
    /// flag and on-wire payload size onto a pooled header once (set_state, Val|Flush|PayloadSize).
    /// Each issue varies the payload destination and the semaphore address.
    /// @param page_size_bytes  Payload size per issue (aligned up using the stream's alignment).
    /// @param val              Semaphore increment carried by every packet.
    /// @param flush            Whether the receiving fabric endpoint flushes the write before the inc.
    FORCE_INLINE FusedWriteIncChannel<ConnT> arm_fused_write_inc(
        uint32_t page_size_bytes, uint32_t val = 1, bool flush = false);

    // --- Armed multicast payload-write channel ---------------------------------------
    /// Arm a chip-MULTICAST payload write: program a MULTICAST route (its own, like
    /// arm_multicast_inc) + the invariant on-wire payload size onto a dedicated pooled header once
    /// (set_state, PayloadSize). Each issue varies only the destination address.
    FORCE_INLINE MulticastWriteChannel<ConnT> arm_multicast_write(
        const ccl_routing_utils::line_multicast_route_info_t& route, uint32_t page_size_bytes);

    // --- Armed multicast fused write + atomic-inc channel ----------------------------
    /// Arm a chip-MULTICAST fused write+inc: arm_multicast_write plus the receiver semaphore bump
    /// folded into the same packet (set_state, Val|Flush|PayloadSize).
    FORCE_INLINE MulticastFusedWriteIncChannel<ConnT> arm_multicast_fused_write_inc(
        const ccl_routing_utils::line_multicast_route_info_t& route,
        uint32_t page_size_bytes,
        uint32_t val = 1,
        bool flush = false);

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

    ConnT* conn_;                                             // borrowed from the FabricStreamSender
    uint32_t alignment_;                                      // L1 alignment for on-wire payload sizing
    ccl_routing_utils::line_unicast_route_info_t route_;      // bound at open(); reused by every unicast arm_*
    volatile PACKET_HEADER_TYPE* payload_hdr_ = nullptr;      // lazily allocated by arm_unicast_write
    volatile PACKET_HEADER_TYPE* scatter_hdr_ = nullptr;      // lazily allocated by arm_scatter_write
    volatile PACKET_HEADER_TYPE* sem_hdr_ = nullptr;          // lazily allocated by arm_inc
    volatile PACKET_HEADER_TYPE* mcast_hdr_ = nullptr;        // lazily allocated by arm_multicast_inc (independent)
    volatile PACKET_HEADER_TYPE* fused_hdr_ = nullptr;        // lazily allocated by arm_fused_write_inc
    volatile PACKET_HEADER_TYPE* mcast_write_hdr_ = nullptr;  // lazily allocated by arm_multicast_write
    volatile PACKET_HEADER_TYPE* mcast_fused_hdr_ = nullptr;  // lazily allocated by arm_multicast_fused_write_inc
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

    /**
     * @brief Split open: begin the connection handshake, do unrelated setup, then finish.
     *
     * @c open_start() starts the handshake and returns immediately; @c open_finish(route) completes it
     * and yields the stream. Between them, do work that does not touch the fabric — build address
     * generators, reset a local semaphore, read remaining runtime args — so the handshake latency
     * overlaps it. This exists because @c line_reduce_scatter_minimal_async_writer hand-rolls exactly
     * that overlap with @c fabric_client_connect_start / @c _finish and could not otherwise adopt the
     * helper without regressing.
     *
     * @warning Unlike the sender->stream->channel progression, this pairing is NOT enforced by the
     *   type system: @c open_start() returns void, so nothing stops a caller from forgetting
     *   @c open_finish() and issuing on a half-open connection. Use plain @c open() unless the overlap
     *   is worth that. Never call both @c open() and this pair on one sender.
     */
    FORCE_INLINE void open_start() { conn_.open_start(); }
    /// Complete a handshake begun by @c open_start(), bind @c route, and yield the opened stream.
    FORCE_INLINE FabricStream<ConnT> open_finish(const ccl_routing_utils::line_unicast_route_info_t& route) {
        conn_.open_finish();
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

// ============================================================================================
// THE DUPLEX TIER — what the reduction / fused collectives need.
//
// point_to_point and all_gather bind ONE direction per worker: a worker is either a
// forward-sender or a backward-sender, and FabricStream models exactly that. The reduction
// collectives instead have a single worker drive BOTH directions of one FabricConnectionManager,
// sending the same payload each way along a different route. Before this tier that meant the op
// carried two raw header pointers and re-derived `has_forward_connection()` /
// `has_backward_connection()` at every send site (see the free functions in
// minimal_ccl_common.hpp). Here the direction set is resolved ONCE at open() and each issue fans
// out to the connected directions automatically, so a forgotten direction check cannot happen.
//
// The type progression is unchanged: FabricDuplexSender -> open() -> FabricDuplexStream ->
// arm_*() -> armed channel -> issue. The CHIP-LEVEL cast is a compile-time property (Cast),
// fixed by which open() overload the op calls, so per-issue dispatch stays branch-free.
// ============================================================================================

/// Chip-level cast mode of a duplex stream's payload route. Chosen by the open() overload: a
/// unicast route pair yields Cast::Unicast, a multicast route pair Cast::Multicast. The NOC send
/// type is a unicast write either way — this selects only how the FABRIC routes the packet between
/// chips (one destination chip vs every chip in a hop range).
enum class Cast : uint8_t { Unicast, Multicast };

// Forward declarations: FabricDuplexStream constructs the duplex channel handles (their ctors are
// private and it is their friend); FabricDuplexSender constructs FabricDuplexStream.
template <Cast C, typename ConnT>
class FabricDuplexStream;
template <typename ConnT>
class FabricDuplexSender;

/**
 * @brief Duplex fabric-connection policy: one FabricConnectionManager, BOTH directions exposed.
 *
 * The counterpart to DirectConn, which pre-binds a single direction. Instead of a bare @c sender(),
 * this exposes @c has(dir) / @c sender(dir) so a duplex channel can fan an issue out over every
 * connected direction. A worker at the end of a line has only one of the two connected; the
 * has(dir) gate is what makes that case correct without any op-side conditional.
 *
 * @note @c open_finish() and @c close() on FabricConnectionManager already gate per-direction
 *   internally, so both are called unconditionally here. @c get_*_connection() does NOT — it
 *   ASSERTs — so @c sender(dir) must only be reached through a @c has(dir) check, which is exactly
 *   what the channels do.
 */
class DuplexConn {
public:
    using SenderT = tt::tt_fabric::WorkerToFabricEdmSender;
    static constexpr uint32_t kForward = 0;
    static constexpr uint32_t kBackward = 1;
    static constexpr uint32_t kNumDirections = 2;

    /// Build the connection (deferred open) from the fabric runtime-arg block; advances conn_arg_idx.
    /// Unlike DirectConn there is no is_forward argument — a duplex sender uses whichever directions
    /// the host actually wired up.
    FORCE_INLINE explicit DuplexConn(size_t& conn_arg_idx) :
        conn_(FabricConnectionManager::build_from_args<
              FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx)) {}

    FORCE_INLINE void open() { conn_.open_finish(); }
    FORCE_INLINE void close() { conn_.close(); }
    FORCE_INLINE bool has(uint32_t dir) const {
        return dir == kForward ? conn_.has_forward_connection() : conn_.has_backward_connection();
    }
    /// Valid ONLY when has(dir) — get_*_connection() asserts otherwise.
    FORCE_INLINE SenderT* sender(uint32_t dir) {
        return dir == kForward ? &conn_.get_forward_connection() : &conn_.get_backward_connection();
    }

private:
    FabricConnectionManager conn_;
};

/**
 * @brief Armed duplex payload-write channel: one call, one packet per connected direction.
 *
 * Replaces @c write_and_advance_local_read_address_for_fabric_write. The armed invariant is the
 * on-wire payload size; each issue varies the destination address.
 *
 * @note VARIABLE payload size. Unlike the unidirectional UnicastWriteChannel, whose size is a pure
 *   arm-time invariant, the reduction collectives size each packet by how many tiles the current
 *   shard/core contributes, so every issue re-programs PayloadSize (the same reason
 *   ScatterWriteChannel re-programs ChunkSizes per issue). The no-size overloads use the armed size
 *   for the common fixed-size case; the explicit-size overloads carry a per-packet size.
 */
template <Cast C, typename ConnT>
class DuplexWriteChannel {
public:
    /// Issue one payload write of the ARMED size to @c dst_noc_addr on EVERY connected direction.
    FORCE_INLINE void write(uint64_t dst_noc_addr, uint32_t src_l1_addr);
    /// Issue one payload write of @c payload_size_bytes (this packet only) on every connected direction.
    FORCE_INLINE void write(uint64_t dst_noc_addr, uint32_t src_l1_addr, uint32_t payload_size_bytes);
    /// write(), plus a LOCAL NoC copy of the same payload to the same logical destination on this
    /// chip — the "mirror the slice locally as well as forwarding it" step every reduction worker
    /// performs. Issues the local write first (so it overlaps the fabric sends) and flushes local
    /// writes before returning, matching the semantics of the free function it replaces.
    /// @note The op still advances its own L1 read cursor; see the banner's ownership split.
    FORCE_INLINE void write_with_local_copy(uint64_t dst_noc_addr, uint32_t src_l1_addr);
    FORCE_INLINE void write_with_local_copy(uint64_t dst_noc_addr, uint32_t src_l1_addr, uint32_t payload_size_bytes);
    /// Convenience over write(): resolve page @c page_idx of a consumed TensorAccessor/ShardedAddrGen.
    template <class AddrGen>
    FORCE_INLINE void write_page(uint32_t src_l1_addr, uint32_t page_idx, const AddrGen& dst);

private:
    friend class FabricDuplexStream<C, ConnT>;
    FORCE_INLINE DuplexWriteChannel(
        ConnT* conn,
        volatile PACKET_HEADER_TYPE* fwd_hdr,
        volatile PACKET_HEADER_TYPE* bwd_hdr,
        uint32_t payload_size) :
        conn_(conn), hdr_{fwd_hdr, bwd_hdr}, payload_size_(payload_size) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_[DuplexConn::kNumDirections];
    uint32_t payload_size_;  // armed on-wire size; the default for the no-size overloads
};

/**
 * @brief Armed duplex FUSED write + atomic-inc channel — the all_reduce / reduce_scatter workhorse.
 *
 * Replaces @c fused_write_atomic_and_advance_local_read_address_for_fabric_write: payload plus the
 * receiver's semaphore bump in one packet, fanned out over every connected direction.
 */
template <Cast C, typename ConnT>
class DuplexFusedWriteIncChannel {
public:
    /// Issue one fused write+inc of the ARMED payload size on every connected direction.
    FORCE_INLINE void write_fused(uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr);
    /// Issue one fused write+inc of @c payload_size_bytes (this packet only) on every connected direction.
    FORCE_INLINE void write_fused(
        uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr, uint32_t payload_size_bytes);
    /// write_fused(), plus the local NoC copy (see DuplexWriteChannel::write_with_local_copy).
    /// @note Unlike the write-only mirror this does NOT flush local writes — the fused free function
    ///   it replaces deliberately leaves flushing to the caller, which pairs the flush with its own
    ///   semaphore protocol. Keep the op's existing flush/barrier placement when migrating.
    FORCE_INLINE void write_fused_with_local_copy(
        uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr);
    FORCE_INLINE void write_fused_with_local_copy(
        uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr, uint32_t payload_size_bytes);

private:
    friend class FabricDuplexStream<C, ConnT>;
    FORCE_INLINE DuplexFusedWriteIncChannel(
        ConnT* conn,
        volatile PACKET_HEADER_TYPE* fwd_hdr,
        volatile PACKET_HEADER_TYPE* bwd_hdr,
        uint32_t payload_size) :
        conn_(conn), hdr_{fwd_hdr, bwd_hdr}, payload_size_(payload_size) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_[DuplexConn::kNumDirections];
    uint32_t payload_size_;
};

/**
 * @brief Armed duplex atomic-inc channel: bump a remote semaphore on EVERY connected direction.
 *
 * The duplex counterpart of AtomicIncChannel. A duplex worker that must announce readiness or
 * completion to BOTH of its neighbours previously issued two hand-routed incs, or re-derived
 * has_forward/has_backward at the call site; here the direction set is the one resolved at open() and
 * a forgotten direction cannot happen.
 *
 * @note The WAITING half stays op-owned (a local @c noc_semaphore_wait_min), exactly as for the
 *   unidirectional channels — the split documented in the file banner is unchanged. Note also that a
 *   duplex inc lands TWO increments (one per direction) when both are connected and one at the end of
 *   a line, so the op's wait threshold must be derived from its own topology, not assumed.
 */
template <Cast C, typename ConnT>
class DuplexIncChannel {
public:
    /// Atomically increment @c remote_sem_noc_addr by the armed value on every connected direction.
    FORCE_INLINE void inc(uint64_t remote_sem_noc_addr);

private:
    friend class FabricDuplexStream<C, ConnT>;
    FORCE_INLINE DuplexIncChannel(
        ConnT* conn, volatile PACKET_HEADER_TYPE* fwd_hdr, volatile PACKET_HEADER_TYPE* bwd_hdr) :
        conn_(conn), hdr_{fwd_hdr, bwd_hdr} {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_[DuplexConn::kNumDirections];
};

/// Armed duplex scatter-write channel (<=4 destinations per packet), fanned out over every connected
/// direction. Replaces @c scatter_write_and_advance_local_read_address_for_fabric_write.
template <Cast C, typename ConnT>
class DuplexScatterWriteChannel {
public:
    /// Issue one armed scatter write on every connected direction. @c num_chunks must be <= the arm.
    FORCE_INLINE void write_scatter(const uint64_t* dst_noc_addrs, uint32_t num_chunks, uint32_t src_l1_addr);

private:
    friend class FabricDuplexStream<C, ConnT>;
    FORCE_INLINE DuplexScatterWriteChannel(
        ConnT* conn,
        volatile PACKET_HEADER_TYPE* fwd_hdr,
        volatile PACKET_HEADER_TYPE* bwd_hdr,
        uint32_t chunk_size_bytes) :
        conn_(conn), hdr_{fwd_hdr, bwd_hdr}, chunk_size_bytes_(chunk_size_bytes) {}
    ConnT* conn_;
    volatile PACKET_HEADER_TYPE* hdr_[DuplexConn::kNumDirections];
    uint32_t chunk_size_bytes_;
};

/**
 * @brief FabricDuplexStream — the OPENED duplex egress. Owns the per-direction lazily-pooled
 *        headers and the per-direction routes; hands out armed duplex channels. Borrows the
 *        connection from the FabricDuplexSender that produced it, so the sender must outlive it
 *        (declare the sender first). RAII-closes on destruction.
 * @tparam C      Chip-level cast mode of the payload route (set by the open() overload used).
 * @tparam ConnT  Connection policy (DuplexConn).
 */
template <Cast C, typename ConnT = DuplexConn>
class FabricDuplexStream {
public:
    FabricDuplexStream(const FabricDuplexStream&) = delete;
    FabricDuplexStream& operator=(const FabricDuplexStream&) = delete;
    /// Move ctor (open() returns by value); transfers `closed_` so the moved-from stream never
    /// double-closes the transferred connection.
    FORCE_INLINE FabricDuplexStream(FabricDuplexStream&& o) :
        conn_(o.conn_), alignment_(o.alignment_), closed_(o.closed_) {
        for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
            uni_route_[d] = o.uni_route_[d];
            mcast_route_[d] = o.mcast_route_[d];
            write_hdr_[d] = o.write_hdr_[d];
            fused_hdr_[d] = o.fused_hdr_[d];
            scatter_hdr_[d] = o.scatter_hdr_[d];
            inc_hdr_[d] = o.inc_hdr_[d];
        }
        o.closed_ = true;
    }
    FabricDuplexStream& operator=(FabricDuplexStream&&) = delete;
    FORCE_INLINE ~FabricDuplexStream() { close(); }  // RAII backstop; idempotent with close()

    /// Arm the duplex payload-write channel: program each connected direction's route + the
    /// invariant on-wire payload size onto that direction's own pooled header (set_state).
    FORCE_INLINE DuplexWriteChannel<C, ConnT> arm_write(uint32_t page_size_bytes);
    /// Arm the duplex fused write+inc channel (set_state, Val|Flush|PayloadSize per direction).
    FORCE_INLINE DuplexFusedWriteIncChannel<C, ConnT> arm_fused_write_inc(
        uint32_t page_size_bytes, uint32_t val = 1, bool flush = false);
    /// Arm the duplex scatter-write channel (2..4 chunks per packet, per direction).
    FORCE_INLINE DuplexScatterWriteChannel<C, ConnT> arm_scatter_write(uint32_t chunk_size_bytes, uint32_t num_chunks);
    /// Arm the duplex atomic-inc channel: program each connected direction's route + increment value
    /// (+ flush) onto that direction's own pooled header (set_state, Val|Flush).
    FORCE_INLINE DuplexIncChannel<C, ConnT> arm_inc(uint32_t val = 1, bool flush = false);

    /// Drain outstanding local NoC writes + fabric atomic-incs. Optional — close() drains.
    FORCE_INLINE void drain();
    /// Drain, then close both directions. Idempotent.
    FORCE_INLINE void close();

private:
    friend class FabricDuplexSender<ConnT>;
    FORCE_INLINE FabricDuplexStream(ConnT* conn, uint32_t alignment) : conn_(conn), alignment_(alignment) {}

    ConnT* conn_;         // borrowed from the FabricDuplexSender
    uint32_t alignment_;  // L1 alignment for on-wire payload sizing
    // Per-direction routes. Only the member matching C is populated; the other stays default. Both
    // are held (rather than a union) so the arm_* bodies can `if constexpr` on C without casting.
    ccl_routing_utils::line_unicast_route_info_t uni_route_[DuplexConn::kNumDirections] = {};
    ccl_routing_utils::line_multicast_route_info_t mcast_route_[DuplexConn::kNumDirections] = {};
    // Per-direction, per-channel pooled headers; allocated lazily and ONLY for connected directions.
    volatile PACKET_HEADER_TYPE* write_hdr_[DuplexConn::kNumDirections] = {nullptr, nullptr};
    volatile PACKET_HEADER_TYPE* fused_hdr_[DuplexConn::kNumDirections] = {nullptr, nullptr};
    volatile PACKET_HEADER_TYPE* scatter_hdr_[DuplexConn::kNumDirections] = {nullptr, nullptr};
    volatile PACKET_HEADER_TYPE* inc_hdr_[DuplexConn::kNumDirections] = {nullptr, nullptr};
    bool closed_ = false;
};

/**
 * @brief FabricDuplexSender — the UNOPENED duplex egress. Owns the connection policy; open()
 *        finishes the connection, binds BOTH directions' routes, and yields the stream.
 *
 * The route-pair type picks the stream's cast mode, so an op cannot accidentally mix a unicast
 * route on one direction with a multicast route on the other.
 */
template <typename ConnT = DuplexConn>
class FabricDuplexSender {
public:
    /**
     * @brief Convenience ctor for the default DuplexConn policy: build the connection (deferred
     *        open) from runtime args. Advances conn_arg_idx past the fabric block.
     * @param conn_arg_idx  Index of the fabric arg block; ADVANCED past the block.
     * @param alignment     L1 alignment used to size the on-wire payload (bytes).
     */
    FORCE_INLINE FabricDuplexSender(size_t& conn_arg_idx, uint32_t alignment) :
        conn_(conn_arg_idx), alignment_(alignment) {}
    /// Construct from a pre-built connection policy.
    FORCE_INLINE FabricDuplexSender(ConnT conn, uint32_t alignment) : conn_(conn), alignment_(alignment) {}

    FabricDuplexSender(const FabricDuplexSender&) = delete;
    FabricDuplexSender& operator=(const FabricDuplexSender&) = delete;

    /// Open with a UNICAST route per direction -> Cast::Unicast stream.
    FORCE_INLINE FabricDuplexStream<Cast::Unicast, ConnT> open(
        const ccl_routing_utils::line_unicast_route_info_t& forward_route,
        const ccl_routing_utils::line_unicast_route_info_t& backward_route) {
        conn_.open();
        FabricDuplexStream<Cast::Unicast, ConnT> s(&conn_, alignment_);
        s.uni_route_[DuplexConn::kForward] = forward_route;
        s.uni_route_[DuplexConn::kBackward] = backward_route;
        return s;
    }

    /// Open with a MULTICAST route per direction -> Cast::Multicast stream (all_reduce's shape).
    FORCE_INLINE FabricDuplexStream<Cast::Multicast, ConnT> open(
        const ccl_routing_utils::line_multicast_route_info_t& forward_route,
        const ccl_routing_utils::line_multicast_route_info_t& backward_route) {
        conn_.open();
        FabricDuplexStream<Cast::Multicast, ConnT> s(&conn_, alignment_);
        s.mcast_route_[DuplexConn::kForward] = forward_route;
        s.mcast_route_[DuplexConn::kBackward] = backward_route;
        return s;
    }

private:
    ConnT conn_;
    uint32_t alignment_;
};

}  // namespace dataflow_kernel_lib::ccl

#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.inl"
