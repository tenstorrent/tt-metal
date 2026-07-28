// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Producer kernel (writer RISC). Shares its core with the receiver kernel and owns the ONE fabric
// sender connection that its eth channel allows (the L1 conn table is indexed by eth channel and the
// EDM stores a single worker_xy per channel, so a second core on the same channel would just hang).
// It therefore does two jobs:
//   1. send payload tokens to the peer worker on the neighbor chip, gated by `write_up_to`;
//   2. forward the co-located receiver's credit returns, since the receiver has no connection of its
//      own (being a fabric DESTINATION needs none — the peer's eth RISC writes into our L1 unasked).
//
// Credits are considered first every iteration. That is what keeps the ring deadlock-free: a producer
// blocked on its own `write_up_to` still forwards the credits the peer producer is waiting for. The
// loop also cannot exit at sent == num_tokens — it must have forwarded all num_tokens credits too, or
// the peer hangs.
//
// Credits do NOT get a packet each. They accumulate into a quarter-ring batch carried by a single
// atomic-inc packet, and are flushed early only when this producer has no payload to send anyway
// (done, or blocked on its own credits) — which is exactly when the peer is the one waiting. Liveness
// therefore does not depend on the batch size, while the link stops carrying one header-only packet
// per token: that alone was 36% of the producer's send window.
//
// All three counters are single-writer monotonic, so no atomics are needed on the read side: each
// reader keeps its own local count and works on the difference.

// Telemetry: the producer stamps tokens sent, credits forwarded, and three wall-clock timestamps into
// a fixed 1 kB L1 region so bandwidth can be recovered after the run without re-profiling. The window
// of interest is first-token-send -> credit-for-last-token-returned, which is why the loop now also
// waits for `write_up_to` to reach num_slots + num_tokens: that is the point at which the peer has
// provably consumed every token we sent (an upper bound on the last token's transfer completing).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "fabric/fabric_edm_packet_header.hpp"

// Wall clock, read the way the hardware requires: reading the LOW register latches HIGH, so the pair
// cannot tear.
inline uint64_t wall_clock() {
#if defined(RISCV_DEBUG_REG_WALL_CLOCK_L) && defined(RISCV_DEBUG_REG_WALL_CLOCK_H)
    volatile uint32_t tt_reg_ptr* lo = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint32_t tt_reg_ptr* hi = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint32_t low = lo[0];  // latches high
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(hi[0]) << 32);
#else
    return 0;  // no wall-clock debug register on this arch; telemetry timestamps read back as 0
#endif
}

// Telemetry word indices — must match TelemetryWord in the program factory.
constexpr uint32_t TELEM_MAGIC = 0;
constexpr uint32_t TELEM_TOKENS_SENT = 1;
constexpr uint32_t TELEM_CREDITS_FORWARDED = 2;
constexpr uint32_t TELEM_CHUNK_SIZE = 3;
constexpr uint32_t TELEM_NUM_SLOTS = 4;
constexpr uint32_t TELEM_T_FIRST_SEND_LO = 5;
constexpr uint32_t TELEM_T_FIRST_SEND_HI = 6;
constexpr uint32_t TELEM_T_LAST_SEND_LO = 7;
constexpr uint32_t TELEM_T_LAST_SEND_HI = 8;
constexpr uint32_t TELEM_T_LAST_CREDIT_LO = 9;
constexpr uint32_t TELEM_T_LAST_CREDIT_HI = 10;
constexpr uint32_t TELEM_WRITE_UP_TO_FINAL = 11;
constexpr uint32_t TELEM_EDM_SLOTS = 12;
constexpr uint32_t TELEM_CREDIT_PACKETS = 13;
constexpr uint32_t TELEM_WAIT_SLOT_CY_LO = 14;
constexpr uint32_t TELEM_WAIT_SLOT_CY_HI = 15;
constexpr uint32_t TELEM_ISSUE_CY_LO = 16;
constexpr uint32_t TELEM_ISSUE_CY_HI = 17;
constexpr uint32_t TELEM_STARVE_CY_LO = 18;
constexpr uint32_t TELEM_STARVE_CY_HI = 19;
constexpr uint32_t TELEM_CREDIT_CY_LO = 20;
constexpr uint32_t TELEM_CREDIT_CY_HI = 21;
constexpr uint32_t TELEM_LOOP_ITERS = 22;
constexpr uint32_t TELEMETRY_MAGIC = 0xCF2D0002u;

void kernel_main() {
    constexpr uint32_t num_tokens = get_compile_time_arg_val(0);
    constexpr uint32_t num_slots = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t peer_chip_id = get_compile_time_arg_val(3);
    constexpr uint32_t peer_mesh_id = get_compile_time_arg_val(4);
    constexpr uint32_t peer_noc_x = get_compile_time_arg_val(5);
    constexpr uint32_t peer_noc_y = get_compile_time_arg_val(6);
    constexpr uint32_t prod_buf_addr = get_compile_time_arg_val(7);
    constexpr uint32_t recv_buf_addr = get_compile_time_arg_val(8);  // same address on every chip
    // Unused since the payload headers moved into the per-slot ring below; the L1 slot stays reserved
    // so the layout (and every other compile-time arg index) is untouched.
    [[maybe_unused]] constexpr uint32_t pkt_hdr_payload_addr = get_compile_time_arg_val(9);
    constexpr uint32_t pkt_hdr_credit_addr = get_compile_time_arg_val(10);
    constexpr uint32_t write_up_to_addr = get_compile_time_arg_val(11);
    constexpr uint32_t data_ready_addr = get_compile_time_arg_val(12);
    constexpr uint32_t credits_to_return_addr = get_compile_time_arg_val(13);
    constexpr uint32_t telemetry_addr = get_compile_time_arg_val(14);
    // Fine-grained stall buckets cost ~3 wall-clock register reads per token (a few percent of the
    // token's own cycles), so they are compile-time optional: on to explain a number, off to quote one.
    constexpr bool stall_telemetry = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t variant = get_compile_time_arg_val(16);
    constexpr uint32_t pkt_hdr_ring_addr = get_compile_time_arg_val(17);
    // Phase 3 — direct-to-DRAM (Approach #1). When set, the producer ignores all credit/semaphore
    // machinery and simply fires every token to an incrementally larger page in the peer chip's DRAM
    // output buffer, starting at `dram_base_page`. Plain unicast write (NOT the fused write+atomic-inc,
    // which is documented to hang Blackhole on a DRAM destination — moe_utils.hpp:486). EDM
    // sender-slot backpressure is the only throttle, so the send window IS the DRAM-bounded rate.
    constexpr uint32_t dram_base_page = get_compile_time_arg_val(18);
    constexpr uint32_t dram_bank_base_addr = get_compile_time_arg_val(19);
    constexpr auto dram_out_args = TensorAccessorArgs<20>();

    // Diagnostics only (see CombineFabric2dParams::variant); both break a guarantee to price it.
    constexpr bool RELAXED_READY = (variant & 4u) != 0;
    constexpr bool NO_FLOW_CONTROL = (variant & 8u) != 0;
    constexpr bool DRAM_DIRECT = (variant & 32u) != 0;  // Approach #1
    // Credit batch threshold: a quarter ring. Small enough that the peer never runs dry (it can be up to
    // a quarter ring ahead of the credits it has been told about, and the ring is num_slots deep), large
    // enough to cut the credit packet count ~4x. Liveness does not depend on it: credits are flushed
    // unconditionally whenever we have nothing else to do (see the loop).
    constexpr uint32_t credit_batch = num_slots >= 4 ? num_slots / 4 : 1;

    // All tokens consumed AND credited by the peer. Reaching this is what closes the timing window.
    constexpr uint32_t write_up_to_target = num_slots + num_tokens;

    volatile tt_l1_ptr uint32_t* telem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(telemetry_addr);
    // Invalidate first so a stale record from an earlier run can never be mistaken for this one's.
    telem[TELEM_MAGIC] = 0;

    std::size_t rt_args_idx = 0;
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
        rt_args_idx, num_connections);
    auto& sender = fabric_connections.get(0).sender;

    volatile PACKET_HEADER_TYPE* pkt_hdr_credit = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_credit_addr);
    // Written only by the remote eth RISC (credit packets); we just read it.
    volatile tt_l1_ptr uint32_t* write_up_to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_up_to_addr);
    // Written only by the receiver kernel on this core; we just read it.
    volatile tt_l1_ptr uint32_t* credits_to_return =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credits_to_return_addr);

    // Remote targets on the peer worker core: its data_ready (bumped by the fused payload write) and
    // its write_up_to (bumped by our credit returns).
    const uint64_t peer_data_ready_noc = get_noc_addr(peer_noc_x, peer_noc_y, data_ready_addr);
    const uint64_t peer_write_up_to_noc = get_noc_addr(peer_noc_x, peer_noc_y, write_up_to_addr);

    // Build every ring slot's header once, up front. The only per-token variation is the destination
    // slot address, so nothing in the loop has to touch a header again — and because a slot's header is
    // not reused until the ring wraps, the send does not have to flush-block on it either.
    auto slot_hdr = [](uint32_t slot) -> volatile PACKET_HEADER_TYPE* {
        return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_ring_addr + slot * sizeof(PACKET_HEADER_TYPE));
    };
    // Approach #1 addresses the peer chip's DRAM output buffer by page. Interleaved layout is identical
    // across chips, so an accessor built from THIS chip's buffer base produces addresses valid on the
    // peer chip (same trick the production combine op relies on).
    const auto dram_out = TensorAccessor(dram_out_args, dram_bank_base_addr);
    if constexpr (DRAM_DIRECT) {
        // Only the route is fixed per slot header (same peer chip every token); the destination page
        // changes per token and is stamped in the loop by to_noc_unicast_write.
        for (uint32_t slot = 0; slot < num_slots; slot++) {
            fabric_set_unicast_route(
                (volatile tt::tt_fabric::HybridMeshPacketHeader*)slot_hdr(slot), peer_chip_id, peer_mesh_id);
        }
    } else {
        for (uint32_t slot = 0; slot < num_slots; slot++) {
            volatile PACKET_HEADER_TYPE* h = slot_hdr(slot);
            const uint64_t dst_noc = get_noc_addr(peer_noc_x, peer_noc_y, recv_buf_addr + slot * chunk_size_bytes);
            fabric_set_unicast_route((volatile tt::tt_fabric::HybridMeshPacketHeader*)h, peer_chip_id, peer_mesh_id);
            h->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    dst_noc, peer_data_ready_noc, /*val=*/1, /*flush=*/!RELAXED_READY},
                chunk_size_bytes);
        }
    }

    uint32_t sent = 0;          // tokens handed to the fabric
    uint32_t credits_sent = 0;  // credits forwarded on the receiver's behalf
    uint64_t t_first_send = 0;
    uint64_t t_last_send = 0;
    uint64_t t_last_credit = 0;
    uint32_t observed_write_up_to = num_slots;

    // Stall attribution. Four disjoint buckets covering the send window, so a run answers "why are we
    // below line rate" directly: wait_slot => the eth side cannot drain us, starve => the credit
    // round-trip gates us, issue/credit => our own per-packet cost. Each bucket costs two wall-clock
    // register reads per occurrence (~1% of a token's cycles at 14 kB).
    uint64_t wait_slot_cy = 0;
    uint64_t issue_cy = 0;
    uint64_t starve_cy = 0;
    uint64_t credit_cy = 0;
    uint64_t starve_start = 0;  // 0 => not currently credit-starved
    uint32_t credit_packets = 0;
    uint32_t loop_iters = 0;

    {
        // One zone per token, plus this outer one. The profiler L1 buffer holds 250 optional markers
        // per RISC, so at num_tokens=100 a second per-token zone would overflow and silently drop.
        DeviceZoneScopedN("PRODUCER_LOOP");
        if constexpr (DRAM_DIRECT) {
            // Approach #1: no credits, no receiver, no ring reuse to protect. Each token lands on its own
            // DRAM page (base_page + sent), so the only backpressure is the EDM refusing an empty write
            // slot when the eth side cannot drain to DRAM fast enough. t_last_credit stays 0: there is no
            // end-to-end signal, so the send window (first send -> last send) IS the measured rate.
            while (sent < num_tokens) {
                const uint32_t slot = sent % num_slots;
                volatile PACKET_HEADER_TYPE* hdr = slot_hdr(slot);
                // Stamp this token's DRAM page into the (route-preset) header. Plain write, no atomic.
                tt::tt_fabric::linear::to_noc_unicast_write(chunk_size_bytes, hdr, dram_base_page + sent, dram_out);
                const uint64_t w0 = wall_clock();
                if (sent == 0) {
                    t_first_send = w0;
                }
                sender.wait_for_empty_write_slot();
                // Single fixed source chunk (payload content is garbage in this harness); a slot's header is
                // not reused until the ring wraps, so the header send need not flush-block.
                sender.send_payload_without_header_non_blocking_from_address(prod_buf_addr, chunk_size_bytes);
                sender.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
                sent++;
                t_last_send = wall_clock();
            }
        } else {
            // In the NO_FLOW_CONTROL diagnostic there are no credits to wait on in either direction, so the
            // window closes at the last send and t_last_credit stays 0 (report the send-window number).
            while (NO_FLOW_CONTROL ? sent < num_tokens
                                   : (sent < num_tokens || credits_sent < num_tokens ||
                                      observed_write_up_to < write_up_to_target)) {
                loop_iters++;
                // ---- 1. Credits first, always. Never gated on anything, so a stalled producer cannot
                // ---- stall its peer. One packet carries the whole pending batch as its inc value.
                invalidate_l1_cache();
                const uint32_t returnable = *credits_to_return;
                // The packet is held back until a batch has accumulated — unless we have no payload to send
                // anyway (either done sending, or blocked on our own credits), in which case sending it now
                // is free and is what keeps the ring live. One packet per token cost 36% of the send window.
                // NO_FLOW_CONTROL also suppresses the credit packets themselves: with nobody gated on
                // credits, forwarding them would only steal link bandwidth from the payload, and the point
                // of that diagnostic is the payload-only ceiling.
                const bool have_credits = !NO_FLOW_CONTROL && returnable > credits_sent;
                const bool send_credits_now = have_credits && ((returnable - credits_sent) >= credit_batch ||
                                                               sent >= num_tokens || sent >= observed_write_up_to);
                if (send_credits_now) {
                    const uint64_t c0 = stall_telemetry ? wall_clock() : 0;
                    const uint32_t batch = returnable - credits_sent;
                    pkt_hdr_credit->to_noc_unicast_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_write_up_to_noc, batch, /*flush=*/false});
                    fabric_set_unicast_route(
                        (volatile tt::tt_fabric::HybridMeshPacketHeader*)pkt_hdr_credit, peer_chip_id, peer_mesh_id);
                    sender.wait_for_empty_write_slot();
                    sender.send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_hdr_credit, sizeof(PACKET_HEADER_TYPE));
                    credits_sent += batch;
                    credit_packets++;
                    if constexpr (stall_telemetry) {
                        credit_cy += wall_clock() - c0;
                    }
                }

                // ---- 2. Track the peer's progress. Once it has credited every token we sent, the
                // ---- transfer is provably complete and the timing window closes.
                invalidate_l1_cache();
                const uint32_t cur_write_up_to = *write_up_to;
                if (cur_write_up_to > observed_write_up_to) {
                    observed_write_up_to = cur_write_up_to;
                    if (observed_write_up_to >= write_up_to_target) {
                        t_last_credit = wall_clock();
                    }
                }

                // ---- 3. Then a token, if the peer's ring has room for it.
                if (sent < num_tokens) {
                    if (NO_FLOW_CONTROL || sent < cur_write_up_to) {
                        DeviceZoneScopedN("PRODUCER_SEND");
                        // Credit starvation ended here, if we were in it.
                        const uint64_t w0 = (stall_telemetry || sent == 0) ? wall_clock() : 0;
                        if constexpr (stall_telemetry) {
                            if (starve_start != 0) {
                                starve_cy += w0 - starve_start;
                                starve_start = 0;
                            }
                        }
                        if (sent == 0) {
                            t_first_send = w0;
                        }
                        // Header first, THEN wait for the slot — building it while the EDM may still be
                        // busy is free overlap, and reversing the two costs ~8% of the bandwidth.
                        const uint32_t slot = sent % num_slots;
                        volatile PACKET_HEADER_TYPE* hdr = slot_hdr(slot);
                        // Waiting for an EDM slot is the one stall that means "the eth side cannot drain
                        // us", so it gets its own bucket; the header build above counts as issue cost.
                        const uint64_t wh = stall_telemetry ? wall_clock() : 0;
                        sender.wait_for_empty_write_slot();
                        uint64_t w1 = 0;
                        if constexpr (stall_telemetry) {
                            w1 = wall_clock();
                            wait_slot_cy += w1 - wh;
                            issue_cy += wh - w0;
                        }
                        sender.send_payload_without_header_non_blocking_from_address(
                            prod_buf_addr + slot * chunk_size_bytes, chunk_size_bytes);
                        // No flush: this slot's header is not touched again until the ring wraps, and the
                        // source payload is never written by us at all. Letting the NoC queue hold the
                        // writes is what allows token N+1 to be issued while N is still draining.
                        //
                        // CAVEAT: the production idiom (moe_utils.hpp) flush-blocks here, which also orders
                        // our payload write ahead of the EDM slot-credit write that
                        // post_send_payload_increment_pointers issues on the sync cmd buf. Dropping the
                        // flush leans on the NoC keeping those in order to the same destination. It has not
                        // misbehaved over 128 producers x 10k tokens (every token acked, no hang), but this
                        // harness never checks payload CONTENT, so a torn packet would be invisible here.
                        // Validate content (Phase 3's DRAM drain) before relying on it in a real op.
                        sender.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
                        sent++;
                        t_last_send = wall_clock();
                        if constexpr (stall_telemetry) {
                            issue_cy += t_last_send - w1;
                        }
                    } else if (stall_telemetry && starve_start == 0) {
                        // Tokens left to send but no credit for them: the credit round-trip is what we are
                        // waiting on. Timed from the first iteration that observes it.
                        starve_start = wall_clock();
                    }
                }
            }
        }  // end !DRAM_DIRECT
    }

    telem[TELEM_TOKENS_SENT] = sent;
    telem[TELEM_CREDITS_FORWARDED] = credits_sent;
    telem[TELEM_CHUNK_SIZE] = chunk_size_bytes;
    telem[TELEM_NUM_SLOTS] = num_slots;
    telem[TELEM_T_FIRST_SEND_LO] = (uint32_t)(t_first_send & 0xFFFFFFFFu);
    telem[TELEM_T_FIRST_SEND_HI] = (uint32_t)(t_first_send >> 32);
    telem[TELEM_T_LAST_SEND_LO] = (uint32_t)(t_last_send & 0xFFFFFFFFu);
    telem[TELEM_T_LAST_SEND_HI] = (uint32_t)(t_last_send >> 32);
    telem[TELEM_T_LAST_CREDIT_LO] = (uint32_t)(t_last_credit & 0xFFFFFFFFu);
    telem[TELEM_T_LAST_CREDIT_HI] = (uint32_t)(t_last_credit >> 32);
    telem[TELEM_WRITE_UP_TO_FINAL] = observed_write_up_to;
    telem[TELEM_EDM_SLOTS] = sender.num_buffers_per_channel;
    telem[TELEM_CREDIT_PACKETS] = credit_packets;
    telem[TELEM_LOOP_ITERS] = loop_iters;
    telem[TELEM_WAIT_SLOT_CY_LO] = (uint32_t)(wait_slot_cy & 0xFFFFFFFFu);
    telem[TELEM_WAIT_SLOT_CY_HI] = (uint32_t)(wait_slot_cy >> 32);
    telem[TELEM_ISSUE_CY_LO] = (uint32_t)(issue_cy & 0xFFFFFFFFu);
    telem[TELEM_ISSUE_CY_HI] = (uint32_t)(issue_cy >> 32);
    telem[TELEM_STARVE_CY_LO] = (uint32_t)(starve_cy & 0xFFFFFFFFu);
    telem[TELEM_STARVE_CY_HI] = (uint32_t)(starve_cy >> 32);
    telem[TELEM_CREDIT_CY_LO] = (uint32_t)(credit_cy & 0xFFFFFFFFu);
    telem[TELEM_CREDIT_CY_HI] = (uint32_t)(credit_cy >> 32);
    // Magic last: a reader that sees it knows every field above is committed.
    telem[TELEM_MAGIC] = TELEMETRY_MAGIC;

    noc_async_writes_flushed();
    fabric_connections.close();
}
