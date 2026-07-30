// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Producer kernel (writer RISC, NOC_1). Owns the ONE fabric sender connection its eth channel allows (the
// L1 connection table is indexed by eth channel and the EDM stores a single worker_xy per channel, so a
// second core on the same channel would just hang) and executes a LIST of assignments: its routing
// plane's share of this chip's movements to every other chip on the axis, one fabric packet per token,
// 1:1 and in order within each assignment.
//
// Destinations are no longer restricted to the chip across this producer's own cable — an assignment may
// name a chip several hops away, and the fabric forwards it. Phase 9 takes that forwarding over.
//
// The tokens do NOT come from DRAM here: the reader kernel on this same core streams them into a
// num_l1_slots-deep L1 ring, and this loop drains it. Token counts are the caller's, unbounded by L1.
//
// Slots are claimed and released in batches of `batch`, which is what amortises the two counter bumps and
// the source flush over several packets. The flush matters now that the ring is REUSED: a payload send
// reads L1 asynchronously, so a slot cannot be handed back to the reader until that read has drained.
// noc_async_writes_flushed() is exactly that guarantee (source reusable), and is cheaper than a barrier.
//
// Which movement a producer gets is decided host-side (see the program factory). Nothing here knows how
// many other producers exist or how the caller's regions are carved up.
//
// Telemetry: counts and wall-clock timestamps into a fixed 1 kB L1 region, so bandwidth can be recovered
// after the run without re-profiling. The effective-bandwidth window opens in the READER (at its first
// DRAM read) and closes here at the drain.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
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
constexpr uint32_t TELEM_TOKEN_SIZE = 2;
constexpr uint32_t TELEM_NUM_L1_SLOTS = 3;
// 4,5 = T_START, written by the reader kernel (first DRAM read). This kernel must not touch them.
constexpr uint32_t TELEM_T_FIRST_SEND_LO = 6;
constexpr uint32_t TELEM_T_FIRST_SEND_HI = 7;
constexpr uint32_t TELEM_T_LAST_SEND_LO = 8;
constexpr uint32_t TELEM_T_LAST_SEND_HI = 9;
constexpr uint32_t TELEM_T_DRAINED_LO = 10;
constexpr uint32_t TELEM_T_DRAINED_HI = 11;
constexpr uint32_t TELEM_EDM_SLOTS = 12;
constexpr uint32_t TELEM_DRAIN_PACKETS = 13;
constexpr uint32_t TELEM_OUT_BASE_PAGE = 14;
constexpr uint32_t TELEM_BATCH = 15;
constexpr uint32_t TELEM_WAIT_SLOT_CY_LO = 16;
constexpr uint32_t TELEM_WAIT_SLOT_CY_HI = 17;
constexpr uint32_t TELEM_ISSUE_CY_LO = 18;
constexpr uint32_t TELEM_ISSUE_CY_HI = 19;
constexpr uint32_t TELEM_RING_WAIT_CY_LO = 20;
constexpr uint32_t TELEM_RING_WAIT_CY_HI = 21;
// Whole-kernel span: entry (before anything, including the fabric connection open) to exit (after the
// connection teardown). t_last_send - t_first_send is the send loop alone; this is what total-time
// optimisation is measured against.
constexpr uint32_t TELEM_T_KERNEL_START_LO = 22;
constexpr uint32_t TELEM_T_KERNEL_START_HI = 23;
constexpr uint32_t TELEM_T_KERNEL_END_LO = 24;
constexpr uint32_t TELEM_T_KERNEL_END_HI = 25;
constexpr uint32_t TELEMETRY_MAGIC = 0xCF2D0006u;

void kernel_main() {
    // First thing the kernel does, so the fabric connection open and the header prebuild below are both
    // inside the kernel-total window.
    const uint64_t t_kernel_start = wall_clock();

    // The ring geometry this producer shares with the reader, plus its assignment table.
    constexpr uint32_t num_l1_slots = get_compile_time_arg_val(0);
    constexpr uint32_t token_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slot_tail_bytes = get_compile_time_arg_val(2);
    // The IMMEDIATE ring neighbour across this producer's own cable. Only the end-of-run drain targets it;
    // payload destinations come from the assignment table and can be several hops away.
    constexpr uint32_t peer_chip_id = get_compile_time_arg_val(3);
    constexpr uint32_t peer_mesh_id = get_compile_time_arg_val(4);
    constexpr uint32_t peer_noc_x = get_compile_time_arg_val(5);
    constexpr uint32_t peer_noc_y = get_compile_time_arg_val(6);
    constexpr uint32_t ring_addr = get_compile_time_arg_val(7);
    constexpr uint32_t pkt_hdr_ring_addr = get_compile_time_arg_val(8);
    constexpr uint32_t pkt_hdr_drain_addr = get_compile_time_arg_val(9);
    // Uniform L1 address of the peer worker's 4-byte drain sink: the target of the drain's value-0
    // atomic increments. Nothing reads it, which is the point.
    constexpr uint32_t drain_sink_addr = get_compile_time_arg_val(10);
    constexpr uint32_t telemetry_addr = get_compile_time_arg_val(11);
    // Fine-grained stall buckets cost ~2 wall-clock register reads per token (a few percent of the
    // token's own cycles), so they are compile-time optional: on to explain a number, off to quote one.
    constexpr bool stall_telemetry = get_compile_time_arg_val(12) != 0;
    // The output buffer's base address, uniform across the mesh, so an accessor built from this chip's
    // base produces addresses valid on any destination chip.
    constexpr uint32_t dram_out_base_addr = get_compile_time_arg_val(13);
    // Ring handshake: the reader bumps `filled`, we bump `freed`. Both monotonic, single-writer.
    constexpr uint32_t batch = get_compile_time_arg_val(14);
    constexpr uint32_t filled_addr = get_compile_time_arg_val(15);
    constexpr uint32_t freed_addr = get_compile_time_arg_val(16);
    constexpr uint32_t my_noc_x = get_compile_time_arg_val(17);
    constexpr uint32_t my_noc_y = get_compile_time_arg_val(18);
    constexpr uint32_t num_assignments = get_compile_time_arg_val(19);
    // Per-assignment table: [out_base_token, num_tokens, dst_chip_id, dst_mesh_id]. Read through
    // kernel_compile_time_args (a constexpr std::array) because get_compile_time_arg_val needs a literal
    // index and this table is walked by a loop variable.
    constexpr uint32_t ASSIGN_BASE = 20;
    constexpr uint32_t ASSIGN_WORDS = 4;
    constexpr auto dram_out_args = TensorAccessorArgs<ASSIGN_BASE + ASSIGN_WORDS * num_assignments>();
    // A ring slot is the token plus a metadata tail; only the token part is sent over the fabric.
    constexpr uint32_t slot_stride = token_size_bytes + slot_tail_bytes;

    volatile tt_l1_ptr uint32_t* telem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(telemetry_addr);
    // Invalidate first so a stale record from an earlier run can never be mistaken for this one's.
    telem[TELEM_MAGIC] = 0;

    std::size_t rt_args_idx = 0;
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
        rt_args_idx, num_connections);
    auto& sender = fabric_connections.get(0).sender;

    // One header per ring slot. A slot's header is not touched again until the ring wraps, and the
    // per-batch noc_async_writes_flushed() below is what makes that wrap safe — so stamping the route here
    // is exactly as safe as stamping the destination page was.
    //
    // The route can no longer be prebuilt once for the whole run: destinations now change per assignment.
    // It is restamped per token rather than per assignment because slots from the previous assignment may
    // still be in flight when the next one starts (the ring is continuous across assignments), so
    // rewriting all slot headers at a boundary could clobber a header still being read.
    auto slot_hdr = [](uint32_t slot) -> volatile PACKET_HEADER_TYPE* {
        return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_ring_addr + slot * sizeof(PACKET_HEADER_TYPE));
    };

    // Written only by the reader on this core; we just read it.
    volatile tt_l1_ptr uint32_t* filled = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(filled_addr);
    const uint64_t my_freed_noc = get_noc_addr(my_noc_x, my_noc_y, freed_addr);

    // Addresses the peer chip's output region by page. The interleaved layout is identical across chips,
    // so an accessor built from THIS chip's buffer base produces addresses valid on the peer chip (the
    // same trick the production combine op relies on).
    const auto dram_out = TensorAccessor(dram_out_args, dram_out_base_addr);

    uint32_t sent = 0;
    uint64_t t_first_send = 0;
    uint64_t t_last_send = 0;
    // Stall attribution over the send window: wait_slot means the eth side cannot drain us (the link or
    // the far end's DRAM write is the limiter), issue is our own per-packet cost. Each costs two
    // wall-clock register reads per token (~1% of a token's cycles at 14 kB).
    uint64_t wait_slot_cy = 0;
    uint64_t issue_cy = 0;
    uint64_t ring_wait_cy = 0;  // blocked on the reader => the DRAM side is the limiter

    // ---- Send. Drain the ring in batches; the only backpressure is the EDM refusing an empty write slot
    // ---- (eth-limited) or the reader not having filled the ring yet (DRAM-limited).
    {
        // One zone per batch. The profiler L1 buffer holds 250 optional markers per RISC, so a per-token
        // zone would overflow and silently drop data at the token counts this op now runs.
        DeviceZoneScopedN("PRODUCER_LOOP");
        // Outer loop over this producer's assignments. `sent` never restarts: the ring stream is
        // continuous, so the assignment boundary only changes WHERE tokens go, not the flow control.
        for (uint32_t a = 0; a < num_assignments; a++) {
            const uint32_t out_base_page = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 0];
            const uint32_t assignment_tokens = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 1];
            const uint32_t dst_chip_id = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 2];
            const uint32_t dst_mesh_id = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 3];
            uint32_t done = 0;
            while (done < assignment_tokens) {
                const uint32_t n = (assignment_tokens - done) < batch ? (assignment_tokens - done) : batch;
                // Wait for n filled slots.
                const uint64_t r0 = stall_telemetry ? wall_clock() : 0;
                while (true) {
                    invalidate_l1_cache();
                    if (*filled - sent >= n) {
                        break;
                    }
                }
                if constexpr (stall_telemetry) {
                    ring_wait_cy += wall_clock() - r0;
                }
                for (uint32_t i = 0; i < n; i++) {
                    const uint32_t slot = (sent + i) % num_l1_slots;
                    volatile PACKET_HEADER_TYPE* hdr = slot_hdr(slot);
                    // Header first, THEN wait for the slot — building it while the EDM may still be busy is
                    // free overlap, and reversing the two costs ~8% of the bandwidth. Plain unicast write,
                    // NOT the fused write+atomic-inc, which is documented to hang Blackhole on a DRAM
                    // destination (moe_utils.hpp:486).
                    fabric_set_unicast_route(
                        (volatile tt::tt_fabric::HybridMeshPacketHeader*)hdr, dst_chip_id, dst_mesh_id);
                    tt::tt_fabric::linear::to_noc_unicast_write(
                        token_size_bytes, hdr, out_base_page + done + i, dram_out);
                    const uint64_t w0 = wall_clock();
                    if (sent + i == 0) {
                        t_first_send = w0;
                    }
                    sender.wait_for_empty_write_slot();
                    uint64_t w1 = 0;
                    if constexpr (stall_telemetry) {
                        w1 = wall_clock();
                        wait_slot_cy += w1 - w0;
                    }
                    sender.send_payload_without_header_non_blocking_from_address(
                        ring_addr + slot * slot_stride, token_size_bytes);
                    // No flush per token: a slot's header is not touched again until the ring wraps, and the
                    // payload is flushed once per batch below. Letting the NoC queue hold the writes is what
                    // allows token N+1 to be issued while N is still draining.
                    //
                    // CAVEAT: the production idiom (moe_utils.hpp) flush-blocks here, which also orders our
                    // payload write ahead of the EDM slot-credit write that post_send_payload_increment_pointers
                    // issues on the sync cmd buf. Dropping the flush leans on the NoC keeping those in order to
                    // the same destination. The op's content check is what makes that safe to rely on: a torn
                    // packet would show up as a wrong output token.
                    sender.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
                    t_last_send = wall_clock();
                    if constexpr (stall_telemetry) {
                        issue_cy += t_last_send - w1;
                    }
                }
                // The batch's payload reads have drained out of L1, so these slots are safe to refill.
                noc_async_writes_flushed();
                sent += n;
                done += n;
                noc_semaphore_inc(my_freed_noc, n);
            }
        }
    }

    // ---- 3. Upper bound on delivery. Nothing in this op tells us when the last token landed, so the
    // send window alone is a LOWER bound on the transfer. The EDM sender channel supplies the missing
    // signal for free.
    //
    // The worker's free-slot count is `D = num_buffers_per_channel` deep and satisfies
    // free = D - (packets_written - credits_returned) (edm_fabric_worker_adapters.hpp:283). A credit is
    // only produced by the FAR END (the router forwards to the worker what the remote receiver channel
    // acked — fabric_erisc_router.cpp:1690). So after `sent` payload packets, writing D-1 more and
    // then obtaining one further free slot forces credits_returned >= sent, i.e. every payload
    // packet has provably reached the destination chip. That moment is what turns the report's GB/s into
    // an upper-bound end-to-end rate while sGB/s stays the push rate. It does NOT prove the destination
    // DRAM write has retired — the far eRISC may ack on write issue.
    //
    // The D-1 filler packets are header-only atomic-incs of value ZERO aimed at the peer worker's drain
    // sink word: a real, valid fabric packet (there is no NOP send type) that changes nothing anywhere.
    // Their own completion is never awaited, so the argument above does not depend on them landing.
    // Reaching a free slot cannot deadlock: the reverse direction is a different eth channel, so no
    // producer waits on another producer here.
    uint32_t drain_packets = 0;
    if (sent > 0) {
        volatile PACKET_HEADER_TYPE* hdr_drain = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_drain_addr);
        const uint64_t peer_drain_sink_noc = get_noc_addr(peer_noc_x, peer_noc_y, drain_sink_addr);
        const uint32_t depth = sender.num_buffers_per_channel;
        hdr_drain->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_drain_sink_noc, /*val=*/0, /*flush=*/false});
        fabric_set_unicast_route(
            (volatile tt::tt_fabric::HybridMeshPacketHeader*)hdr_drain, peer_chip_id, peer_mesh_id);
        for (uint32_t d = 0; d + 1 < depth; d++) {
            sender.wait_for_empty_write_slot();
            sender.send_payload_flush_blocking_from_address((uint32_t)hdr_drain, sizeof(PACKET_HEADER_TYPE));
            drain_packets++;
        }
        sender.wait_for_empty_write_slot();
    }
    const uint64_t t_drained = wall_clock();

    telem[TELEM_TOKENS_SENT] = sent;
    telem[TELEM_TOKEN_SIZE] = token_size_bytes;
    telem[TELEM_NUM_L1_SLOTS] = num_l1_slots;
    telem[TELEM_BATCH] = batch;
    telem[TELEM_T_FIRST_SEND_LO] = (uint32_t)(t_first_send & 0xFFFFFFFFu);
    telem[TELEM_T_FIRST_SEND_HI] = (uint32_t)(t_first_send >> 32);
    telem[TELEM_T_LAST_SEND_LO] = (uint32_t)(t_last_send & 0xFFFFFFFFu);
    telem[TELEM_T_LAST_SEND_HI] = (uint32_t)(t_last_send >> 32);
    telem[TELEM_T_DRAINED_LO] = (uint32_t)(t_drained & 0xFFFFFFFFu);
    telem[TELEM_T_DRAINED_HI] = (uint32_t)(t_drained >> 32);
    telem[TELEM_EDM_SLOTS] = sender.num_buffers_per_channel;
    telem[TELEM_DRAIN_PACKETS] = drain_packets;
    // First assignment's destination page. Only a breadcrumb now that a producer serves several
    // assignments; the full table lives in the factory's log line.
    telem[TELEM_OUT_BASE_PAGE] = kernel_compile_time_args[ASSIGN_BASE + 0];
    telem[TELEM_WAIT_SLOT_CY_LO] = (uint32_t)(wait_slot_cy & 0xFFFFFFFFu);
    telem[TELEM_WAIT_SLOT_CY_HI] = (uint32_t)(wait_slot_cy >> 32);
    telem[TELEM_ISSUE_CY_LO] = (uint32_t)(issue_cy & 0xFFFFFFFFu);
    telem[TELEM_ISSUE_CY_HI] = (uint32_t)(issue_cy >> 32);
    telem[TELEM_RING_WAIT_CY_LO] = (uint32_t)(ring_wait_cy & 0xFFFFFFFFu);
    telem[TELEM_RING_WAIT_CY_HI] = (uint32_t)(ring_wait_cy >> 32);
    telem[TELEM_T_KERNEL_START_LO] = (uint32_t)(t_kernel_start & 0xFFFFFFFFu);
    telem[TELEM_T_KERNEL_START_HI] = (uint32_t)(t_kernel_start >> 32);

    noc_async_writes_flushed();
    fabric_connections.close();

    // Stamped after the teardown so the kernel-total window really covers the whole kernel.
    const uint64_t t_kernel_end = wall_clock();
    telem[TELEM_T_KERNEL_END_LO] = (uint32_t)(t_kernel_end & 0xFFFFFFFFu);
    telem[TELEM_T_KERNEL_END_HI] = (uint32_t)(t_kernel_end >> 32);
    // Magic last: a reader that sees it knows every field above is committed.
    telem[TELEM_MAGIC] = TELEMETRY_MAGIC;
    // Kept from the original tail. These are plain local L1 stores, so this is not known to be required
    // for the host readback to see them — it is retained because it costs nothing at kernel exit and the
    // previously working version had it.
    noc_async_writes_flushed();
}
