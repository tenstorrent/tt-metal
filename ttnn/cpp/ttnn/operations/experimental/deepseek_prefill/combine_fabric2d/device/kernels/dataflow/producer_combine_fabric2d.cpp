// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Producer kernel — the op's only kernel. It owns the ONE fabric sender connection its eth channel
// allows (the L1 connection table is indexed by eth channel and the EDM stores a single worker_xy per
// channel, so a second core on the same channel would just hang) and executes exactly ONE movement
// descriptor: read this chip's input tokens [in_base_page, in_base_page + num_in_tokens) and fill the
// PEER chip's output tokens [out_base_page, out_base_page + num_out_tokens) with them, cycling the
// inputs, one fabric packet per token.
//
// Which movement a producer gets is decided host-side (see the program factory); from here it is just
// four compile-time numbers. Nothing in this kernel knows how many other producers exist or how the
// caller's regions are carved up.
//
// There is no receiver kernel and no application-level credit loop. Being a fabric destination requires
// no handshake — the far eth RISC writes to whatever address the packet header names — and output pages
// are write-once within a movement, so there is no ring to protect. The EDM sender channel refusing an
// empty write slot is the only backpressure, which means the send window IS the link-bounded rate.
//
// Steps:
//   1. PREFILL. Read the movement's num_in_tokens input pages into the L1 source ring with a local DRAM
//      read, before any timestamp is taken. What leaves this producer is therefore known content, which
//      is what makes the destination DRAM checkable.
//   2. SEND. For token t: stamp output page (out_base_page + t) into that ring slot's prebuilt header and
//      push the payload from source slot (t mod num_in_tokens).
//   3. DRAIN. Push num_buffers_per_channel - 1 header-only fillers and take one more empty write slot,
//      which forces every payload credit back from the far chip; see the block near the bottom.
//
// Telemetry: the producer stamps its counts and three wall-clock timestamps into a fixed 1 kB L1 region
// so bandwidth can be recovered after the run without re-profiling.

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
constexpr uint32_t TELEM_TOKEN_SIZE = 2;
constexpr uint32_t TELEM_NUM_IN_TOKENS = 3;
constexpr uint32_t TELEM_T_FIRST_SEND_LO = 4;
constexpr uint32_t TELEM_T_FIRST_SEND_HI = 5;
constexpr uint32_t TELEM_T_LAST_SEND_LO = 6;
constexpr uint32_t TELEM_T_LAST_SEND_HI = 7;
constexpr uint32_t TELEM_T_DRAINED_LO = 8;
constexpr uint32_t TELEM_T_DRAINED_HI = 9;
constexpr uint32_t TELEM_EDM_SLOTS = 10;
constexpr uint32_t TELEM_DRAIN_PACKETS = 11;
constexpr uint32_t TELEM_OUT_BASE_PAGE = 12;
constexpr uint32_t TELEM_WAIT_SLOT_CY_LO = 13;
constexpr uint32_t TELEM_WAIT_SLOT_CY_HI = 14;
constexpr uint32_t TELEM_ISSUE_CY_LO = 15;
constexpr uint32_t TELEM_ISSUE_CY_HI = 16;
constexpr uint32_t TELEMETRY_MAGIC = 0xCF2D0004u;

void kernel_main() {
    // The movement this producer was assigned: how many output tokens to write, how many input tokens to
    // cycle over, and the two base pages. Everything else about the caller's region layout is invisible.
    constexpr uint32_t num_out_tokens = get_compile_time_arg_val(0);
    constexpr uint32_t num_in_tokens = get_compile_time_arg_val(1);  // also the L1 source ring depth
    constexpr uint32_t token_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t peer_chip_id = get_compile_time_arg_val(3);
    constexpr uint32_t peer_mesh_id = get_compile_time_arg_val(4);
    constexpr uint32_t peer_noc_x = get_compile_time_arg_val(5);
    constexpr uint32_t peer_noc_y = get_compile_time_arg_val(6);
    constexpr uint32_t prod_buf_addr = get_compile_time_arg_val(7);
    constexpr uint32_t pkt_hdr_ring_addr = get_compile_time_arg_val(8);
    constexpr uint32_t pkt_hdr_drain_addr = get_compile_time_arg_val(9);
    // Uniform L1 address of the peer worker's 4-byte drain sink: the target of the drain's value-0
    // atomic increments. Nothing reads it, which is the point.
    constexpr uint32_t drain_sink_addr = get_compile_time_arg_val(10);
    constexpr uint32_t telemetry_addr = get_compile_time_arg_val(11);
    // Fine-grained stall buckets cost ~2 wall-clock register reads per token (a few percent of the
    // token's own cycles), so they are compile-time optional: on to explain a number, off to quote one.
    constexpr bool stall_telemetry = get_compile_time_arg_val(12) != 0;
    // The movement's two base pages, and the base addresses of the two interleaved DRAM buffers (uniform
    // across the mesh, so an accessor built from this chip's base produces addresses valid on the peer
    // chip too).
    constexpr uint32_t in_base_page = get_compile_time_arg_val(13);
    constexpr uint32_t out_base_page = get_compile_time_arg_val(14);
    constexpr uint32_t dram_out_base_addr = get_compile_time_arg_val(15);
    constexpr uint32_t dram_in_base_addr = get_compile_time_arg_val(16);
    constexpr auto dram_out_args = TensorAccessorArgs<17>();
    constexpr auto dram_in_args = TensorAccessorArgs<dram_out_args.next_compile_time_args_offset()>();

    volatile tt_l1_ptr uint32_t* telem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(telemetry_addr);
    // Invalidate first so a stale record from an earlier run can never be mistaken for this one's.
    telem[TELEM_MAGIC] = 0;

    std::size_t rt_args_idx = 0;
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
        rt_args_idx, num_connections);
    auto& sender = fabric_connections.get(0).sender;

    // One prebuilt header per ring slot. Only the destination page varies per token, so a slot's header
    // is not touched again until the ring wraps — which is what lets the payload send skip its flush.
    auto slot_hdr = [](uint32_t slot) -> volatile PACKET_HEADER_TYPE* {
        return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_ring_addr + slot * sizeof(PACKET_HEADER_TYPE));
    };
    for (uint32_t slot = 0; slot < num_in_tokens; slot++) {
        fabric_set_unicast_route(
            (volatile tt::tt_fabric::HybridMeshPacketHeader*)slot_hdr(slot), peer_chip_id, peer_mesh_id);
    }

    // ---- 1. Prefill the L1 source ring from this movement's input region (a local DRAM read; no fabric
    // ---- involved), before any timestamp is taken.
    {
        const auto dram_in = TensorAccessor(dram_in_args, dram_in_base_addr);
        for (uint32_t slot = 0; slot < num_in_tokens; slot++) {
            noc_async_read(
                dram_in.get_noc_addr(in_base_page + slot), prod_buf_addr + slot * token_size_bytes, token_size_bytes);
        }
        noc_async_read_barrier();
    }

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

    // ---- 2. Send. Each token lands on its own DRAM page, so the only backpressure is the EDM refusing
    // ---- an empty write slot. The send window is a LOWER bound on the transfer time; the drain below
    // ---- supplies the matching upper bound.
    {
        // One zone per token, plus this outer one. The profiler L1 buffer holds 250 optional markers per
        // RISC, so a second per-token zone would overflow and silently drop data.
        DeviceZoneScopedN("PRODUCER_LOOP");
        while (sent < num_out_tokens) {
            const uint32_t out_page = out_base_page + sent;
            // Output token t carries input token t mod num_in_tokens — the movement's whole contract. The
            // header ring is indexed the same way, which costs nothing and keeps a header's reuse distance
            // at a full num_in_tokens.
            const uint32_t src_slot = sent % num_in_tokens;
            volatile PACKET_HEADER_TYPE* hdr = slot_hdr(src_slot);
            // Header first, THEN wait for the slot — building it while the EDM may still be busy is free
            // overlap, and reversing the two costs ~8% of the bandwidth. Plain unicast write, NOT the
            // fused write+atomic-inc, which is documented to hang Blackhole on a DRAM destination
            // (moe_utils.hpp:486).
            tt::tt_fabric::linear::to_noc_unicast_write(token_size_bytes, hdr, out_page, dram_out);
            const uint64_t w0 = wall_clock();
            if (sent == 0) {
                t_first_send = w0;
            }
            sender.wait_for_empty_write_slot();
            uint64_t w1 = 0;
            if constexpr (stall_telemetry) {
                w1 = wall_clock();
                wait_slot_cy += w1 - w0;
            }
            sender.send_payload_without_header_non_blocking_from_address(
                prod_buf_addr + src_slot * token_size_bytes, token_size_bytes);
            // No flush: this slot's header is not touched again until the ring wraps, and we never write
            // the source payload at all. Letting the NoC queue hold the writes is what allows token N+1
            // to be issued while N is still draining.
            //
            // CAVEAT: the production idiom (moe_utils.hpp) flush-blocks here, which also orders our
            // payload write ahead of the EDM slot-credit write that post_send_payload_increment_pointers
            // issues on the sync cmd buf. Dropping the flush leans on the NoC keeping those in order to
            // the same destination. The op's content check is what makes that safe to rely on: a torn
            // packet would show up as a wrong output page.
            sender.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
            sent++;
            t_last_send = wall_clock();
            if constexpr (stall_telemetry) {
                issue_cy += t_last_send - w1;
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
    // acked — fabric_erisc_router.cpp:1690). So after num_out_tokens payload packets, writing D-1 more and
    // then obtaining one further free slot forces credits_returned >= num_out_tokens, i.e. every payload
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
    telem[TELEM_NUM_IN_TOKENS] = num_in_tokens;
    telem[TELEM_T_FIRST_SEND_LO] = (uint32_t)(t_first_send & 0xFFFFFFFFu);
    telem[TELEM_T_FIRST_SEND_HI] = (uint32_t)(t_first_send >> 32);
    telem[TELEM_T_LAST_SEND_LO] = (uint32_t)(t_last_send & 0xFFFFFFFFu);
    telem[TELEM_T_LAST_SEND_HI] = (uint32_t)(t_last_send >> 32);
    telem[TELEM_T_DRAINED_LO] = (uint32_t)(t_drained & 0xFFFFFFFFu);
    telem[TELEM_T_DRAINED_HI] = (uint32_t)(t_drained >> 32);
    telem[TELEM_EDM_SLOTS] = sender.num_buffers_per_channel;
    telem[TELEM_DRAIN_PACKETS] = drain_packets;
    telem[TELEM_OUT_BASE_PAGE] = out_base_page;
    telem[TELEM_WAIT_SLOT_CY_LO] = (uint32_t)(wait_slot_cy & 0xFFFFFFFFu);
    telem[TELEM_WAIT_SLOT_CY_HI] = (uint32_t)(wait_slot_cy >> 32);
    telem[TELEM_ISSUE_CY_LO] = (uint32_t)(issue_cy & 0xFFFFFFFFu);
    telem[TELEM_ISSUE_CY_HI] = (uint32_t)(issue_cy >> 32);
    // Magic last: a reader that sees it knows every field above is committed.
    telem[TELEM_MAGIC] = TELEMETRY_MAGIC;

    noc_async_writes_flushed();
    fabric_connections.close();
}
