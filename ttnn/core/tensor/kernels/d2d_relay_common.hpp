// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Helpers shared by the exit and entry relay kernels; the sender kernel uses the page streaming.
// Sizes are template parameters because a relay copies many tensors.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"

namespace d2d_relay {

struct MuxWaitState {
    uint8_t mux_x = 0;
    uint8_t mux_y = 0;
    uint32_t mux_status_addr = 0;
    uint32_t local_status_addr = 0;
    bool ready = false;
};

// Build either a direct EDM router connection (ViaMux == false, 3 RT args) or a
// FabricMux client connection (ViaMux == true, 15 RT args), returning the same
// WorkerToFabricEdmSender type.
template <bool ViaMux>
FORCE_INLINE tt::tt_fabric::WorkerToFabricEdmSender build_connection(size_t& arg, MuxWaitState& wait_state) {
    if constexpr (!ViaMux) {
        return tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    } else {
        const uint8_t mux_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg++));
        const uint8_t mux_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg++));
        const uint8_t ch_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg++));
        const uint8_t num_bufs = static_cast<uint8_t>(get_arg_val<uint32_t>(arg++));
        const size_t buf_bytes = get_arg_val<uint32_t>(arg++);
        const size_t ch_base = get_arg_val<uint32_t>(arg++);
        const size_t conn_info = get_arg_val<uint32_t>(arg++);
        const size_t conn_hs = get_arg_val<uint32_t>(arg++);
        const size_t flow_ctrl = get_arg_val<uint32_t>(arg++);
        const size_t buf_idx = get_arg_val<uint32_t>(arg++);
        const uint32_t status_addr = get_arg_val<uint32_t>(arg++);
        const uint32_t local_fc = get_arg_val<uint32_t>(arg++);
        const uint32_t local_td = get_arg_val<uint32_t>(arg++);
        const uint32_t local_bi = get_arg_val<uint32_t>(arg++);
        const uint32_t local_st = get_arg_val<uint32_t>(arg++);
        wait_state = {mux_x, mux_y, status_addr, local_st, false};
        return tt::tt_fabric::build_connection_to_fabric_endpoint<0>(
            mux_x,
            mux_y,
            ch_id,
            num_bufs,
            buf_bytes,
            ch_base,
            conn_info,
            conn_hs,
            flow_ctrl,
            buf_idx,
            local_fc,
            local_td,
            local_bi);
    }
}

template <bool ViaMux>
FORCE_INLINE void open_connection(tt::tt_fabric::WorkerToFabricEdmSender& conn, MuxWaitState& wait_state) {
    if constexpr (ViaMux) {
        if (!wait_state.ready) {
            tt::tt_fabric::wait_for_fabric_endpoint_ready(
                wait_state.mux_x, wait_state.mux_y, wait_state.mux_status_addr, wait_state.local_status_addr);
            wait_state.ready = true;
        }
    }
    conn.open<ViaMux>();
}

// go_slot values that are not slots: at the start and end of a lease window the master tells
// the sub to open or close its fabric connections. Normal work items pack (slice << 8) | slot.
constexpr uint32_t kGoOpenConns = 0xFFFFFFFEu;
constexpr uint32_t kGoCloseConns = 0xFFFFFFFFu;

FORCE_INLINE constexpr uint32_t pack_go_slot_slice(uint32_t slot, uint32_t slice) {
    return (slice << 8) | (slot & 0xFFu);
}
FORCE_INLINE constexpr uint32_t unpack_go_slot(uint32_t packed) { return packed & 0xFFu; }
FORCE_INLINE constexpr uint32_t unpack_go_slice(uint32_t packed) { return packed >> 8; }

// Split a contiguous page range [begin, end) across 1 or 2 lanes (master gets the extra page on odd length).
FORCE_INLINE void lane_page_range(
    uint32_t begin, uint32_t end, bool master, bool two_lanes, uint32_t& out_start, uint32_t& out_end) {
    if (!two_lanes) {
        out_start = begin;
        out_end = end;
    } else {
        const uint32_t mid = begin + ((end - begin + 1u) >> 1);
        out_start = master ? begin : mid;
        out_end = master ? mid : end;
    }
}

// Partition [0, TotalPages) into NumSlices contiguous slices, then split slice `s` across 1 or 2
// lanes so the union of both lanes in slice `s` is always the exact contiguous range [sb, se),
// regardless of whether an upstream or downstream hop uses 1 lane or 2 lanes.
template <uint32_t TotalPages, uint32_t NumSlices>
FORCE_INLINE void slice_page_range(uint32_t s, bool master, bool two_lanes, uint32_t& out_start, uint32_t& out_end) {
    static_assert(NumSlices >= 1, "num_slices must be >= 1");
    constexpr uint32_t kQuot = TotalPages / NumSlices;
    constexpr uint32_t kRem = TotalPages % NumSlices;
    const uint32_t sb = (s * kQuot) + (s < kRem ? s : kRem);
    const uint32_t s1 = s + 1u;
    const uint32_t se = (s1 * kQuot) + (s1 < kRem ? s1 : kRem);
    lane_page_range(sb, se, master, two_lanes, out_start, out_end);
}

// Write `size` bytes from L1 to a remote NoC address over fabric, in MaxPayload packets.
// A nonzero inc_addr fuses a flushed atomic increment into the last packet ("all data landed").
template <uint32_t MaxPayload>
FORCE_INLINE void fabric_write_bytes(
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    uint64_t dst_addr,
    uint32_t l1_src_addr,
    uint32_t size,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint64_t inc_addr = 0) {
    uint32_t remaining = size;
    while (remaining > 0) {
        const uint32_t packet_size = remaining > MaxPayload ? MaxPayload : remaining;
        if (inc_addr != 0 && packet_size == remaining) {
            header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_addr, inc_addr, 1, true}, packet_size);
        } else {
            header->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, packet_size);
        }
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(l1_src_addr, packet_size);
        conn.send_payload_flush_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        dst_addr += packet_size;
        l1_src_addr += packet_size;
        remaining -= packet_size;
    }
}

// Atomic-increment a remote L1 word by 1 over fabric. flush = true makes the far router
// write every earlier packet on this connection before it applies the increment, so the
// increment means "all my data has landed".
FORCE_INLINE void fabric_atomic_inc(
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    uint64_t dst_noc_addr,
    bool flush,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
    header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dst_noc_addr, 1, flush});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
}

// Point a packet header at chip `chip_id` of mesh `mesh_id`. 2D fabric only. A template so the check
// fires only where a kernel routes by chip id: the sender includes this header on any fabric.
template <typename Header>
FORCE_INLINE void set_route(volatile tt_l1_ptr Header* header, uint32_t chip_id, uint32_t mesh_id) {
    static_assert(
        std::is_same_v<Header, tt::tt_fabric::HybridMeshPacketHeader>,
        "D2D relay mode needs a 2D fabric (HybridMeshPacketHeader)");
    fabric_set_unicast_route(header, chip_id, mesh_id);
}

constexpr uint32_t floor_pow2(uint32_t n) { return n == 0 ? 0 : (1u << (31 - __builtin_clz(n))); }

// Wrap-safe "counter advanced by at least n since last".
FORCE_INLINE bool advanced(uint32_t now, uint32_t last, uint32_t n) { return (now - last) >= n; }

FORCE_INLINE bool terminated(uint32_t termination_addr, uint32_t terminate_value) {
    invalidate_l1_cache();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_addr)[0] == terminate_value;
}

// The host's packed_slot_bytes caps groups at the same chunk count (kMaxPagesPerPacket).
static_assert(NOC_SCATTER_WRITE_MAX_CHUNKS == 4, "keep kMaxPagesPerPacket in d2d_stream_service.cpp in sync");

// Pages per fabric packet: the most that fit both MaxPayload and one ring slot (SlotBytes), capped
// by the scatter chunk limit, as a power of 2. Per-packet router and flush costs dominate small
// pages, so tile pages ride several to a packet. A scatter chunk is sourced at payload base + the
// earlier chunk sizes, so pages off 16 B stay one per packet.
template <uint32_t MaxPayload, uint32_t SlotBytes>
FORCE_INLINE uint32_t pages_per_packet(uint32_t page_size) {
    if (page_size == 0 || page_size % 16 != 0) {
        return 1;
    }
    constexpr uint32_t cap = MaxPayload < SlotBytes ? MaxPayload : SlotBytes;
    uint32_t fit = cap / page_size;
    fit = fit < NOC_SCATTER_WRITE_MAX_CHUNKS ? fit : NOC_SCATTER_WRITE_MAX_CHUNKS;
    return fit < 2 ? 1 : floor_pow2(fit);
}

// Write `count` pages that sit back to back in L1 at l1_src_addr, starting at page `first`, as
// scatter packets (up to min(NOC_SCATTER_WRITE_MAX_CHUNKS, MaxPayload / page_size) pages per packet);
// a single page (or oversized page > MaxPayload) goes through fabric_write_bytes, and a nonzero
// inc_addr is fused into the final page.
template <uint32_t MaxPayload, typename OutAcc>
FORCE_INLINE void fabric_write_pages(
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const OutAcc& out_acc,
    uint32_t first,
    uint32_t count,
    uint32_t l1_src_addr,
    uint32_t page_size,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint64_t inc_addr = 0) {
    const uint32_t fit_payload = page_size > 0 ? (MaxPayload / page_size) : 0u;
    const uint32_t max_scatter =
        fit_payload < NOC_SCATTER_WRITE_MAX_CHUNKS ? fit_payload : static_cast<uint32_t>(NOC_SCATTER_WRITE_MAX_CHUNKS);
    uint32_t remaining = count;
    while (remaining > 0) {
        if (remaining == 1 || max_scatter < 2) {
            const uint64_t pkt_inc = (remaining == 1) ? inc_addr : 0;
            fabric_write_bytes<MaxPayload>(conn, out_acc.get_noc_addr(first), l1_src_addr, page_size, header, pkt_inc);
            first += 1;
            l1_src_addr += page_size;
            remaining -= 1;
            continue;
        }
        // Leave the final page for fabric_write_bytes when carrying a fused atomic increment.
        const uint32_t scatter_budget = (inc_addr != 0 && remaining <= max_scatter) ? (remaining - 1) : remaining;
        if (scatter_budget == 1) {
            fabric_write_bytes<MaxPayload>(conn, out_acc.get_noc_addr(first), l1_src_addr, page_size, header, 0);
            first += 1;
            l1_src_addr += page_size;
            remaining -= 1;
            continue;
        }
        const uint32_t chunk = scatter_budget <= max_scatter ? scatter_budget : max_scatter;
        uint64_t dst[NOC_SCATTER_WRITE_MAX_CHUNKS];
        uint16_t sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1];
        for (uint32_t i = 0; i < chunk; ++i) {
            dst[i] = out_acc.get_noc_addr(first + i);
            if (i + 1 < chunk) {
                sizes[i] = static_cast<uint16_t>(page_size);
            }
        }
        const uint32_t pkt_bytes = chunk * page_size;
        header->to_noc_unicast_scatter_write(
            tt::tt_fabric::NocUnicastScatterCommandHeader(dst, sizes, static_cast<uint8_t>(chunk)), pkt_bytes);
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(l1_src_addr, pkt_bytes);
        conn.send_payload_flush_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        first += chunk;
        l1_src_addr += pkt_bytes;
        remaining -= chunk;
    }
}

// Copy pages [page_start, page_end) to another chip over fabric, reading up to Depth packets of
// pages ahead (one NoC transaction id per ring slot of SlotBytes). The sender kernel streams with it
// too. A nonzero inc_addr is fused into the last packet, which then carries the last page alone.
template <uint32_t MaxPayload, uint32_t SlotBytes, uint32_t Depth, uint32_t TridBase, typename InAcc, typename OutAcc>
FORCE_INLINE void stream_sized_pages(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    uint32_t cb_l1_addr,
    uint32_t page_start,
    uint32_t page_end,
    uint32_t page_size,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint64_t inc_addr = 0) {
    static_assert(Depth > 0 && (Depth & (Depth - 1)) == 0, "ring depth must be a power of 2");
    const uint32_t n = page_end - page_start;
    const uint32_t pack = pages_per_packet<MaxPayload, SlotBytes>(page_size);
    const uint32_t group_bytes = pack * page_size;
    const uint32_t groups = (n + pack - 1) / pack;
    CoreLocalMem<uint32_t> scratch(cb_l1_addr);
    auto read_group = [&](uint32_t g) {
        const uint32_t slot = g & (Depth - 1);
        const uint32_t first = g * pack;
        const uint32_t count = n - first < pack ? n - first : pack;
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read<NocOptions::TXN_ID>(
                in_acc,
                scratch,
                page_size,
                {.page_id = page_start + first + i},
                {.offset_bytes = (slot * group_bytes) + (i * page_size)},
                NocOptVals{.trid = TridBase + slot});
        }
    };
    const uint32_t prime = Depth < groups ? Depth : groups;
    for (uint32_t g = 0; g < prime; ++g) {
        read_group(g);
    }
    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t slot = g & (Depth - 1);
        noc.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = TridBase + slot});
        const uint32_t first = g * pack;
        const uint32_t count = n - first < pack ? n - first : pack;
        const uint32_t src = cb_l1_addr + (slot * group_bytes);
        if (g + 1 < groups || inc_addr == 0) {
            fabric_write_pages<MaxPayload>(conn, out_acc, page_start + first, count, src, page_size, header);
        } else {
            if (count > 1) {
                fabric_write_pages<MaxPayload>(conn, out_acc, page_start + first, count - 1, src, page_size, header);
            }
            fabric_write_pages<MaxPayload>(
                conn,
                out_acc,
                page_start + first + count - 1,
                1,
                src + ((count - 1) * page_size),
                page_size,
                header,
                inc_addr);
        }
        // The flush in fabric_write_pages has read the slot out of L1, so it can be refilled.
        if (g + Depth < groups) {
            read_group(g + Depth);
        }
    }
}

// The primary tensor's page size is a compile-time constant, so its slot offsets fold.
template <
    uint32_t PageSize,
    uint32_t MaxPayload,
    uint32_t SlotBytes,
    uint32_t Depth,
    uint32_t TridBase,
    typename InAcc,
    typename OutAcc>
FORCE_INLINE void stream_pages(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    uint32_t cb_l1_addr,
    uint32_t page_start,
    uint32_t page_end,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint64_t inc_addr = 0) {
    stream_sized_pages<MaxPayload, SlotBytes, Depth, TridBase>(
        noc, conn, in_acc, out_acc, cb_l1_addr, page_start, page_end, PageSize, header, inc_addr);
}

// Extra segments (D2DStreamConfig::extra_tensors) ride with the primary tensor in every copy.
// Their CT args follow the primary accessor's at CtOffset: kRelayExtraSegmentCtArgs words per segment
// [page_size, total_pages], then one accessor block shared by all of them (they are DRAM interleaved).
// The caller's RT args at addr_arg hold a (src, dst) address pair per segment.
constexpr uint32_t kRelayExtraSegmentCtArgs = 2;

// NumExtra is a template parameter so that `if constexpr` defers the accessor block: with no
// extra segments it would sit past the last CT arg.
template <
    uint32_t NumExtra,
    uint32_t CtOffset,
    uint32_t MaxPayload,
    uint32_t SlotBytes,
    uint32_t Depth,
    uint32_t TridBase>
FORCE_INLINE void stream_extra_segments(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    uint32_t cb_l1_addr,
    size_t addr_arg,
    bool master,
    bool two_lanes,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
    if constexpr (NumExtra > 0) {
        constexpr auto seg_args = TensorAccessorArgs<CtOffset + (kRelayExtraSegmentCtArgs * NumExtra)>();
        for (uint32_t s = 0; s < NumExtra; ++s) {
            const uint32_t ct = CtOffset + (kRelayExtraSegmentCtArgs * s);
            const uint32_t seg_page_size = kernel_compile_time_args[ct];
            const uint32_t total = kernel_compile_time_args[ct + 1];
            uint32_t p_start = 0;
            uint32_t p_end = 0;
            lane_page_range(0u, total, master, two_lanes, p_start, p_end);
            const auto in_acc = TensorAccessor(seg_args, get_arg_val<uint32_t>(addr_arg + (2 * s)), seg_page_size);
            const auto out_acc = TensorAccessor(seg_args, get_arg_val<uint32_t>(addr_arg + (2 * s) + 1), seg_page_size);
            stream_sized_pages<MaxPayload, SlotBytes, Depth, TridBase>(
                noc, conn, in_acc, out_acc, cb_l1_addr, p_start, p_end, seg_page_size, header);
        }
    }
}

// Copy pages [page_start, page_end), then md_size bytes of metadata from md_l1_addr to md_dst
// (none when md_size is 0), and increment inc_addr by 1 in the same packet as the last write.
// With nothing to write, the increment goes alone with the same flush.
template <
    uint32_t PageSize,
    uint32_t MaxPayload,
    uint32_t SlotBytes,
    uint32_t Depth,
    uint32_t TridBase,
    typename InAcc,
    typename OutAcc>
FORCE_INLINE void stream_pages_then_inc(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    uint32_t cb_l1_addr,
    uint32_t page_start,
    uint32_t page_end,
    uint64_t md_dst,
    uint32_t md_l1_addr,
    uint32_t md_size,
    uint64_t inc_addr,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
    if (md_size > 0) {
        stream_pages<PageSize, MaxPayload, SlotBytes, Depth, TridBase>(
            noc, conn, in_acc, out_acc, cb_l1_addr, page_start, page_end, header);
        fabric_write_bytes<MaxPayload>(conn, md_dst, md_l1_addr, md_size, header, inc_addr);
    } else if (page_end > page_start) {
        stream_pages<PageSize, MaxPayload, SlotBytes, Depth, TridBase>(
            noc, conn, in_acc, out_acc, cb_l1_addr, page_start, page_end, header, inc_addr);
    } else {
        fabric_atomic_inc(conn, inc_addr, true, header);
    }
}

// Master side of the two-lane sync over the go_slot / go_count / done_count L1 words. Each go
// is a slot to copy, kGoOpenConns or kGoCloseConns; the sub acks every go with done_count, and
// post() waits for the previous ack first, so go_slot is never overwritten before it is read.
struct SubLane {
    uint32_t go_slot_addr;
    uint32_t go_count_addr;
    uint32_t done_count_addr;
    uint32_t termination_addr;
    uint32_t terminate_value;
    uint32_t go_local = 0;
    uint32_t last_done = 0;
    bool busy = false;

    // Wait for the ack of the last go. Gives up on termination: the sub may have exited.
    FORCE_INLINE void wait() {
        if (!busy) {
            return;
        }
        auto* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_count_addr);
        while (!advanced(done[0], last_done, 1)) {
            if (terminated(termination_addr, terminate_value)) {
                return;
            }
        }
        last_done += 1;
        busy = false;
    }

    FORCE_INLINE void post(uint32_t what) {
        wait();
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_slot_addr)[0] = what;
        go_local += 1;
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_count_addr)[0] = go_local;
        busy = true;
    }
};

// Sub side: run each go from the master and ack it. A go is checked before termination, so a
// posted close is never skipped; on termination the sub closes its connections if still open.
template <typename OpenFn, typename CloseFn, typename CopyFn>
FORCE_INLINE void run_sub_lane(
    uint32_t go_slot_addr,
    uint32_t go_count_addr,
    uint32_t done_count_addr,
    uint32_t termination_addr,
    uint32_t terminate_value,
    OpenFn open_conns,
    CloseFn close_conns,
    CopyFn copy) {
    auto* go_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_count_addr);
    auto* go_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_slot_addr);
    auto* done_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_count_addr);
    uint32_t last_go = 0;
    bool conns_open = false;
    while (true) {
        invalidate_l1_cache();
        if (!advanced(go_count[0], last_go, 1)) {
            if (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_addr)[0] == terminate_value) {
                break;
            }
            continue;
        }
        last_go += 1;
        const uint32_t what = go_slot[0];
        if (what == kGoOpenConns) {
            open_conns();
            conns_open = true;
        } else if (what == kGoCloseConns) {
            close_conns();
            conns_open = false;
        } else {
            copy(what);
        }
        done_count[0] += 1;
    }
    if (conns_open) {
        close_conns();
    }
}

}  // namespace d2d_relay
