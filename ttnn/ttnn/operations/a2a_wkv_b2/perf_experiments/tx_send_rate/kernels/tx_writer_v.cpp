// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx writer — tx_send_rate bake-off variant (perf_experiments/tx_send_rate).
//
// Same protocol as kernels/a2a_wkv_b2_tx_writer.cpp (ready / address table / batches / done). The connection
// is still the helper's policy (dataflow_kernel_lib::ccl::DirectConn, or MuxConn<N> under A2A_TX_MUX), but
// the packet headers and sends are driven RAW (tt_fabric linear set_state + the adapter's send calls) instead
// of through FabricStream / ScatterWriteChannel, so the send can be timed per step and re-shaped.
// Helper bypassed: FabricStreamSender::open / FabricStream::arm_inc / arm_unicast_write / arm_scatter_write /
// ScatterWriteChannel::write_scatter / FabricStream::flush / close. Why: write_scatter re-runs
// populate_unicast_scatter_write_fields per packet (per chunk: get_noc_address_components + safe_get_noc_addr
// re-encoding for the EDM-local NoC, plus chunk sizes/count/payload stores) = 208 cycles of the ~370-cycle
// per-packet send (measured, TXW_DIAG); the helper has no way to issue a packet whose addresses were encoded
// once ahead of time (TXW_PREHDR), nor to send a header that sits in front of its payload (TXW_COLO/PULLHDR).
//
// Variant flags (compile-time defines, host side tx_descriptor.TX_VARIANT):
//   TXW_DIAG       : wall-clock accumulators per send step, emitted once at the end as
//                    DeviceTimestampedData (no zones in the loop).
//   TXW_PREHDR     : the address table stores each packet's whole scatter command field block
//                    (4 noc addresses already in the EDM-local encoding), copied into the header with
//                    plain word stores (no per-chunk get_noc_address_components + safe_get_noc_addr).
//   TXW_STATEFUL   : payload + header + credit go out through the adapter's stateful send
//                    (setup_stateful_send_cmd_bufs once, send_current_slot_stateful_non_blocking per packet).
//   TXW_COLO       : each cb_tx_stage page is a packet slot [header | payload] (tx_reader_v TXR_COLO); every
//                    slot's header is armed once (copy of the armed scatter template) and only its 4 dst
//                    addresses + chunk count change per packet; header + payload leave in ONE stateful
//                    NoC write (adapter send_current_slot_stateful_non_blocking_from_address). Implies
//                    TXW_PREHDR + TXW_STATEFUL. No header ring, no mid-batch ring flush.
//   TXW_PULLHDR    : the address table becomes a HEADER table: one fully armed fabric packet header per
//                    packet (template armed once, replicated by local NoC copies, then each entry's 4
//                    EDM-encoded dst addresses written once). The tx reader (tx_reader_v TXR_PULLHDR)
//                    NoC-reads the packet's header from the table into the stage slot in front of the
//                    payload it pulls, so the send loop does no per-packet header work at all: one
//                    stateful [header | payload] NoC write + the credit per packet. Uses the colo slot
//                    layout; implies TXW_STATEFUL.
//   TXW_BATCH_SLOTS: one EDM free-slot check per batch (wait until >= k slots are free) instead of one
//                    poll (with its L1 invalidate) per packet.

#include <array>
#include <cstdint>
#include <utility>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp"  // DirectConn / MuxConn connection policies
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace lf = tt::tt_fabric::linear::experimental;
using tt::tt_fabric::common::experimental::UnicastAtomicIncUpdateMask;
using tt::tt_fabric::common::experimental::UnicastScatterWriteUpdateMask;
using tt::tt_fabric::common::experimental::UnicastWriteUpdateMask;

#ifdef TXW_DIAG
#define TCLK() (*reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#else
#define TCLK() 0u
#endif

void kernel_main() {
    constexpr uint32_t cb_tx_stage = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt_block = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t ring_size = get_compile_time_arg_val(6);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(7);
    constexpr uint32_t tx_batch = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tx_addrs = get_compile_time_arg_val(9);
    constexpr uint32_t cb_tx_meta = get_compile_time_arg_val(10);
    constexpr auto out_args = TensorAccessorArgs<11>();
#if defined(TXW_PULLHDR) && !defined(TXW_STATEFUL)
#define TXW_STATEFUL
#endif
#if defined(TXW_COLO) || defined(TXW_PULLHDR)
#define TXW_SLOTLAYOUT 1
#endif
#ifdef TXW_COLO
#ifndef TXW_PREHDR
#define TXW_PREHDR
#endif
#ifndef TXW_STATEFUL
#define TXW_STATEFUL
#endif
#endif
#ifdef TXW_SLOTLAYOUT
    constexpr uint32_t colo_ct = out_args.next_compile_time_args_offset();
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(colo_ct);
    constexpr uint32_t pay_off = get_compile_time_arg_val(colo_ct + 1);
    constexpr uint32_t stage_slots = get_compile_time_arg_val(colo_ct + 2);
    static_assert(pay_off >= sizeof(PACKET_HEADER_TYPE), "slot header region smaller than the packet header");
#endif

    static_assert(tiles_per_packet >= 1 && tiles_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS);

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t tiles_per_source_head = ring_size * Mt * Nt;
    constexpr uint32_t packet_bytes = tiles_per_packet * tile_bytes;
#ifdef TXW_SLOTLAYOUT
    constexpr uint32_t batch_pages = tx_batch;  // one page per packet slot
    constexpr uint32_t slot_stride = slot_bytes;
#else
    constexpr uint32_t batch_pages = tx_batch * tiles_per_packet;
    constexpr uint32_t slot_stride = packet_bytes;
#endif
    constexpr uint32_t headers_per_risc = NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;
    constexpr uint32_t other_headers = 2;
    static_assert(headers_per_risc > other_headers, "no packet header left for the scatter ring");
    constexpr uint32_t hdr_ring =
        tx_batch < headers_per_risc - other_headers ? tx_batch : headers_per_risc - other_headers;

    size_t arg_idx = 0;
#ifdef A2A_TX_MUX
    constexpr uint32_t mux_ct = out_args.next_compile_time_args_offset();
    constexpr uint8_t mux_num_buffers = get_compile_time_arg_val(mux_ct);
    constexpr size_t mux_buffer_bytes = get_compile_time_arg_val(mux_ct + 1);
    constexpr size_t mux_status_address = get_compile_time_arg_val(mux_ct + 2);
    constexpr size_t mux_termination_address = get_compile_time_arg_val(mux_ct + 3);
    constexpr uint32_t mux_num_clients = get_compile_time_arg_val(mux_ct + 4);
    dataflow_kernel_lib::ccl::MuxConn<mux_num_buffers> conn(
        arg_idx, mux_buffer_bytes, mux_status_address, mux_termination_address, mux_num_clients);
#else
    dataflow_kernel_lib::ccl::DirectConn conn(arg_idx, /*is_forward=*/true);  // starts the open handshake
#endif

    const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t peer_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t peer_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_ready_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_done_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_pos = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_heads = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t heads_base = arg_idx;

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    ccl_routing_utils::line_unicast_route_info_t route{};
    route.dst_mesh_id = static_cast<uint16_t>(dst_mesh_id);
    route.dst_chip_id = static_cast<uint16_t>(dst_chip_id);

    const uint64_t peer_ready_noc = safe_get_noc_addr(peer_noc_x, peer_noc_y, sem_ready_addr, 0);
    const uint64_t peer_done_noc = safe_get_noc_addr(peer_noc_x, peer_noc_y, sem_done_addr, 0);
    volatile tt_l1_ptr uint32_t* sem_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready_addr);
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done_addr);

    // open (helper: FabricStreamSender::open -> ConnT::open)
    conn.open();
    auto* snd = conn.sender();

    // inc header (helper: FabricStream::arm_inc(1))
    auto* inc_hdr = PacketHeaderPool::allocate_header();
    lf::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        inc_hdr, static_cast<uint8_t>(route.distance_in_hops), tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
    ccl_routing_utils::fabric_set_line_unicast_route(inc_hdr, route);
    auto send_inc = [&](uint64_t a) {
        lf::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            snd, inc_hdr, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{a, 0});
    };

    constexpr uint32_t full_block_packets = (mt_block * Nt + tiles_per_packet - 1) / tiles_per_packet;
    constexpr uint32_t last_block_tiles = (Mt - (num_mt_blocks - 1) * mt_block) * Nt;
    constexpr uint32_t packets_per_head =
        (num_mt_blocks - 1) * full_block_packets + (last_block_tiles + tiles_per_packet - 1) / tiles_per_packet;
    const uint32_t stream_packets = num_heads * packets_per_head;

    tt_l1_ptr uint64_t* addr_table = reinterpret_cast<tt_l1_ptr uint64_t*>(get_write_ptr(cb_tx_addrs));
#ifdef TXW_PULLHDR
    // Header table: entry j at table_base + j * pay_off (pay_off = aligned header size). Writer produces it
    // (cb_tx_addrs reserve -> push of stream_packets pages), the tx reader consumes it (NoC-reads entries).
    cb_reserve_back(cb_tx_addrs, stream_packets);
    const uint32_t table_base = get_write_ptr(cb_tx_addrs);
    {
        // arm entry 0 exactly as the helper arms a scatter channel (set_state + route), then replicate it
        // with local NoC copies that double the armed range each step (7 copies for 120 entries).
        volatile PACKET_HEADER_TYPE* h0 = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(table_base);
        uint64_t dummy[4] = {0, 0, 0, 0};
        uint16_t cs[3] = {tile_bytes, tile_bytes, tile_bytes};
        constexpr uint32_t arm_chunks0 = tiles_per_packet > 1 ? tiles_per_packet : 2;
        lf::fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            h0,
            static_cast<uint8_t>(route.distance_in_hops),
            tt::tt_fabric::NocUnicastScatterCommandHeader(dummy, cs, static_cast<uint8_t>(arm_chunks0)),
            static_cast<uint16_t>(tile_bytes * arm_chunks0));
        ccl_routing_utils::fabric_set_line_unicast_route(h0, route);
        for (uint32_t done = 1; done < stream_packets; done *= 2) {
            const uint32_t n = done < stream_packets - done ? done : stream_packets - done;
            noc_async_write(table_base, get_noc_addr(table_base + done * pay_off), n * pay_off);
            noc_async_write_barrier();
        }
    }
#endif
    uint32_t filled = 0;
    uint32_t f_mb = 0, f_s = 0, f_pp = 0;
    auto fill_next = [&]() {
        const uint32_t block_tiles = f_mb + 1 == num_mt_blocks ? last_block_tiles : mt_block * Nt;
        const uint32_t h_local = get_arg_val<uint32_t>(heads_base + f_s);
        const uint32_t tile0 =
            h_local * tiles_per_source_head + (my_pos * Mt + f_mb * mt_block) * Nt + f_pp * tiles_per_packet;
        const uint32_t chunks = (block_tiles - f_pp * tiles_per_packet) < tiles_per_packet
                                    ? (block_tiles - f_pp * tiles_per_packet)
                                    : tiles_per_packet;
#ifdef TXW_PULLHDR
        volatile PACKET_HEADER_TYPE* h = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(table_base + filled * pay_off);
#ifdef TXW_ABL_UNICAST
        // MICROBENCH (wrong output by design): every packet becomes ONE contiguous unicast write of the whole
        // payload into the destination's cb_out region (CB index = TXW_ABL_UNICAST) on the bank core of the
        // packet's first tile, to measure the router channel rate without the receiver's 4-chunk scatter.
        if (true) {
            const uint64_t a = tt::tt_fabric::linear::addrgen_detail::get_noc_address(out, tile0, 0);
            auto comps = get_noc_address_components(a);
            h->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
            h->command_fields.unicast_write.noc_address =
                safe_get_noc_addr(comps.first.x, comps.first.y, get_write_ptr(TXW_ABL_UNICAST), edm_to_local_chip_noc);
            h->payload_size_bytes = static_cast<uint16_t>(tile_bytes * chunks);
        } else
#endif
            if (chunks == 1) {
            // single-tile tail: a unicast write (the helper's arm_unicast_write + write); not on the focus shape
            const uint64_t a = tt::tt_fabric::linear::addrgen_detail::get_noc_address(out, tile0, 0);
            auto comps = get_noc_address_components(a);
            h->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
            h->command_fields.unicast_write.noc_address =
                safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
            h->payload_size_bytes = static_cast<uint16_t>(align(tile_bytes, l1_alignment));
        } else {
            for (uint32_t c = 0; c < NOC_SCATTER_WRITE_MAX_CHUNKS; ++c) {
                uint64_t v = 0;
                if (c < chunks) {
                    const uint64_t a = tt::tt_fabric::linear::addrgen_detail::get_noc_address(out, tile0 + c, 0);
                    auto comps = get_noc_address_components(a);
                    v = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
                }
                h->command_fields.unicast_scatter_write.noc_address[c] = v;
            }
            h->command_fields.unicast_scatter_write.chunk_count = chunks;
            h->payload_size_bytes = static_cast<uint16_t>(tile_bytes * chunks);
        }
#else
        tt_l1_ptr uint64_t* e = addr_table + filled * tiles_per_packet;
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint64_t a = tt::tt_fabric::linear::addrgen_detail::get_noc_address(out, tile0 + c, 0);
#ifdef TXW_PREHDR
            // the exact value populate_unicast_scatter_write_fields would compute per packet (scatter only;
            // a single-tile tail goes through the unicast with_state call, which encodes it itself)
            if (chunks > 1) {
                auto comps = get_noc_address_components(a);
                e[c] = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
            } else {
                e[c] = a;
            }
#else
            e[c] = a;
#endif
        }
#ifdef TXW_PREHDR
        for (uint32_t c = chunks; c < tiles_per_packet; ++c) {
            e[c] = 0;
        }
#endif
#endif  // TXW_PULLHDR
        ++filled;
        if (++f_pp * tiles_per_packet >= block_tiles) {
            f_pp = 0;
            if (++f_s == num_heads) {
                f_s = 0;
                ++f_mb;
            }
        }
    };

    send_inc(peer_ready_noc);
    {
        MaybeDeviceZoneScope("tx_writer_addr_table");
        while (filled < stream_packets) {
            fill_next();
        }
#ifdef TXW_PULLHDR
        cb_push_back(cb_tx_addrs, stream_packets);  // header table complete: the tx reader may pull from it
#endif
    }
    {
        MaybeDeviceZoneScope("tx_writer_await_ready");
        noc_semaphore_wait_min(sem_ready, 1);
        noc_semaphore_set(sem_ready, 0);
    }

    // unicast tail header (helper: arm_unicast_write(tile_bytes))
    auto* uni_hdr = PacketHeaderPool::allocate_header();
    lf::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        uni_hdr,
        static_cast<uint8_t>(route.distance_in_hops),
        nullptr,
        static_cast<uint16_t>(align(tile_bytes, l1_alignment)));
    ccl_routing_utils::fabric_set_line_unicast_route(uni_hdr, route);

    // scatter header ring (helper: arm_scatter_write(tile_bytes, tiles_per_packet) x hdr_ring)
    constexpr uint32_t arm_chunks = tiles_per_packet > 1 ? tiles_per_packet : 2;
    volatile PACKET_HEADER_TYPE* ring[hdr_ring];
    for (uint32_t r = 0; r < hdr_ring; ++r) {
        ring[r] = PacketHeaderPool::allocate_header();
        uint64_t dummy[4] = {0, 0, 0, 0};
        uint16_t cs[3] = {tile_bytes, tile_bytes, tile_bytes};
        lf::fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            ring[r],
            static_cast<uint8_t>(route.distance_in_hops),
            tt::tt_fabric::NocUnicastScatterCommandHeader(dummy, cs, static_cast<uint8_t>(arm_chunks)),
            static_cast<uint16_t>(tile_bytes * arm_chunks));
        ccl_routing_utils::fabric_set_line_unicast_route(ring[r], route);
    }

#ifdef TXW_COLO
    // Arm every stage slot's header once: byte copy of the armed scatter template (route, send type,
    // chunk sizes, payload size). Header bytes of a slot are never written by the tx reader (it pulls
    // behind them), and a slot's header is only re-written when the slot is sent again, i.e. after the
    // flush + pop that let the reader refill it: the same flush that frees the payload frees the header.
    {
        const uint32_t stage_base = get_read_ptr(cb_tx_stage);
        const volatile uint32_t* tpl = reinterpret_cast<const volatile uint32_t*>(ring[0]);
        for (uint32_t j = 0; j < stage_slots; ++j) {
            volatile uint32_t* h = reinterpret_cast<volatile uint32_t*>(stage_base + j * slot_bytes);
            for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; ++w) {
                h[w] = tpl[w];
            }
        }
    }
#endif
#ifdef TXW_STATEFUL
    snd->setup_stateful_send_cmd_bufs();
#endif

#ifdef TXW_DIAG
    uint32_t c_wait = 0, c_hdr = 0, c_slot = 0, c_pay = 0, c_hw = 0, c_flush = 0, n_slot_stall = 0;
    uint32_t n_batches = 0;
#endif

    // One scatter packet: fill the header's per-packet fields, wait for an EDM slot, payload + header
    // NoC writes into the slot, credit inline write (helper: ScatterWriteChannel::write_scatter).
    auto send_scatter = [&](volatile PACKET_HEADER_TYPE* hdr,
                            const uint64_t* dst,
                            uint32_t chunks,
                            uint32_t l1,
                            [[maybe_unused]] uint32_t i_in_batch) {
        [[maybe_unused]] uint32_t t0 = TCLK();
#ifdef TXW_PREHDR
        // dst holds the EDM-local encoded addresses + zero padding (filled once in the table)
        volatile uint32_t* hf =
            reinterpret_cast<volatile uint32_t*>(&hdr->command_fields.unicast_scatter_write.noc_address[0]);
        const uint32_t* sf = reinterpret_cast<const uint32_t*>(dst);
        for (uint32_t w = 0; w < 2 * NOC_SCATTER_WRITE_MAX_CHUNKS; ++w) {
            hf[w] = sf[w];
        }
        hdr->command_fields.unicast_scatter_write.chunk_count = chunks;
        hdr->payload_size_bytes = static_cast<uint16_t>(tile_bytes * chunks);
#else
        uint16_t cs[3] = {tile_bytes, tile_bytes, tile_bytes};
        populate_unicast_scatter_write_fields<
            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
            UnicastScatterWriteUpdateMask::PayloadSize>(
            hdr,
            static_cast<uint16_t>(tile_bytes * chunks),
            tt::tt_fabric::NocUnicastScatterCommandHeader(dst, cs, static_cast<uint8_t>(chunks)));
#endif
        [[maybe_unused]] uint32_t t1 = TCLK();
#ifdef TXW_BATCH_SLOTS
        if (i_in_batch >= snd->num_buffers_per_channel) {  // batch longer than the EDM channel
            snd->wait_for_empty_write_slot();
        }
#else
        snd->wait_for_empty_write_slot();
#endif
        [[maybe_unused]] uint32_t t2 = TCLK();
#if defined(TXW_COLO)
        snd->send_current_slot_stateful_non_blocking_from_address((uint32_t)hdr, pay_off + tile_bytes * chunks);
        [[maybe_unused]] uint32_t t3 = t2;
#elif defined(TXW_STATEFUL)
        snd->send_current_slot_stateful_non_blocking(l1, hdr->payload_size_bytes, (uint32_t)hdr);
        [[maybe_unused]] uint32_t t3 = t2;
#else
        snd->send_payload_without_header_non_blocking_from_address(l1, hdr->payload_size_bytes);
        [[maybe_unused]] uint32_t t3 = TCLK();
        snd->send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
#endif
        [[maybe_unused]] uint32_t t4 = TCLK();
#ifdef TXW_DIAG
        c_hdr += t1 - t0;
        c_slot += t2 - t1;
        n_slot_stall += (t2 - t1) > 40 ? 1 : 0;
        c_pay += t3 - t2;
        c_hw += t4 - t3;
#endif
    };

    {
        MaybeDeviceZoneScope("tx_writer_stream");
        for (uint32_t sent = 0; sent < stream_packets;) {
            [[maybe_unused]] uint32_t w0 = TCLK();
            cb_wait_front(cb_tx_meta, 1);
            cb_wait_front(cb_tx_stage, batch_pages);
            [[maybe_unused]] uint32_t w1 = TCLK();
#ifdef TXW_DIAG
            c_wait += w1 - w0;
            ++n_batches;
#endif
            const uint32_t base = get_read_ptr(cb_tx_stage);
            volatile tt_l1_ptr uint32_t* meta =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_tx_meta));
            const uint32_t k = meta[0];
            bool unicast_used = false;
#ifdef TXW_BATCH_SLOTS
            {
                [[maybe_unused]] uint32_t s0 = TCLK();
                const uint32_t need = k < snd->num_buffers_per_channel ? k : snd->num_buffers_per_channel;
                while (snd->get_num_free_write_slots() < need) {
                }
#ifdef TXW_DIAG
                c_slot += TCLK() - s0;
#endif
            }
#endif
#ifdef TXW_PULLHDR
            constexpr bool all_full =
                (mt_block * Nt) % tiles_per_packet == 0 && last_block_tiles % tiles_per_packet == 0;
            for (uint32_t i = 0; i < k; ++i) {
                [[maybe_unused]] uint32_t t1 = TCLK();
                const uint32_t l1 = base + i * slot_stride;
#ifdef TXW_BATCH_SLOTS
                if (i >= snd->num_buffers_per_channel) {
                    snd->wait_for_empty_write_slot();
                }
#else
                snd->wait_for_empty_write_slot();
#endif
                [[maybe_unused]] uint32_t t2 = TCLK();
                uint32_t bytes = pay_off + packet_bytes;
                if constexpr (!all_full) {
                    bytes = pay_off + reinterpret_cast<volatile PACKET_HEADER_TYPE*>(l1)->payload_size_bytes;
                }
                snd->send_current_slot_stateful_non_blocking_from_address(l1, bytes);
                [[maybe_unused]] uint32_t t4 = TCLK();
#ifdef TXW_DIAG
                c_slot += t2 - t1;
                n_slot_stall += (t2 - t1) > 40 ? 1 : 0;
                c_hw += t4 - t2;
#endif
            }
            if (false) {
#else
            {
#endif
                for (uint32_t i = 0; i < k; ++i) {
                    const uint32_t code = meta[1 + i];
                    const uint32_t mb = code >> 16;
                    const uint32_t s = (code >> 8) & 0xFF;
                    const uint32_t pp = code & 0xFF;
                    const uint32_t block_tiles = mb + 1 == num_mt_blocks ? last_block_tiles : mt_block * Nt;
                    const uint32_t packets_per_block = (block_tiles + tiles_per_packet - 1) / tiles_per_packet;
                    const uint32_t p = pp * tiles_per_packet;
                    const uint32_t chunks = (block_tiles - p) < tiles_per_packet ? (block_tiles - p) : tiles_per_packet;
                    const uint32_t l1 = base + i * slot_stride;
                    const uint32_t entry = mb * num_heads * full_block_packets + s * packets_per_block + pp;
                    const uint64_t* dst_addrs = addr_table + entry * tiles_per_packet;
                    if (chunks == 1) {
                        if (unicast_used) {
                            noc_async_writes_flushed();
                        }
                        lf::fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            snd, uni_hdr, l1, tt::tt_fabric::NocUnicastCommandHeader{dst_addrs[0]});
                        unicast_used = true;
                    } else {
#ifdef TXW_COLO
                        send_scatter(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(l1), dst_addrs, chunks, l1, i);
#else
                        const uint32_t r = i % hdr_ring;
                        if (r == 0 && i != 0) {
                            noc_async_writes_flushed();
                        }
                        send_scatter(ring[r], dst_addrs, chunks, l1, i);
#endif
                    }
                }
            }
            sent += k;
            [[maybe_unused]] uint32_t f0 = TCLK();
            noc_async_writes_flushed();
            [[maybe_unused]] uint32_t f1 = TCLK();
#ifdef TXW_DIAG
            c_flush += f1 - f0;
#endif
            cb_pop_front(cb_tx_stage, batch_pages);
            cb_pop_front(cb_tx_meta, 1);
        }
    }

    {
        MaybeDeviceZoneScope("tx_writer_close");
        send_inc(peer_done_noc);
        noc_async_write_barrier();  // helper: FabricStream::close -> drain -> ConnT::close
        noc_async_atomic_barrier();
        conn.close();
    }

#ifdef TXW_DIAG
    DeviceTimestampedData("TXD_PACKETS", stream_packets);
    DeviceTimestampedData("TXD_BATCHES", n_batches);
    DeviceTimestampedData("TXD_WAIT", c_wait);
    DeviceTimestampedData("TXD_HDR", c_hdr);
    DeviceTimestampedData("TXD_SLOT", c_slot);
    DeviceTimestampedData("TXD_SLOTSTALLS", n_slot_stall);
    DeviceTimestampedData("TXD_PAY", c_pay);
    DeviceTimestampedData("TXD_HW", c_hw);
    DeviceTimestampedData("TXD_FLUSH", c_flush);
    DeviceTimestampedData("TXD_NBUF", snd->num_buffers_per_channel);
#endif

    MaybeDeviceZoneScope("tx_writer_await_done");
    noc_semaphore_wait_min(sem_done, 1);
    noc_semaphore_set(sem_done, 0);
}
