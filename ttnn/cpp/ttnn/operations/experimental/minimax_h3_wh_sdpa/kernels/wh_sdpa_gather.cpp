// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The WH SDPA op's fused K/V gather around the SP ring (wh_sdpa_op_common.h, "Ring gather"), on the otherwise idle
// NCRISC of eight SDPA cores of each chip: link l (0-3) of each direction. The chip's own K and V^T shard ([heads]
// [keys / ring][128] and [heads][128][keys / ring] tiles) go head by head into the full K and V^T of every chip of the
// ring, at the chip's slice of the keys, while the SDPA runs: each sender reads its share of a head (2-tile packets
// p = l, l + links, ...) from DRAM into L1 and multicasts it in its direction (fabric 1D, hops 1 .. range; the
// diametric chip comes from the forward senders for even heads and the backward ones for odd heads); the forward
// senders also write it to the chip's own full K and V^T. After a head each sender increments the head's landing
// counter on every chip it wrote (fabric atomic increments behind the data, flushed at each hop; locally a NoC atomic
// after a write barrier). The feed reads a head's blocks once its counter shows every sender of the launch.
//
// Counters never reset: launch L of the op (counted on the host per counter buffer) expects WHS_RING_SENDS x L on
// each head's counter and (ring - 1) x L on the barrier, which every chip's forward and backward link-0 senders
// increment on the others at the launch's start: a chip writes into the others' full K and V^T only once every chip
// has started the launch (its earlier programs done, so nothing else still uses those buffers).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "wh_sdpa_op_common.h"

using namespace tt::tt_fabric::linear::experimental;

namespace {

constexpr uint32_t batch = WHS_GATHER_BATCH;  // packets a batch; two batches in flight
constexpr uint32_t packet_bytes = 2u * WHS_TILE;

}  // namespace

void kernel_main() {
    size_t a = 0;
    const uint32_t k_local = get_arg_val<uint32_t>(a++);
    const uint32_t vt_local = get_arg_val<uint32_t>(a++);
    const uint32_t k_full = get_arg_val<uint32_t>(a++);
    const uint32_t vt_full = get_arg_val<uint32_t>(a++);
    const uint32_t heads = get_arg_val<uint32_t>(a++);
    const uint32_t local_rows = get_arg_val<uint32_t>(a++);  // key tile rows of the shard
    const uint32_t position = get_arg_val<uint32_t>(a++);    // this chip's slice of the keys
    const uint32_t ring = get_arg_val<uint32_t>(a++);
    const uint32_t link = get_arg_val<uint32_t>(a++);
    const uint32_t links = get_arg_val<uint32_t>(a++);
    const uint32_t backward = get_arg_val<uint32_t>(a++);
    const uint32_t counter_x = get_arg_val<uint32_t>(a++);  // the counter core, NoC 0 coordinates
    const uint32_t counter_y = get_arg_val<uint32_t>(a++);
    const uint32_t counters = get_arg_val<uint32_t>(a++);  // L1 address: head h's counter at + h x 16, barrier at
                                                           // + WHS_RING_BARRIER x 16
    const uint32_t launch = get_arg_val<uint32_t>(a++);
    const uint32_t stage = get_arg_val<uint32_t>(a++);  // L1: 2 x batch packets, then the packet headers
    const uint32_t debug = get_arg_val<uint32_t>(a++);

    const InterleavedAddrGen<true> k_src = {.bank_base_address = k_local, .page_size = WHS_TILE};
    const InterleavedAddrGen<true> vt_src = {.bank_base_address = vt_local, .page_size = WHS_TILE};
    const InterleavedAddrGen<true> k_dst = {.bank_base_address = k_full, .page_size = WHS_TILE};
    const InterleavedAddrGen<true> vt_dst = {.bank_base_address = vt_full, .page_size = WHS_TILE};

    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    sender.open();

    const uint32_t headers = stage + 2u * batch * packet_bytes;
    const auto header = [&](uint32_t i) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(headers + i * WHS_HEADER_BYTES);
    };
    // Headers 0 .. 2 batch - 1: one a staging slot; 2 batch: the atomic increments.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* inc_header = header(2u * batch);
    for (uint32_t i = 0; i <= 2u * batch; ++i) {
        for (uint32_t w = 0; w < WHS_HEADER_BYTES / 4u; ++w) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(headers + i * WHS_HEADER_BYTES)[w] = 0u;
        }
    }
    const uint64_t counter_noc = get_noc_addr(counter_x, counter_y, counters);
    const auto counter_word = [&](uint32_t line) -> uint64_t { return counter_noc + line * 16u; };
    const auto fabric_counter = [&](uint32_t line) -> uint64_t {
        return safe_get_noc_addr(
            static_cast<uint8_t>(counter_x), static_cast<uint8_t>(counter_y), counters + line * 16u, 0);
    };

    // Hops this sender covers for head h: the forward direction takes the diametric chip on even heads.
    const uint32_t half = ring / 2u;
    const auto range_of = [&](uint32_t h) -> uint8_t {
        const bool takes_diametric = (h % 2u == 0u) != (backward != 0u);
        return static_cast<uint8_t>(takes_diametric ? half : ring - 1u - half);
    };

    // Barrier: tell every other chip this launch has started, then wait for all of them.
    if (link == 0u) {
        const uint8_t r = static_cast<uint8_t>(backward ? ring - 1u - half : half);
        fabric_multicast_noc_unicast_atomic_inc(
            &sender,
            inc_header,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_counter(WHS_RING_BARRIER), 1u, false},
            1u,
            r);
        noc_async_writes_flushed();
    }
    {
        const uint32_t probe = stage;  // a word of the staging area, before any data lands there
        const uint32_t want = (ring - 1u) * launch;
        while (true) {
            noc_async_read(counter_word(WHS_RING_BARRIER), probe, 4u);
            noc_async_read_barrier();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe) - want < 0x80000000u) {
                break;
            }
            for (uint32_t i = 0; i < 64u; ++i) {
                __asm__ volatile("nop");
            }
        }
    }

    // A head's tiles: K (rows x 4), then V^T (4 x rows); packet p holds tiles 2p and 2p + 1.
    const uint32_t head_tiles = 8u * local_rows;
    const uint32_t full_rows = local_rows * ring;
    const auto src_addr = [&](uint32_t h, uint32_t t) -> uint64_t {
        if (t < 4u * local_rows) {
            return k_src.get_noc_addr((h * local_rows + t / 4u) * 4u + t % 4u);
        }
        t -= 4u * local_rows;
        return vt_src.get_noc_addr((h * 4u + t / local_rows) * local_rows + t % local_rows);
    };
    const auto dst_page = [&](uint32_t h, uint32_t t, bool& is_k) -> uint32_t {
        if (t < 4u * local_rows) {
            is_k = true;
            return (h * full_rows + position * local_rows + t / 4u) * 4u + t % 4u;
        }
        is_k = false;
        t -= 4u * local_rows;
        return (h * 4u + t / local_rows) * full_rows + position * local_rows + t % local_rows;
    };
    const bool local_copy = !backward;
    const bool no_fabric = debug & 1u;

    const uint32_t packets = head_tiles / 2u;
    const uint32_t mine = packets > link ? (packets - link + links - 1u) / links : 0u;
    for (uint32_t h = 0; h < heads; ++h) {
        const uint8_t range = range_of(h);
        const uint32_t batches = (mine + batch - 1u) / batch;
        const auto read_batch = [&](uint32_t b) {
            const uint32_t base = stage + (b % 2u) * batch * packet_bytes;
            for (uint32_t i = 0; i < batch && b * batch + i < mine; ++i) {
                const uint32_t p = link + (b * batch + i) * links;
                for (uint32_t k = 0; k < 2u; ++k) {
                    noc_async_read(src_addr(h, 2u * p + k), base + i * packet_bytes + k * WHS_TILE, WHS_TILE);
                }
            }
        };
        if (batches > 0u) {
            read_batch(0u);
        }
        for (uint32_t b = 0; b < batches; ++b) {
            noc_async_read_barrier();
            noc_async_writes_flushed();  // batch b - 1's packets and local writes have left its slots
            if (b + 1u < batches) {
                read_batch(b + 1u);
            }
            const uint32_t base = stage + (b % 2u) * batch * packet_bytes;
            for (uint32_t i = 0; i < batch && b * batch + i < mine; ++i) {
                const uint32_t p = link + (b * batch + i) * links;
                const uint32_t src = base + i * packet_bytes;
                uint64_t dst[2];
                for (uint32_t k = 0; k < 2u; ++k) {
                    bool is_k;
                    const uint32_t page = dst_page(h, 2u * p + k, is_k);
                    dst[k] = is_k ? tt::tt_fabric::linear::addrgen_detail::get_noc_address(k_dst, page, 0)
                                  : tt::tt_fabric::linear::addrgen_detail::get_noc_address(vt_dst, page, 0);
                    if (local_copy) {
                        noc_async_write(
                            src + k * WHS_TILE, is_k ? k_dst.get_noc_addr(page) : vt_dst.get_noc_addr(page), WHS_TILE);
                    }
                }
                if (!no_fabric) {
                    fabric_multicast_noc_scatter_write(
                        &sender,
                        header((b % 2u) * batch + i),
                        src,
                        packet_bytes,
                        tt::tt_fabric::NocUnicastScatterCommandHeader(
                            {dst[0], dst[1]}, {static_cast<uint16_t>(WHS_TILE)}),
                        1u,
                        range);
                }
            }
        }
        // Head h is out: count it on every chip this sender wrote.
        noc_async_writes_flushed();
        if (!no_fabric) {
            fabric_multicast_noc_unicast_atomic_inc(
                &sender,
                inc_header,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_counter(h), 1u, true},
                1u,
                range);
            noc_async_writes_flushed();
        }
        if (local_copy) {
            noc_async_write_barrier();
            noc_semaphore_inc(counter_word(h), 1u);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    sender.close();
}
