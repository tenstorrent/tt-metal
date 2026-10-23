# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_all_gather — a line / ring all-gather over fabric, DRAM to DRAM, as one ttnn.generic_op.

Groups: `cluster_axis` 0 (each mesh column), 1 (each mesh row) or None (one snake over the whole mesh). Every hop
must be one physical link. Each chip's shard is TILE or ROW_MAJOR (any dtype), interleaved in DRAM; the output
concatenates the group's shards in row-major chip order along dim 0 (or along dim -2 when every leading dim is 1),
whatever order the data travels in (a snake ring over the whole mesh still writes chip (r, c) at slot r * cols + c).

Per chip, per direction (toward the next / previous chip in the group) and per link there is one *port* core:
  reader (RISCV_1, NoC0): reads its chip's own shard from the input, then the shards it relays, from the output,
                          each chunk only once the arrival counter says it has landed.
  sender (RISCV_0, NoC1): first tells the peer port that writes into this chip that it may (a *ready*), and waits for
                          its own peer's ready (the fence between calls); then sends every chunk into the same pages
                          of the neighbour's output, every 8th packet a fused write + increment of the neighbour
                          port's arrival counter. After its sends it waits until everything it expects from upstream
                          has arrived, then re-arms the counter.
  Line:  toward p+1 it sends shards p, p-1, ..., 0; toward p-1 it sends p, p+1, ..., G-1.
  Ring:  toward p+1 it sends G//2 shards (p, p-1, ...); toward p-1 the other G-1-G//2 (p, p+1, ...).
Chunks: runs of up to payload/page tiles that sit consecutively in one DRAM bank (interleaved pages i, i+B, i+2B, ...,
B = number of banks); their places in the output are consecutive in one bank too. Links split a shard by input bank.
Copy cores (one per link) write the chip's own shard into its own output, so the port cores only read and send.

Placement ("auto"): each port core goes in the NoC column of its connection's Ethernet core, directly below the
Ethernet row. The Ethernet cores are found once per mesh by a probe program that builds each connection on device
and reports the router's NoC coordinates. An mcast packet bumps one arrival-counter address on every chip it writes,
so with per-chip placement the mcast modes keep their counters on *sink* cores (one per ring, direction and link, the
same core on every chip) that wait for the arrivals instead of the port.
"""

import math

import ttnn

CB_CHUNKS = 0
NOC0 = ttnn.NOC.RISCV_0_default  # routes +X, then +Y
NOC1 = ttnn.NOC.RISCV_1_default  # routes -Y, then -X

# Shared by reader, sender and copy writer: the chunk walk of one shard for one link. Banks b = first, first+stride, ...
# are visited round-robin; a chunk is n <= run_pages pages b + (m + t) * B, consecutive in bank b.
_CHUNK_WALK = r"""
// Profiling (only under the device profiler): ZONE_K(name, k, body) runs body in zone name_e<k>, k < 3; WAIT_T(acc, x)
// adds the cycles x takes to acc (reported per entry as timestamped data, see the sender / reader).
#if defined(PROFILE_KERNEL)
#define ZONE_K(name, k, ...)                         \
    if ((k) == 0) {                                  \
        DeviceZoneScopedN(name "_e0");               \
        __VA_ARGS__;                                 \
    } else if ((k) == 1) {                           \
        DeviceZoneScopedN(name "_e1");               \
        __VA_ARGS__;                                 \
    } else {                                         \
        DeviceZoneScopedN(name "_e2");               \
        __VA_ARGS__;                                 \
    }
#define WAIT_T(acc, ...)                             \
    {                                                \
        const uint32_t _t0 = get_timestamp_32b();    \
        __VA_ARGS__;                                 \
        acc += get_timestamp_32b() - _t0;            \
    }
#else
#define ZONE_K(name, k, ...) __VA_ARGS__;
#define WAIT_T(acc, ...) __VA_ARGS__;
#endif
// The chunk walk of one shard for one port. The port owns banks first, first+stride, ... (its bank set, nb banks); it
// visits them round-robin starting at set index `rot`. A chunk is n <= run_pages pages b + (m + t) * B, consecutive in
// bank b. `part` 0 = every bank of the set, 1 = its first half, 2 = its second half (a shard split between the two
// directions of a ring). f(page, n, idx): idx = the chunk's index in the full walk (part 0) of this shard.
template <typename F>
FORCE_INLINE void for_each_chunk(
    uint32_t shard_pages, uint32_t first_bank, uint32_t bank_stride, uint32_t num_banks, uint32_t run_pages,
    uint32_t rot, uint32_t part, F&& f) {
    // num_banks is a compile-time constant at every call site, so the per-bank divisions fold to shifts; the bank-set
    // size is computed once and the rotation wraps by subtraction (no runtime division in the loop).
    const uint32_t max_per_bank = (shard_pages + num_banks - 1) / num_banks;
    uint32_t idx = 0;
    if (rot == 0 && part == 0) {  // the common case: every bank, in order
        for (uint32_t m = 0; m < max_per_bank; m += run_pages) {
            for (uint32_t b = first_bank; b < num_banks; b += bank_stride) {
                const uint32_t per_bank = b < shard_pages ? (shard_pages - b + num_banks - 1) / num_banks : 0;
                if (m >= per_bank) {
                    continue;
                }
                const uint32_t n = (per_bank - m) < run_pages ? (per_bank - m) : run_pages;
                f(b + m * num_banks, n, idx++);
            }
        }
        return;
    }
    const uint32_t nb = first_bank < num_banks ? (num_banks - first_bank + bank_stride - 1) / bank_stride : 0;
    const uint32_t half = nb / 2, lo = part == 2 ? half : 0, hi = part == 1 ? half : nb;
    for (uint32_t m = 0; m < max_per_bank; m += run_pages) {
        uint32_t slot = rot;
        for (uint32_t i = 0; i < nb; ++i, ++slot) {
            if (slot >= nb) {
                slot -= nb;
            }
            const uint32_t b = first_bank + slot * bank_stride;
            const uint32_t per_bank = b < shard_pages ? (shard_pages - b + num_banks - 1) / num_banks : 0;
            if (m >= per_bank) {
                continue;
            }
            if (slot >= lo && slot < hi) {
                const uint32_t n = (per_bank - m) < run_pages ? (per_bank - m) : run_pages;
                f(b + m * num_banks, n, idx);  // first page (shard-local), page count, index in the full walk
            }
            ++idx;
        }
    }
}
"""

_READER_SOURCE = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
"""
    + _CHUNK_WALK
    + r"""
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t group = get_compile_time_arg_val(4);  // chunks per barrier (half the CB)
    constexpr uint32_t inc_every = get_compile_time_arg_val(5);  // upstream increments my counter once per this many chunks
    constexpr auto in_args = TensorAccessorArgs<6>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t shard_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t rot = get_arg_val<uint32_t>(a++);           // bank-walk rotation (same as my upstream's)
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);  // my arrival counter (relays wait on it)
    const uint32_t full = get_arg_val<uint32_t>(a++);          // chunks of a whole shard for this port (host-counted)
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);         // chunks over all my entries (host-counted)
    const uint32_t num_shards = get_arg_val<uint32_t>(a++);    // entries follow: shard | part << 16 | own << 28
    const uint32_t shards_idx = a;
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    // A batch ends at the next multiple of `group` chunks in the CB (2 x group chunks, so a reservation never wraps)
    // or at the end of an entry: a relay waits for its upstream's shard, and an unpushed tail of my own shard would
    // keep my sender, and around a ring every sender, from sending it (a short shard would deadlock).
    uint32_t batch = 0, need = 0, pos = 0, wptr = 0;
    for (uint32_t k = 0; k < num_shards; ++k) {
        uint32_t arr_wait = 0, cb_wait = 0;  // profiling: cycles waiting on upstream arrivals / for CB space
        auto read_entry = [&]() {
        const uint32_t entry = get_arg_val<uint32_t>(shards_idx + k);
        const uint32_t out_base = (entry & 0xFFFF) * shard_pages;
        // relay entry k is (part of) upstream entry k - 1; every upstream entry before it is a whole shard. Upstream
        // increments once per inc_every chunks and at the end of every entry, so chunk up_base + idx is covered by
        // increment (up_base + idx) / inc_every + 1, plus one per earlier entry end that is not a multiple of inc_every
        const uint32_t up_base = k > 0 ? (k - 1) * full : 0;
        uint32_t ends = 0;
        for (uint32_t e = 1; e < k; ++e) {
            ends += (e * full) % inc_every != 0;
        }
        const bool own = (entry >> 28) & 1;  // my own shard (relay mode: entry 0; direct / mcast modes: every entry)
        for_each_chunk(shard_pages, first_bank, bank_stride, num_banks, run_pages, rot, (entry >> 16) & 0xF, [&](uint32_t page, uint32_t n, uint32_t idx) {
            if (batch == 0) {
                need = group - pos % group;
                WAIT_T(cb_wait, cb_reserve_back(cb, run_pages * need));
                wptr = get_write_ptr(cb);
            }
            uint64_t src;
            if (own) {
                src = in.get_noc_addr(page);  // my own shard, from the input
            } else {
                // upstream increments my counter once per inc_every chunks it sends; this is its chunk up_base + idx
                WAIT_T(arr_wait, noc_semaphore_wait_min(arrived, (up_base + idx) / inc_every + ends + 1));  // landed
                src = out.get_noc_addr(out_base + page);
            }
            noc_async_read(src, wptr + batch * chunk_bytes, n * page_bytes);
            --chunks_left;
            if (++batch == need || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(cb, run_pages * batch);
                pos += batch;
                batch = 0;
            }
        });
        if (batch > 0) {
            noc_async_read_barrier();
            cb_push_back(cb, run_pages * batch);
            pos += batch;
            batch = 0;
        }
        };
        ZONE_K("R", k, read_entry());
        DeviceTimestampedData("R_arrwait", (uint64_t(k) << 32) | arr_wait);
        DeviceTimestampedData("R_cbwait", (uint64_t(k) << 32) | cb_wait);
    }
}
"""
)

_SENDER_SOURCE = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

// The route of one packet, by hop count in this connection's direction: hops = count | direction << 8, mcast =
// range | direction << 8 (direction: eth_chan_directions, from the host; used on 2D only, ROUTING_MODE from the
// build). range > 0: write at each of the next `range` chips; otherwise at the chip `count` hops away. On 2D both run
// straight along the connection's direction, never by destination: a destination route picks its own path (a tie on
// an even ring, the short way round for count > G/2), which loses packets injected on the other direction's router.
// A ring that turns corners cannot be one multicast (host-refused).
inline void set_route(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint32_t hops, uint32_t mcast, uint16_t chip_id, uint16_t mesh_id) {
    const uint32_t mcast_range = mcast & 0xFF;
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    if (mcast_range > 0) {
        const uint32_t dir = mcast >> 8;
        fabric_set_mcast_route(
            hdr,
            chip_id,
            mesh_id,
            dir == static_cast<uint32_t>(eth_chan_directions::EAST) ? mcast_range : 0,
            dir == static_cast<uint32_t>(eth_chan_directions::WEST) ? mcast_range : 0,
            dir == static_cast<uint32_t>(eth_chan_directions::NORTH) ? mcast_range : 0,
            dir == static_cast<uint32_t>(eth_chan_directions::SOUTH) ? mcast_range : 0);
    } else {
        // as fabric_set_unicast_route for a straight path (test_fabric_set_unicast_route.cpp checks the two agree)
        hdr->dst_start_node_id = ((uint32_t)mesh_id << 16) | (uint32_t)chip_id;
        hdr->mcast_params_64 = 0;
        hdr->is_mcast_active = 0;
        hdr->routing_fields.value = 0;
        fabric_set_route(hdr, static_cast<eth_chan_directions>(hops >> 8), 0, 0, hops & 0xFF, true);
    }
#else
    (void)chip_id;
    (void)mesh_id;
    auto* h = reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr);
    if (mcast_range > 0) {
        h->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(mcast_range)});
    } else {
        (void)fabric_set_unicast_route<false>(h, static_cast<uint16_t>(hops & 0xFF));
    }
#endif
}
"""
    + _CHUNK_WALK
    + r"""
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t group = get_compile_time_arg_val(4);  // chunks per flush; the CB holds 2 x group chunks
    constexpr uint32_t inc_every = get_compile_time_arg_val(5);  // fused write+increment once per this many chunks
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t num_headers = group;

    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t shard_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t rot = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);  // my counter; the downstream port's is at the same address
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);     // increments I receive from upstream
    const uint32_t total = get_arg_val<uint32_t>(a++);         // chunks I send (host-counted)
    uint32_t entry_chunks[3];                                    // chunks of one entry: whole shard, first half, second half
    entry_chunks[0] = get_arg_val<uint32_t>(a++);
    entry_chunks[1] = get_arg_val<uint32_t>(a++);
    entry_chunks[2] = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);    // my ready counter; every port's is at the same address
    const uint32_t ready_expect = get_arg_val<uint32_t>(a++);  // readies I wait for: one per chip I write into
    const uint32_t n_ready = get_arg_val<uint32_t>(a++);       // readies I send: one per port that writes into me
    const uint32_t ready_idx = a;                              // per ready: hops, mesh id, chip id, port x, port y
    a += 5 * n_ready;
    const uint32_t num_shards = get_arg_val<uint32_t>(a++);    // 0 = nothing to send; entries: shard | part << 16 | own << 28
    const uint32_t shards_idx = a;
    a += num_shards;
    const uint32_t route_idx = a;                              // per entry: hops, mcast range, mesh id, chip id, port x, y
    a += 6 * num_shards;
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    if (num_shards > 0 || n_ready > 0) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
        }
        conn.open();
        {
        DeviceZoneScopedN("S_fence");
        // Fence between calls. This program running here means every earlier command on this chip is done, including
        // whatever read the output of the previous call, so every port that writes into my output may now: say so,
        // to each of them (`hops` back along this connection). Then send nothing until every chip I write into has
        // said the same (each call sends, and consumes, exactly one ready per sender / receiver pair).
        for (uint32_t i = 0; i < n_ready; ++i) {
            const uint32_t r = ready_idx + 5 * i;
            set_route(hdrs[0], get_arg_val<uint32_t>(r), 0, get_arg_val<uint32_t>(r + 2), get_arg_val<uint32_t>(r + 1));
            hdrs[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                get_noc_addr(get_arg_val<uint32_t>(r + 3), get_arg_val<uint32_t>(r + 4), ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (ready_expect > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, ready_expect);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - ready_expect);
            noc_async_atomic_barrier();
        }
        }
        uint32_t chunks_left = total;
        uint32_t sent = 0;
        // Up to `group` chunks in flight; the read pointer sits on a group boundary of a 2-group CB (no wrap). A header
        // is reused only after the flush that follows its send, so re-routing it per chunk is safe.
        uint32_t h = 0, unflushed = 0;
        for (uint32_t k = 0; k < num_shards; ++k) {
            uint32_t cb_wait = 0, slot_wait = 0;  // profiling: cycles waiting for the reader / for a fabric slot
            auto send_entry = [&]() {
            const uint32_t entry = get_arg_val<uint32_t>(shards_idx + k);
            const uint32_t out_base = (entry & 0xFFFF) * shard_pages;
            const uint32_t part = (entry >> 16) & 0xF;
            const uint32_t entry_total = entry_chunks[part];
            const uint32_t rt = route_idx + 6 * k;
            const uint32_t hops = get_arg_val<uint32_t>(rt), mcast_range = get_arg_val<uint32_t>(rt + 1);
            const uint16_t mesh_id = get_arg_val<uint32_t>(rt + 2), chip_id = get_arg_val<uint32_t>(rt + 3);
            // the destination port's arrival counter (an mcast packet bumps the same address on every chip it writes)
            const uint64_t arrival_noc = get_noc_addr(get_arg_val<uint32_t>(rt + 4), get_arg_val<uint32_t>(rt + 5), arrival_addr);
            uint32_t entry_sent = 0;
            for_each_chunk(shard_pages, first_bank, bank_stride, num_banks, run_pages, rot, part, [&](uint32_t page, uint32_t n, uint32_t) {
                WAIT_T(cb_wait, cb_wait_front(cb, run_pages * (unflushed + 1)));
                const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
                const uint32_t bytes = n * page_bytes;
                const uint64_t dst = out.get_noc_addr(out_base + page, 0, 0);  // NoC0 coordinates
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                set_route(hdr, hops, mcast_range, chip_id, mesh_id);
                // The increment is issued only after its write has landed (flush), which stalls the receiving
                // router, so only every inc_every-th chunk carries one, and the last of each entry: a relay of a
                // short shard must not wait for chunks its upstream sends only after relaying (a ring would deadlock).
                ++entry_sent;
                if ((++sent % inc_every) == 0 || entry_sent == entry_total) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                WAIT_T(slot_wait, conn.wait_for_empty_write_slot());
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                --chunks_left;
                if (++unflushed == group || chunks_left == 0) {
                    noc_async_writes_flushed();  // sources + headers of the in-flight chunks have left L1
                    cb_pop_front(cb, run_pages * unflushed);
                    unflushed = 0;
                }
            });
            };
            ZONE_K("S", k, send_entry());
            DeviceTimestampedData("S_cbwait", (uint64_t(k) << 32) | cb_wait);
            DeviceTimestampedData("S_slotwait", (uint64_t(k) << 32) | slot_wait);
        }
        conn.close();
    }

    // Everything from upstream has landed (the relays needed part of it already); then re-arm for the next call.
    if (expect_in > 0) {
        DeviceZoneScopedN("S_tail");
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
"""
)

_COPY_WRITER_SOURCE = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
"""
    + _CHUNK_WALK
    + r"""
// Copy core: writes the chunks its reader fetched (my own shard) into my own output.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t group = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t shard_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t out_base = get_arg_val<uint32_t>(a++) * shard_pages;  // my shard's place in the output
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);                   // host-counted
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    uint32_t pending = 0;
    for_each_chunk(shard_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t) {
        cb_wait_front(cb, run_pages * (pending + 1));
        noc_async_write(get_read_ptr(cb) + pending * chunk_bytes, out.get_noc_addr(out_base + page), n * page_bytes);
        --chunks_left;
        if (++pending == group || chunks_left == 0) {
            noc_async_write_barrier();
            cb_pop_front(cb, run_pages * pending);
            pending = 0;
        }
    });
}
"""
)

_SINK_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Sink core (mcast modes with per-chip port placement): holds the arrival counter of one receiving port, on the same
// core on every chip so that one mcast packet bumps it everywhere. Waits until everything from upstream has landed,
// then re-arms the counter for the next call (the sender's tail, moved here).
void kernel_main() {
    const uint32_t arrival_addr = get_arg_val<uint32_t>(0);
    const uint32_t expect_in = get_arg_val<uint32_t>(1);
    if (expect_in > 0) {
        DeviceZoneScopedN("K_tail");
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
"""

_PROBE_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

// Report where this connection's router (Ethernet core) sits: build the connection (without opening it) and write
// the router's NoC coordinates into my L1 slot.
void kernel_main() {
    size_t a = 0;
    const uint32_t slot = get_arg_val<uint32_t>(a++);
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
    p[0] = conn.edm_noc_x;
    p[1] = conn.edm_noc_y;
}
"""


# ----------------------------------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------------------------------


def build_groups(mesh_shape, cluster_axis):
    """Ordered lists of mesh coords, one per group."""
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    if cluster_axis == 1:
        return [[(r, c) for c in range(cols)] for r in range(rows)]
    snake = []
    for r in range(rows):
        snake += [(r, c) for c in (range(cols) if r % 2 == 0 else reversed(range(cols)))]
    return [snake]


MODES = ("relay", "direct", "mcast", "mcast_split", "mcast_halves")
_MCAST_MODES = ("mcast", "mcast_split", "mcast_halves")


def _schedule_routes(p, G, ring, balance, mode):
    """Ring send modes without relays (mode "direct" / "mcast"): every entry is the chip's own shard.

    Returns (fwd, bwd), each a list of (position, part, hops, mcast_range): the entry is written `hops` chips away in
    that direction (unicast; the fabric forwards it through the chips in between), or, with mcast_range > 0, at each of
    the next mcast_range chips by one multicast packet.
      direct: toward p+1 one unicast per distance 1..G//2, toward p-1 per distance 1..G-1-G//2. balance (even G): the
              shard for the opposite chip goes half each way (parts 1 / 2), so both directions carry G/2 - 1/2.
      mcast:  one multicast toward p+1 over G//2 chips and one toward p-1 over the rest. balance (even G): chips of
              even position swap the two ranges, so the long leg alternates direction around the ring and both
              directions carry the same total. Per link it alternates (G = 4: 2 and 1 shards): a multicast cannot
              split a shard the way the relay and direct modes do, so the busiest hop stays at G/2.
      mcast_split (even G; odd G is plain mcast, already even): the shard goes as two halves, each in both
              directions: toward p+1 half 1 over G/2 chips and half 2 over G/2 - 1, toward p-1 the reverse. The opposite
              chip gets one half from each side, every chip both halves, and every hop carries G/2 - 1/2 shards (the
              relay balance's load) with no relays.
      mcast_halves: half 1 multicast toward p+1 over all G - 1 other chips, half 2 toward p-1 over all of them; every
              hop carries (G - 1) / 2 shards, as mcast_split, but every packet goes the whole way round."""
    assert ring and mode in ("direct",) + _MCAST_MODES
    one = lambda k, part=0: [] if k == 0 else [(p, part, 1, 0)] if k == 1 else [(p, part, 1, k)]  # noqa: E731
    if mode == "mcast_halves":
        return one(G - 1, 1), one(G - 1, 2)
    if mode == "mcast_split":
        if G % 2 == 0 and G >= 4:
            h = G // 2
            return one(h, 1) + one(h - 1, 2), one(h, 2) + one(h - 1, 1)
        mode = "mcast"
    kf, kb = G // 2, G - 1 - G // 2
    if mode == "direct":
        if balance and G % 2 == 0 and G >= 4:
            fwd = [(p, 1 if h == kf else 0, h, 0) for h in range(1, kf + 1)]
            bwd = [(p, 2 if h == kf else 0, h, 0) for h in range(1, kf + 1)]
            return fwd, bwd
        return [(p, 0, h, 0) for h in range(1, kf + 1)], [(p, 0, h, 0) for h in range(1, kb + 1)]
    if balance and G % 2 == 0 and p % 2 == 0:
        kf, kb = kb, kf
    return one(kf), one(kb)


def _schedule(p, G, ring, balance=False):
    """(shards sent toward p+1, shards sent toward p-1), own shard first, in the order they arrive downstream.

    Entries are (position, part): part 0 = the whole shard, 1 / 2 = the first / second half of the port's banks.
    balance (ring, even G >= 4): the shard opposite each receiver goes half forward (part 1), half backward (part 2),
    so both directions carry G/2 - 1/2 shards instead of G/2 and G/2 - 1."""
    if ring and balance and G % 2 == 0 and G >= 4:
        h = G // 2
        fwd = [((p - i) % G, 1 if i == h - 1 else 0) for i in range(h)]
        bwd = [((p + i) % G, 2 if i == h - 1 else 0) for i in range(h)]
        return fwd, bwd
    if ring:
        kf = G // 2
        kb = G - 1 - kf
        return [((p - i) % G, 0) for i in range(kf)], [((p + i) % G, 0) for i in range(kb)] if kb > 0 else []
    fwd = [(p - i, 0) for i in range(p + 1)] if p < G - 1 else []
    bwd = [(p + i, 0) for i in range(G - p)] if p > 0 else []
    return fwd, bwd


_OPP = {"fwd": "bwd", "bwd": "fwd"}


def _needs_ready(chips, coord, j, d):
    """Whether port (ring j, direction d) of `coord` sends a ready to its peer: the peer's opposite-direction port
    sends to `coord` (its peer is `coord`), so it waits for that ready before writing into `coord`'s output."""
    rg = chips[coord]["rings"][j]
    peer = rg["next"] if d == "fwd" else rg["prev"]
    return peer is not None and bool(chips[peer]["rings"][j][_OPP[d]])


def _num_banks(mesh_device):
    g = mesh_device.dram_grid_size()
    return g.x * g.y


def _chunks_per_shard(shard_pages, first_bank, stride, num_banks, run_pages, part=0):
    """Chunks the kernel walk visits for one shard (replica of for_each_chunk; rotation does not change the count)."""
    banks = list(range(first_bank, num_banks, stride))
    nb = len(banks)
    total = 0
    for slot, b in enumerate(banks):
        if part == 1 and slot >= nb // 2 or part == 2 and slot < nb // 2:
            continue
        per_bank = (shard_pages - b + num_banks - 1) // num_banks if b < shard_pages else 0
        total += (per_bank + run_pages - 1) // run_pages
    return total


_PROBE_CACHE = {}


def probe_ethernet_cores(mesh_device, connections, allowed):
    """connections: {coord: [(peer_coord, link_index), ...]} -> {(coord, peer_coord, link_index): (noc_x, noc_y)}.

    One probe program per chip; each connection gets its own probe core: the first cores, left to right, of the
    lowest row of `allowed` (logical (x, y) set) that has enough of them."""
    key = (id(mesh_device), ttnn.get_fabric_config(), tuple(sorted((k, tuple(v)) for k, v in connections.items())))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key][1]
    rows, cols = tuple(mesh_device.shape)
    k_max = max(len(v) for v in connections.values())
    row = next(
        (y for y in sorted({y for _, y in allowed}, reverse=True) if sum(1 for _, yy in allowed if yy == y) >= k_max),
        None,
    )
    if row is None:
        raise ValueError(f"fabric_all_gather: no row of the core grid has {k_max} cores for the placement probe")
    cores = [ttnn.CoreCoord(x, row) for x in sorted(x for x, y in allowed if y == row)[:k_max]]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_set, (1, 8), ttnn.ShardOrientation.ROW_MAJOR),
    )
    import torch

    slots = ttnn.from_torch(
        torch.zeros((k_max, 8), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    slot_addr = int(slots.buffer_address())
    mesh_desc = ttnn.MeshProgramDescriptor()
    for (r, c), conns in connections.items():
        me = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        program = ttnn.ProgramDescriptor()
        rt = ttnn.RuntimeArgs()
        used = []
        for i, (peer, link) in enumerate(conns):
            peer_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*peer))
            rt[cores[i].x][cores[i].y] = [slot_addr] + list(
                ttnn.setup_fabric_connection(me, peer_node, link, program, cores[i])
            )
            used.append(cores[i])
        if used:
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_PROBE_SOURCE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(u, u) for u in used]),
                    compile_time_args=[],
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0),
                )
            ]
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    ttnn.generic_op([slots, slots], mesh_desc)  # (input, output): this tree's generic_op wants both
    ttnn.synchronize_device(mesh_device)
    per_dev = [ttnn.to_torch(t).reshape(k_max, 8) for t in ttnn.get_device_tensors(slots)]
    result = {}
    for (r, c), conns in connections.items():
        got = per_dev[r * cols + c]
        for i, (peer, link) in enumerate(conns):
            result[((r, c), peer, link)] = (int(got[i, 0]), int(got[i, 1]))
    _PROBE_CACHE[key] = (mesh_device, result)
    return result


_ETH_MAP_CACHE = {}
# Blackhole, with NoC translation on:
#   Ethernet: core i (i-th entry of the chip's non-harvested Ethernet list) has translated coordinate (20 + i, 25).
#   Tensix:   columns are compacted - the k-th live column gets the k-th column's x; harvested columns move to the end.
#             So a live column right of a harvested one has translated x != physical x.
_BH_ETH_TRANSLATED_X0 = 20


def _eth_physical_lists(mesh_device):
    """{asic unique id: [physical (x, y) of Ethernet core i]} from the cluster descriptor + the arch SoC descriptor."""
    if id(mesh_device) in _ETH_MAP_CACHE:
        return _ETH_MAP_CACHE[id(mesh_device)][1]
    import os
    import yaml

    desc = yaml.safe_load(open(ttnn.cluster.serialize_cluster_descriptor()))
    root = os.environ.get("TT_METAL_HOME", os.getcwd())
    arch_yaml = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
        str(mesh_device.arch()).split(".")[-1]
    ]
    arch_eth = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))["eth"]
    arch_eth = [tuple(int(v) for v in e.split("-")) for e in arch_eth]
    soc = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))
    tensix_cols = sorted({int(w.split("-")[0]) for w in soc["functional_workers"]})
    lists = {}
    for chip, uid in desc.get("chip_unique_ids", {}).items():
        h = desc.get("harvesting", {}).get(chip, {})
        mask = int(h.get("eth_harvesting_mask", 0))
        translated = bool(h.get("noc_translation", False))
        tmask = int(h.get("harvest_mask", 0))
        live = [x for i, x in enumerate(tensix_cols) if not (tmask >> i) & 1]
        t2p = {tensix_cols[k]: px for k, px in enumerate(live)} if translated else {}
        lists[int(uid)] = ([e for i, e in enumerate(arch_eth) if not (mask >> i) & 1], translated, t2p)
    _ETH_MAP_CACHE[id(mesh_device)] = (mesh_device, lists)
    return lists


def eth_noc_coord(mesh_device, coord, xy):
    """Physical NoC (x, y) of the Ethernet core the probe reported (translated or physical coordinates)."""
    x, y = xy
    eth_list, translated, _ = _chip_maps(mesh_device, coord)
    if translated and eth_list is not None and x >= _BH_ETH_TRANSLATED_X0:
        return eth_list[x - _BH_ETH_TRANSLATED_X0]
    return x, y


def eth_noc_column(mesh_device, coord, xy):
    """Physical NoC column of the Ethernet core the probe reported (translated or physical coordinates)."""
    return eth_noc_coord(mesh_device, coord, xy)[0]


_NOC_GRID_CACHE = {}


def _noc_grid(mesh_device):
    """(x_size, y_size) of the chip's NoC grid, from the arch SoC descriptor."""
    if id(mesh_device) not in _NOC_GRID_CACHE:
        import os
        import yaml

        root = os.environ.get("TT_METAL_HOME", os.getcwd())
        arch_yaml = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
            str(mesh_device.arch()).split(".")[-1]
        ]
        grid = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))["grid"]
        _NOC_GRID_CACHE[id(mesh_device)] = (mesh_device, (int(grid["x_size"]), int(grid["y_size"])))
    return _NOC_GRID_CACHE[id(mesh_device)][1]


def _chip_maps(mesh_device, coord):
    node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    uid = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(node.mesh_id), int(node.chip_id)))
    return _eth_physical_lists(mesh_device).get(uid, (None, False, {}))


def _worker_nearest_noc1(mesh_device, coord, eth_xy, taken, allowed):
    """Logical worker core (from `allowed`, not yet taken) with the fewest NoC1 hops to the Ethernet core at physical
    eth_xy; ties to the lower row, then the lower column. NoC1 (the sender's NoC) routes -Y, then -X, wrapping around
    the grid, so the hop count is directional: a core one column left of a harvested column would wrap the whole row.
    Two links whose Ethernet cores sit over one live and one harvested column then get workers in two columns (as
    fabric_all_gather's C++ op places them), instead of both in the live column, where their writes to the routers
    would share the column's NoC1 segment into the Ethernet row."""
    t2p = _chip_maps(mesh_device, coord)[2]
    gx, gy = _noc_grid(mesh_device)
    ex, ey = eth_xy
    best = None
    for x, y in sorted(allowed):
        lc = ttnn.CoreCoord(x, y)
        if (x, y) in taken:
            continue
        v = mesh_device.worker_core_from_logical_core(lc)
        px = t2p.get(v.x, v.x)
        score = ((v.y - ey) % gy + (px - ex) % gx, v.y, px)
        if best is None or score < best[0]:
            best = (score, lc)
    return best[1] if best is not None else None


def _worker_below(mesh_device, coord, noc_x, taken, allowed):
    """Logical worker core (from `allowed`) in physical NoC column noc_x closest below the Ethernet row, not yet
    taken; else the nearest physical column. Worker coordinates are translated, so they are mapped back per chip."""
    t2p = _chip_maps(mesh_device, coord)[2]
    best = None
    for x, y in sorted(allowed):
        lc = ttnn.CoreCoord(x, y)
        if (x, y) in taken:
            continue
        v = mesh_device.worker_core_from_logical_core(lc)
        score = (abs(t2p.get(v.x, v.x) - noc_x), v.y)
        if best is None or score < best[0]:
            best = (score, lc)
    return best[1] if best is not None else None


# Two edge-disjoint Hamiltonian cycles per torus shape, from hamiltonian_decomposition(R, C) (seed=1): 4 x 4 (a
# Galaxy sub-torus) and 4 x 8 (a full Galaxy). Other shapes are searched in plan().
# fmt: off
_HAMILTONIAN_CYCLES = {
    (4, 4): (
        [
            (0, 0), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3), (1, 0),
            (2, 0), (2, 3), (3, 3), (3, 2), (2, 2), (2, 1), (3, 1), (3, 0),
        ],
        [
            (0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (3, 3), (0, 3),
            (1, 3), (2, 3), (2, 2), (1, 2), (0, 2), (3, 2), (3, 1), (0, 1),
        ],
    ),
    (4, 8): (
        [
            (0, 0), (3, 0), (2, 0), (1, 0), (1, 1), (0, 1), (3, 1), (2, 1),
            (2, 2), (1, 2), (0, 2), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3),
            (0, 4), (3, 4), (2, 4), (1, 4), (1, 5), (0, 5), (3, 5), (2, 5),
            (2, 6), (2, 7), (3, 7), (3, 6), (0, 6), (1, 6), (1, 7), (0, 7),
        ],
        [
            (0, 0), (1, 0), (1, 7), (2, 7), (2, 0), (2, 1), (1, 1), (1, 2),
            (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (0, 7), (3, 7), (3, 0),
            (3, 1), (3, 2), (2, 2), (2, 3), (2, 4), (2, 5), (1, 5), (1, 6),
            (2, 6), (3, 6), (3, 5), (3, 4), (3, 3), (0, 3), (0, 2), (0, 1),
        ],
    ),
}
# fmt: on


def hamiltonian_decomposition(R, C, seed=1, tries=20000):
    """Two edge-disjoint Hamiltonian cycles covering every edge of an R x C torus (R, C >= 3), as coord lists.

    Randomized Warnsdorff search for cycle A, accepted when the complementary edges form one cycle B."""
    import random
    import sys

    sys.setrecursionlimit(max(10000, 4 * R * C))
    rng = random.Random(seed)
    nodes = [(r, c) for r in range(R) for c in range(C)]
    nb = {
        n: [((n[0] + 1) % R, n[1]), ((n[0] - 1) % R, n[1]), (n[0], (n[1] + 1) % C), (n[0], (n[1] - 1) % C)]
        for n in nodes
    }
    N = len(nodes)
    for _ in range(tries):
        path, used = [nodes[0]], {nodes[0]}

        def dfs():
            if len(path) == N:
                return path[0] in nb[path[-1]]
            cand = [n for n in nb[path[-1]] if n not in used]
            rng.shuffle(cand)
            cand.sort(key=lambda n: sum(1 for m in nb[n] if m not in used))
            for n in cand:
                path.append(n)
                used.add(n)
                if dfs():
                    return True
                path.pop()
                used.discard(n)
            return False

        if not dfs():
            continue
        a_edges = {frozenset((path[i], path[(i + 1) % N])) for i in range(N)}
        comp = {n: [m for m in nb[n] if frozenset((n, m)) not in a_edges] for n in nodes}
        if any(len(v) != 2 for v in comp.values()):
            continue
        cyc_b, prev, cur = [nodes[0]], None, nodes[0]
        while True:
            x, y = comp[cur]
            nxt = x if x != prev else y
            if nxt == nodes[0]:
                break
            cyc_b.append(nxt)
            prev, cur = cur, nxt
        if len(cyc_b) == N:
            return path, cyc_b
    raise ValueError(f"no Hamiltonian decomposition found for a {R}x{C} torus")


def allowed_cores(mesh_device, sub_core_grid=None):
    """Logical (x, y) worker cores the op may use: the whole compute grid, or the cores of `sub_core_grid`."""
    if sub_core_grid is None:
        grid = mesh_device.compute_with_storage_grid_size()
        return {(x, y) for x in range(grid.x) for y in range(grid.y)}
    return {
        (x, y)
        for r in sub_core_grid.ranges()
        for x in range(r.start.x, r.end.x + 1)
        for y in range(r.start.y, r.end.y + 1)
    }


def plan(
    mesh_device,
    *,
    cluster_axis,
    topology,
    num_links,
    placement="auto",
    scheme="ring",
    balance=False,
    sub_core_grid=None,
    mode="relay",
):
    """Rings, per-port shard schedules and core placement for every chip.

    scheme "ring":        one ring (or line) per group, in group order.
    scheme "dual_cycles": one group over the whole mesh (a 2D torus, both sides >= 3) and two rings over it — two
                          edge-disjoint Hamiltonian cycles — each carrying half of every shard (half the DRAM banks).
    mode "relay":  every send goes one hop; intermediate chips relay the shards of chips further away.
    mode "direct" / "mcast" / "mcast_split" / "mcast_halves" (ring only): every chip sends only its own shard, routed
                   by the fabric through the chips in between (see _schedule_routes); nothing is relayed.
    placement "uniform": the "auto" placement of the first chip, used on every chip (an mcast packet carries one
                   arrival-counter address for every chip it writes, so the receiving ports sit on the same core).
                   With any other placement the mcast modes add sink cores (ch["sinks"]) that hold those counters.
    The output concatenates the group's shards in row-major chip order, independent of the ring order."""
    import os

    if mode not in MODES:
        raise ValueError(f"fabric_all_gather: unknown mode {mode!r}")
    if placement == "uniform":
        first, _ = plan(
            mesh_device,
            cluster_axis=cluster_axis,
            topology=topology,
            num_links=num_links,
            placement="auto",
            scheme=scheme,
            balance=balance,
            sub_core_grid=sub_core_grid,
            mode=mode,
        )
        ports0 = next(iter(first.values()))["ports"]
        return plan(
            mesh_device,
            cluster_axis=cluster_axis,
            topology=topology,
            num_links=num_links,
            placement=dict(ports0),
            scheme=scheme,
            balance=balance,
            sub_core_grid=sub_core_grid,
            mode=mode,
        )
    mesh_shape = tuple(mesh_device.shape)
    R, C = mesh_shape
    if scheme == "dual_cycles":
        if cluster_axis is not None or topology != ttnn.Topology.Ring or R < 3 or C < 3:
            raise ValueError(
                "fabric_all_gather: dual_cycles needs cluster_axis=None, Ring, and a torus with both sides >= 3"
            )
        groups = [[(r, c) for r in range(R) for c in range(C)]]
        a, b = _HAMILTONIAN_CYCLES.get((R, C)) or hamiltonian_decomposition(R, C)
        ring_lists = [[a, b]]
        ring = True
    elif scheme == "ring":
        groups = build_groups(mesh_shape, cluster_axis)
        ring_lists = [[grp] for grp in groups]
        ring = topology == ttnn.Topology.Ring and len(groups[0]) > 2  # a 2-chip ring is a line
    else:
        raise ValueError(f"fabric_all_gather: unknown scheme {scheme!r}")
    G = len(groups[0])
    assert G >= 2, "a group needs at least two chips"
    if mode != "relay" and not ring:
        raise ValueError(f"fabric_all_gather: mode {mode!r} needs a ring")
    if mode != "relay" and "2D" in str(ttnn.get_fabric_config()) and (scheme != "ring" or cluster_axis is None):
        # on 2D every route runs straight along its connection's physical direction (set_route); a send that skips
        # chips on a ring that turns corners (snake, dual cycles) would leave the ring
        raise ValueError(f"fabric_all_gather: mode {mode!r} on a 2D fabric needs a straight ring (cluster_axis 0 / 1)")
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    chips = {}
    for grp, rings in zip(groups, ring_lists):
        out_idx = {coord: i for i, coord in enumerate(sorted(grp))}  # output slot = row-major rank in the group
        for coord in grp:
            chips[coord] = dict(out=out_idx[coord], G=G, rings=[])
        for cyc in rings:
            for p, coord in enumerate(cyc):
                nxt = cyc[(p + 1) % G] if (ring or p < G - 1) else None
                prv = cyc[(p - 1) % G] if (ring or p > 0) else None
                rg = dict(p=p, next=nxt, prev=prv)
                if mode == "relay":
                    fwd, bwd = _schedule(p, G, ring, balance)  # (position along this ring, part), own first
                    routes = {d: [(q, part, 1, 0) for q, part in e] for d, e in (("fwd", fwd), ("bwd", bwd))}
                else:
                    fwd, bwd = _schedule_routes(p, G, ring, balance, mode)
                    routes = {"fwd": fwd, "bwd": bwd}
                for d, sign in (("fwd", 1), ("bwd", -1)):
                    rg[d] = [(out_idx[cyc[q]], part) for q, part, _, _ in routes[d]]
                    # per entry: (hops, mcast range, own shard?) and the chips it writes
                    rg[f"{d}_route"] = [(h, m, mode != "relay" or i == 0) for i, (_, _, h, m) in enumerate(routes[d])]
                    rg[f"{d}_dsts"] = [
                        [cyc[(p + sign * k) % G] for k in (range(1, m + 1) if m else [h])] for _, _, h, m in routes[d]
                    ]
                chips[coord]["rings"].append(rg)
    n_rings = len(ring_lists[0])
    # The fence between calls, generalized to sends that skip chips: a sending port waits for one *ready* from every
    # chip it writes into (ready_expect); a chip sends each of those readies from its port that faces the sender
    # (direction d' = opposite of the sender's), `k` hops back to it (ready_to). Relay mode: one ready, one hop, the
    # same as _needs_ready.
    pos = {(coord, j): rg["p"] for coord, ch in chips.items() for j, rg in enumerate(ch["rings"])}
    for coord, ch in chips.items():
        for rg in ch["rings"]:
            for d in ("fwd", "bwd"):
                rg[f"{d}_ready_to"] = []
                rg[f"{d}_ready_expect"] = len({t for ds in rg[f"{d}_dsts"] for t in ds}) if rg[d] else 0
    for s, ch in chips.items():
        for j, rg in enumerate(ch["rings"]):
            for d in ("fwd", "bwd"):
                if not rg[d]:
                    continue
                sign = 1 if d == "fwd" else -1
                for t in sorted({t for ds in rg[f"{d}_dsts"] for t in ds}):
                    # t sits h hops from s in direction d, so s is h hops from t in direction opp(d)
                    h = (sign * (pos[(t, j)] - pos[(s, j)])) % G
                    chips[t]["rings"][j][f"{_OPP[d]}_ready_to"].append((h, s, d))
    # links for every neighbour this chip sends to; every hop must be direct
    connections = {}
    for coord, ch in chips.items():
        conns = []
        for j, rg in enumerate(ch["rings"]):
            for d, peer in (("fwd", rg["next"]), ("bwd", rg["prev"])):
                if peer is None or not (rg[d] or rg[f"{d}_ready_to"]):
                    continue
                links = ttnn.get_forwarding_link_indices(node(coord), node(peer))
                if len(links) < num_links:
                    raise ValueError(
                        f"fabric_all_gather: {coord} -> {peer} has {len(links)} usable link(s), {num_links} requested "
                        f"(topology={topology}, fabric={ttnn.get_fabric_config()})"
                    )
                rg[f"{d}_links"] = links[:num_links]
                conns += [(peer, links[l]) for l in range(num_links)]
        connections[coord] = conns
    allowed = allowed_cores(mesh_device, sub_core_grid)
    if placement == "auto" and os.environ.get("TT_METAL_EMULE_MODE"):
        placement = "simple"  # the emulator runs no Ethernet cores (nothing to probe) and has no NoC timing
    eth = probe_ethernet_cores(mesh_device, connections, allowed) if placement == "auto" else None
    # Port cores: one per (ring, direction, link) on every chip, including receive-only ends of a line.
    for coord, ch in chips.items():
        taken = set()
        ports = {}
        for j, rg in enumerate(ch["rings"]):
            for d, peer in (("fwd", rg["next"]), ("bwd", rg["prev"])):
                for l in range(num_links):
                    core = None
                    if eth is not None and rg.get(f"{d}_links"):
                        exy = eth_noc_coord(mesh_device, coord, eth[(coord, peer, rg[f"{d}_links"][l])])
                        core = _worker_nearest_noc1(mesh_device, coord, exy, taken, allowed)
                        ch.setdefault("eth_cols", {})[(j, d, l)] = exy[0]
                    elif isinstance(placement, dict):
                        core = placement.get((j, d, l))
                    ports[(j, d, l)] = core
                    if core is not None:
                        taken.add((core.x, core.y))
        for key, core in ports.items():  # receive-only ports, or simple placement: any free core
            if core is None:
                ports[key] = _worker_below(mesh_device, coord, 0, taken, allowed)
                if ports[key] is not None:
                    taken.add((ports[key].x, ports[key].y))
        copy = []
        ys = sorted({y for _, y in allowed})
        mid = ys[len(ys) // 2]
        for l in range(num_links):  # one per link, from the middle row down (away from the Ethernet row), then up
            for y in [y for y in ys if y >= mid] + [y for y in ys if y < mid]:
                free = sorted(x for x, yy in allowed if yy == y and (x, y) not in taken)
                if free:
                    copy.append(ttnn.CoreCoord(free[0], y))
                    taken.add((free[0], y))
                    break
        if len(copy) < num_links or any(c is None for c in ports.values()):
            raise ValueError(f"fabric_all_gather: the core grid has too few cores for {len(ports)} ports + copy cores")
        ch["ports"], ch["copy"] = ports, copy
    if mode in _MCAST_MODES and not isinstance(placement, dict):
        # Sink cores: one per port key, the same core on every chip, free on every chip (any spot will do: they only
        # take atomic increments). Middle row down, then up, as the copy cores.
        used = {(c.x, c.y) for ch in chips.values() for c in list(ch["ports"].values()) + ch["copy"]}
        ys = sorted({y for _, y in allowed})
        mid = ys[len(ys) // 2]
        free = [
            (x, y)
            for y in [y for y in ys if y >= mid] + [y for y in ys if y < mid]
            for x in sorted(x for x, yy in allowed if yy == y)
            if (x, y) not in used
        ]
        keys = list(next(iter(chips.values()))["ports"])
        if len(free) < len(keys):
            raise ValueError(f"fabric_all_gather: the core grid has too few cores for {len(keys)} sink cores")
        for ch in chips.values():
            ch["sinks"] = {k: ttnn.CoreCoord(*xy) for k, xy in zip(keys, free)}
    for ch in chips.values():
        ch["n_rings"] = n_rings
    return chips, ring


def link_load(chips):
    """Shards carried by each directed chip-to-chip hop (per link used), from a plan.

    A ring port sends len(sends) shards' worth of its (1 / n_rings) share of the banks. The busiest hop bounds the
    call: time >= busiest * shard_bytes / (num_links * link_rate)."""
    load = {}
    for coord, ch in chips.items():
        for rg in ch["rings"]:
            for d, peer in (("fwd", rg["next"]), ("bwd", rg["prev"])):
                if peer is None or not rg[d]:
                    continue
                if f"{d}_dsts" not in rg:  # a plan without routes (every entry one hop)
                    shards = sum(1.0 if part == 0 else 0.5 for _, part in rg[d])
                    load[(coord, peer)] = load.get((coord, peer), 0.0) + shards / ch["n_rings"]
                    continue
                # an entry loads every hop on its way: from this chip to its farthest destination
                for (_, part), ds in zip(rg[d], rg[f"{d}_dsts"]):
                    w = (1.0 if part == 0 else 0.5) / ch["n_rings"]
                    a = coord
                    for t in _path(chips, coord, ds[-1], rg, d):
                        load[(a, t)] = load.get((a, t), 0.0) + w
                        a = t
    return load


def _path(chips, src, dst, rg, d):
    """Chips a packet from `src` toward direction d passes through, up to and including `dst` (one ring)."""
    out, cur, j = [], src, next(i for i, r in enumerate(chips[src]["rings"]) if r is rg)
    while cur != dst:
        r = chips[cur]["rings"][j]
        cur = r["next"] if d == "fwd" else r["prev"]
        out.append(cur)
    return out


def _incs(entry_chunks, inc_every):
    """Increments a sender issues for entries of these chunk counts: every inc_every-th chunk and each entry's last."""
    return sum(_incs_per_entry(entry_chunks, inc_every))


def _incs_per_entry(entry_chunks, inc_every):
    """The same increments, per entry (each lands at that entry's destinations); the count runs across entries."""
    sent, per = 0, []
    for c in entry_chunks:
        n = 0
        for i in range(1, c + 1):
            sent += 1
            n += sent % inc_every == 0 or i == c
        per.append(n)
    return per


def create_mesh_program_descriptor(
    mesh_device,
    input_tensor,
    output_tensor,
    arrival_addr,
    chips,
    *,
    num_links,
    ready_addr=0,
    cb_bytes=112 * 1024,
    inc_every=8,
    desync=False,
):
    page_bytes = int(input_tensor.buffer_aligned_page_size())
    num_banks = _num_banks(mesh_device)
    shape = list(input_tensor.padded_shape)
    # a page is a 32 x 32 tile (TILE) or one row (ROW_MAJOR); the kernels only move whole pages, so any dtype works
    shard_pages = _shard_pages(input_tensor)
    run_pages = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)
    chunk_bytes = run_pages * page_bytes
    group = max(1, min(8, cb_bytes // (2 * chunk_bytes)))
    in_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    in_addr, out_addr = int(input_tensor.buffer_address()), int(output_tensor.buffer_address())
    ct = [CB_CHUNKS, page_bytes, run_pages, num_banks, group, inc_every]
    # The send mode and balance go last in the compile-time args (after the tensor accessors, so no index moves): the
    # program cache keys on compile-time args, not on runtime-arg counts, and the modes' runtime args differ in length.
    ch0 = next(iter(chips.values()))
    r0 = ch0["rings"][0]
    variant_ct = [
        len(r0["fwd"])
        | len(r0["bwd"]) << 4
        | len(r0["fwd_ready_to"]) << 8
        | len(r0["bwd_ready_to"]) << 12
        | (max((m for _, m, _ in r0["fwd_route"] + r0["bwd_route"]), default=0) << 16)
        | (max((h for h, _, _ in r0["fwd_route"] + r0["bwd_route"]), default=0) << 20)
        | int("sinks" in ch0) << 24
    ]
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )

    two_d = "2D" in str(ttnn.get_fabric_config())
    # increments every receiving port (chip, ring, direction, link) gets: from every sender of the same key whose
    # entries write it (relay: just the upstream neighbour)
    expect = {}
    for s, sch in chips.items():
        stride_s = sch["n_rings"] * num_links
        for j, rg in enumerate(sch["rings"]):
            for d in ("fwd", "bwd"):
                if not rg[d]:
                    continue
                for l in range(num_links):
                    first = j * num_links + l
                    counts = [
                        _chunks_per_shard(shard_pages, first, stride_s, num_banks, run_pages, part) for _, part in rg[d]
                    ]
                    for n, ds in zip(_incs_per_entry(counts, inc_every), rg[f"{d}_dsts"]):
                        for t in ds:
                            expect[(t, j, d, l)] = expect.get((t, j, d, l), 0) + n

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        n_rings = ch["n_rings"]
        stride = n_rings * num_links  # (ring j, link l) owns DRAM banks j*L + l, then every stride-th bank
        assert stride <= num_banks, f"{n_rings} rings x {num_links} links need at least {stride} DRAM banks"
        program = ttnn.ProgramDescriptor()
        port_cores = list(ch["ports"].values())
        all_cores = port_cores + ch["copy"]
        program.cbs = [
            ttnn.CBDescriptor(
                total_size=2 * group * chunk_bytes,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_cores]),
                format_descriptors=[ttnn.CBFormatDescriptor(CB_CHUNKS, input_tensor.dtype, page_bytes)],
            )
        ]
        reader_rt, sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        sinks = ch.get("sinks")
        for (j, d, l), core in ch["ports"].items():
            rg = ch["rings"][j]
            peer = rg["next"] if d == "fwd" else rg["prev"]
            sends = rg[d] if peer is not None else []
            # what arrives at this port: the upstream chip's port of the same ring and direction sends to me
            up = rg["prev"] if d == "fwd" else rg["next"]
            up_sends = chips[up]["rings"][j][d] if up is not None else []
            first = j * num_links + l
            nb = len(range(first, num_banks, stride))
            # desync: the two directions of a link walk the same banks; start the backward walk halfway round
            rot = nb // 2 if (desync and d == "bwd") else 0
            routes = rg[f"{d}_route"] if peer is not None else []
            dsts = rg[f"{d}_dsts"] if peer is not None else []
            packed = [out | (part << 16) | (int(own) << 28) for (out, part), (_, _, own) in zip(sends, routes)]
            full = _chunks_per_shard(shard_pages, first, stride, num_banks, run_pages)
            mine = sum(_chunks_per_shard(shard_pages, first, stride, num_banks, run_pages, part) for _, part in sends)
            per_part = [_chunks_per_shard(shard_pages, first, stride, num_banks, run_pages, part) for part in (0, 1, 2)]
            head = [in_addr, out_addr, shard_pages, first, stride, rot, arrival_addr, full, mine]
            reader_rt[core.x][core.y] = head + [len(sends)] + packed
            expect_in = 0 if sinks else expect.get((coord, j, d, l), 0)  # with sinks, they wait for the arrivals
            args = [out_addr, shard_pages, first, stride, rot, arrival_addr, expect_in, mine]
            args += per_part + [ready_addr]
            # readies: wait for one from every chip I write into; send one to every port that writes into me
            ready_to = rg[f"{d}_ready_to"] if peer is not None else []
            args += [rg[f"{d}_ready_expect"] if sends else 0, len(ready_to)]
            # 2D: every route runs along this connection's physical direction (plan() keeps multicast rings straight)
            mdir = ttnn.get_eth_forwarding_direction(node(coord), node(peer)) if two_d and peer is not None else 0
            for h, s, sd in ready_to:
                sc, sn = virt(chips[s]["ports"][(j, sd, l)]), node(s)
                args += [h | mdir << 8, int(sn.mesh_id), int(sn.chip_id), sc.x, sc.y]
            args += [len(sends)] + packed
            for (h, m, _), ds in zip(routes, dsts):
                counter = lambda t: virt(chips[t]["sinks" if sinks else "ports"][(j, d, l)])  # noqa: E731
                cores = {(counter(t).x, counter(t).y) for t in ds}
                if len(cores) > 1:
                    raise ValueError(
                        "fabric_all_gather: an mcast entry needs its receiving ports on the same core "
                        "on every chip (placement='uniform')"
                    )
                (px, py), tn = next(iter(cores)), node(ds[0] if m else ds[-1])
                args += [h | mdir << 8, m | (mdir << 8 if m else 0), int(tn.mesh_id), int(tn.chip_id), px, py]
            if rg.get(f"{d}_links") and (sends or ready_to):
                args += list(ttnn.setup_fabric_connection(node(coord), node(peer), rg[f"{d}_links"][l], program, core))
            sender_rt[core.x][core.y] = args
        copy_reader_rt, copy_writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        J = len(ch["copy"])
        for jj, core in enumerate(ch["copy"]):
            n_copy = _chunks_per_shard(shard_pages, jj, J, num_banks, run_pages)
            head = [in_addr, out_addr, shard_pages, jj, J, 0, arrival_addr, n_copy, n_copy]
            copy_reader_rt[core.x][core.y] = head + [1, ch["out"] | (1 << 28)]  # one entry: my own shard
            copy_writer_rt[core.x][core.y] = [out_addr, shard_pages, jj, J, ch["out"], n_copy]
        port_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in port_cores])
        copy_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in ch["copy"]])
        src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_READER_SOURCE,
                source_type=src,
                core_ranges=port_set,
                compile_time_args=ct + in_ct + out_ct + variant_ct,
                runtime_args=reader_rt,
                config=dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_SENDER_SOURCE,
                source_type=src,
                core_ranges=port_set,
                compile_time_args=ct + out_ct + variant_ct,
                runtime_args=sender_rt,
                config=dm("RISCV_0", NOC1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_READER_SOURCE,
                source_type=src,
                core_ranges=copy_set,
                compile_time_args=ct + in_ct + out_ct,
                runtime_args=copy_reader_rt,
                config=dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COPY_WRITER_SOURCE,
                source_type=src,
                core_ranges=copy_set,
                compile_time_args=ct + out_ct,
                runtime_args=copy_writer_rt,
                config=dm("RISCV_0", NOC1),
            ),
        ]
        if sinks:
            sink_rt = ttnn.RuntimeArgs()
            for key, core in sinks.items():
                sink_rt[core.x][core.y] = [arrival_addr, expect.get((coord,) + key, 0)]
            program.kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=_SINK_SOURCE,
                    source_type=src,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in sinks.values()]),
                    compile_time_args=[],
                    runtime_args=sink_rt,
                    config=dm("RISCV_0", NOC1),
                )
            )
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc


_SEM_CACHE = {}


def _shard_pages(t):
    shape = list(t.padded_shape)
    if t.layout == ttnn.TILE_LAYOUT:
        return (shape[-1] // 32) * (shape[-2] // 32) * math.prod(shape[:-2])
    return math.prod(shape[:-1])  # ROW_MAJOR: one page per row


def _check_dram_interleaved(t, what):
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"fabric_all_gather: {what} must be DRAM interleaved")
    if t.layout == ttnn.TILE_LAYOUT and list(t.shape)[-2:] != list(t.padded_shape)[-2:]:
        raise ValueError(f"fabric_all_gather: a TILE {what} must be tile-aligned (got {list(t.shape)})")


def output_shape(input_tensor, G, dim):
    shape = list(input_tensor.shape)
    assert dim in (0, len(shape) - 2), "gather along dim 0, or dim -2 with every leading dim 1"
    if dim != 0:
        assert all(s == 1 for s in shape[:dim]), "gather along dim -2 needs every leading dim to be 1"
    shape[dim] *= G
    return shape


def worker_cores(chips):
    """The logical cores the op runs on (the same on every chip: the union over chips)."""
    cores = {
        (c.x, c.y)
        for ch in chips.values()
        for c in list(ch["ports"].values()) + ch["copy"] + list(ch.get("sinks", {}).values())
    }
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(cores)])


def fabric_all_gather(
    input_tensor,
    *,
    cluster_axis=0,
    topology=ttnn.Topology.Linear,
    num_links=1,
    dim=0,
    placement="auto",
    scheme="ring",
    balance=False,
    desync=False,
    output=None,
    subdevice_id=None,
    sub_core_grid=None,
    ready_semaphore=None,
    data_valid_semaphore=None,
    inc_every=8,
    cb_bytes=112 * 1024,
    mode="relay",
):
    """All-gather `input_tensor` (TILE or ROW_MAJOR, any dtype, DRAM interleaved, one shard per chip) over each group;
    returns the output.

    mode: "relay" (default; intermediate chips forward other chips' shards), or, on a ring, "direct" (each chip
        unicasts its own shard to every other chip, the fabric routing it through the chips in between), "mcast"
        (one multicast per direction), "mcast_split" (each half of the shard multicast both ways, the far chip
        getting one half from each side) or "mcast_halves" (one half multicast all the way round each way). On a 2D
        fabric every mode but "relay" needs a straight ring (cluster_axis 0 / 1). See _schedule_routes.

    scheme="dual_cycles" gathers over the whole 2D torus with two edge-disjoint Hamiltonian cycles (cluster_axis=None,
    topology=Ring). The output is always in row-major chip order (as high_bw_all_gather's), for snakes too.

    output: a preallocated output (the gathered shape, same dtype and layout, DRAM interleaved), written in place; it
        can be reused call after call. Calls are fenced: no chip writes into a neighbour's output before the
        neighbour has started the same call, i.e. finished everything queued before it (including whatever read the
        previous result).
    sub_core_grid: logical worker cores (CoreRangeSet) the op may use; it picks its port and copy cores inside it.
    subdevice_id: the sub-device those cores belong to. Pass it together with its cores as sub_core_grid (the op
        cannot look a sub-device's cores up); the internal one-time semaphore setup then synchronizes only it.
    ready_semaphore, data_valid_semaphore: caller-owned global semaphores (initial value 0) covering every core the
        op uses (e.g. all of sub_core_grid; `worker_cores(plan(...)[0])` lists them), supplied together. They are
        left at 0 after every call. With them the op allocates nothing and never synchronizes, so it can run
        alongside work on other sub-devices."""
    if (ready_semaphore is None) != (data_valid_semaphore is None):
        raise ValueError("fabric_all_gather: ready_semaphore and data_valid_semaphore go together")
    if subdevice_id is not None and sub_core_grid is None:
        raise ValueError("fabric_all_gather: pass the sub-device's cores as sub_core_grid along with subdevice_id")
    mesh_device = input_tensor.device()
    chips, ring = plan(
        mesh_device,
        cluster_axis=cluster_axis,
        topology=topology,
        num_links=num_links,
        placement=placement,
        scheme=scheme,
        balance=balance,
        sub_core_grid=sub_core_grid,
        mode=mode,
    )
    G = next(iter(chips.values()))["G"]
    _check_dram_interleaved(input_tensor, "input")
    want = output_shape(input_tensor, G, dim)
    layout = input_tensor.layout
    if output is None:
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(want), input_tensor.dtype, layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
    elif list(output.shape) != want or output.dtype != input_tensor.dtype or output.layout != layout:
        raise ValueError(f"fabric_all_gather: output must be {want} {input_tensor.dtype} {layout}")
    _check_dram_interleaved(output, "output")
    if ready_semaphore is not None:
        sems = (data_valid_semaphore, ready_semaphore)
    else:
        port_cores = {
            (c.x, c.y) for ch in chips.values() for c in list(ch["ports"].values()) + list(ch.get("sinks", {}).values())
        }
        key = (id(mesh_device), tuple(sorted(port_cores)))
        if key not in _SEM_CACHE:
            cores = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(port_cores)]
            )
            sems = tuple(ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2))
            # every chip's counters must be zero before any chip's first call can increment a neighbour's
            ttnn.synchronize_device(mesh_device, sub_device_ids=[subdevice_id] if subdevice_id is not None else [])
            _SEM_CACHE[key] = (mesh_device, sems)
        sems = _SEM_CACHE[key][1]
    arrival_addr, ready_addr = (int(ttnn.get_global_semaphore_address(x)) for x in sems)
    desc = create_mesh_program_descriptor(
        mesh_device,
        input_tensor,
        output,
        arrival_addr,
        chips,
        num_links=num_links,
        ready_addr=ready_addr,
        desync=desync,
        cb_bytes=cb_bytes,
        inc_every=inc_every,
    )
    return ttnn.generic_op([input_tensor, output], desc)
