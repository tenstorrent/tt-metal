// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port reader (NCRISC / NoC0), one per (direction, link) per chip — TRANSPORT + OVERLAP.
//
// send_entry (per outgoing entry, in send order): own entries wait until every own member that this port reads
//   from has loaded its block (sem_own_loaded); relay entries wait until the upstream entry they forward has
//   landed (arrival counter >= threshold). The entry's runs are read from the compute cores' cb_x (mirror layout:
//   the same core / offset on every chip) into cb_port_stage, one slot per packet (<= chunk_pages tiles).
//   Stream mode (default): every slot's read carries its own NoC transaction id; up to the whole stage ring is in
//   flight and each slot is pushed, in order, the moment its own read has landed (no batch barrier: one 14 KB L1
//   read in flight is latency-bound). Batch mode: batches of `group` slots, one read barrier + push per
//   `push_every` slots (a batch also ends at the entry end).
// notify_landed (serviced before every blocking point, so a final-hop landing is never starved by a relay wait or
//   by stage back-pressure): incoming entry e has landed once the arrival counter reaches e + 1 (one fused
//   increment on the entry's last packet); every member it covers gets one local increment of sem_landed.
// drain_arrivals: after every incoming entry is notified, the arrival counter is re-armed by decrementing it by
//   exactly what was consumed (global semaphore, 0 at every call boundary).
//
// Raw dataflow API (documented helper bypass): the kernel_lib ccl stream helpers are unmeasured for this pattern
// (op_design.md: "port sender ... copy the measured raw pattern"), and none of them polls two event sources.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_common.hpp"
#include "all_gather_rms_norm_port_pack.hpp"

using namespace agrms_cb;

constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
constexpr uint32_t chunk_pages = get_compile_time_arg_val(1);  // tiles per packet slot
constexpr uint32_t group = get_compile_time_arg_val(2);        // packet slots per batch; the CB holds 2 batches
constexpr uint32_t sem_own_loaded_id = get_compile_time_arg_val(3);
constexpr uint32_t sem_landed_id = get_compile_time_arg_val(4);
constexpr uint32_t push_every = get_compile_time_arg_val(5);  // batch mode: packets per read barrier + push (<= group)
constexpr bool stream = get_compile_time_arg_val(6) != 0;     // stream mode: per-slot transaction ids, no batch barrier
constexpr bool pack = get_compile_time_arg_val(7) != 0;       // packets span runs of an entry (fabric scatter write)
constexpr uint32_t slots = get_compile_time_arg_val(8) * group;  // cb_port_stage ring (one packet per slot)
static_assert(!stream || slots <= 15, "one NoC transaction id per stage slot (ids 1..15)");
constexpr uint32_t chunk_bytes = chunk_pages * tile_bytes;
#ifndef AGRMS_ABL
#define AGRMS_ABL 0
#endif
constexpr bool abl_no_read = (AGRMS_ABL & 1) != 0;   // perf probes: skip the compute-L1 reads
constexpr bool abl_no_relay = (AGRMS_ABL & 2) != 0;  // perf probes: relay entries do not wait for their arrival

constexpr uint32_t OWN_MASK_FLAG = 1u << 31;  // own entry wait word: flag | bit mask of its own members

FORCE_INLINE uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(i); }

void kernel_main() {
    const uint32_t arrival_addr = arg(0);
    const uint32_t own_expected = arg(1);
    const uint32_t num_out = arg(2);
    const uint32_t num_in = arg(3);
    uint32_t a = 4;
    // locate the incoming entries (after the outgoing ones)
    uint32_t in_ptr = a;
    for (uint32_t e = 0; e < num_out; ++e) {
        in_ptr += 2 + 2 * arg(in_ptr + 1);
    }

    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    volatile tt_l1_ptr uint32_t* own_loaded =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_loaded_id));
    const uint32_t landed_addr = get_semaphore(sem_landed_id);
    const uint32_t x_base = get_write_ptr(cb_x);  // identical on every core of every chip (cb_x is the first CB)

    uint32_t next_in = 0;
    auto service = [&]() {
        while (next_in < num_in) {
            invalidate_l1_cache();
            if (*arrived < next_in + 1) {
                return;
            }
            const uint32_t n = arg(in_ptr);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t xy = arg(in_ptr + 1 + i);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, landed_addr), 1);
            }
            in_ptr += 1 + n;
            ++next_in;
        }
    };

    AGZ("pr_all");
    bool own_ready = own_expected == 0;
    // pos: stage slots (one per packet) issued so far; head: slots pushed to the sender. A slot's address is fixed
    // by its ring position (the sender pops whole batches of `group` slots aligned to multiples of `group`).
    const uint32_t stage_base = get_write_ptr(cb_stage);
    uint32_t pos = 0, head = 0;
    // batch mode: slots of the current reserved run of `need` slots read so far / already pushed
    uint32_t batch = 0, pushed = 0, need = 0, wptr = 0;
    // stream mode: hand every landed slot to the sender, in order (each slot's read carries its own transaction id)
    auto drain = [&]() {
        if constexpr (stream) {
            while (head < pos && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + head % slots)) {
                cb_push_back(cb_stage, chunk_pages);
                ++head;
            }
        }
    };
    for (uint32_t e = 0; e < num_out; ++e) {
        const uint32_t wait = arg(a);
        const uint32_t n_runs = arg(a + 1);
        a += 2;
        if ((wait & OWN_MASK_FLAG) != 0) {
            // own entry, per-entry readiness: every own member signals its own bit (distinct bits never carry)
            const uint32_t mask = wait & ~OWN_MASK_FLAG;
            while (true) {
                drain();
                service();
                invalidate_l1_cache();
                if ((*own_loaded & mask) == mask) {
                    break;
                }
            }
        } else if (wait == 0 || abl_no_relay) {
            while (!own_ready) {
                drain();
                service();
                invalidate_l1_cache();
                own_ready = *own_loaded >= own_expected;
            }
        } else {
            while (true) {
                drain();
                service();
                invalidate_l1_cache();
                if (*arrived >= wait) {
                    break;
                }
            }
        }
        agrms_pack::EntryPacker<pack, chunk_pages> packer{a, n_runs};
        agrms_pack::Piece pc[agrms_pack::MAX_PIECES];
        // one packet = one stage slot; its pieces are read back to back into the slot
        auto read_packet = [&](uint32_t dst, uint32_t np) {
            if (!abl_no_read) {
                for (uint32_t i = 0; i < np; ++i) {
                    noc_async_read(
                        get_noc_addr(pc[i].xy & 0xFFFF, pc[i].xy >> 16, x_base + pc[i].off * tile_bytes),
                        dst,
                        pc[i].n * tile_bytes);
                    dst += pc[i].n * tile_bytes;
                }
            }
        };
        {
            while (!packer.done()) {
                uint32_t tiles;
                bool last;
                const uint32_t np = packer.next(pc, tiles, last);
                if constexpr (stream) {
                    // a free slot: fewer than `slots` in flight + unpopped
                    while (pos - head == slots ||
                           !cb_pages_reservable_at_back(cb_stage, chunk_pages * (pos - head + 1))) {
                        drain();
                        service();
                    }
                    noc_async_read_set_trid(1 + pos % slots);
                    read_packet(stage_base + (pos % slots) * chunk_bytes, np);
                    ++pos;
                    drain();
                } else {
                    if (batch == 0) {
                        need = group - pos % group;
                        while (!cb_pages_reservable_at_back(cb_stage, chunk_pages * need)) {
                            service();
                        }
                        cb_reserve_back(cb_stage, chunk_pages * need);
                        wptr = get_write_ptr(cb_stage);
                    }
                    read_packet(wptr + batch * chunk_bytes, np);
                    ++batch;
                    if (batch - pushed == push_every || batch == need) {
                        noc_async_read_barrier();
                        cb_push_back(cb_stage, chunk_pages * (batch - pushed));
                        pushed = batch;
                    }
                    if (batch == need) {
                        pos += batch;
                        batch = pushed = 0;
                    }
                }
            }
        }
        a += 2 * n_runs;
        if constexpr (!stream) {
            if (batch > 0) {  // an entry's tail is pushed at once: its fused increment must not wait for later entries
                if (batch > pushed) {
                    noc_async_read_barrier();
                    cb_push_back(cb_stage, chunk_pages * (batch - pushed));
                }
                pos += batch;
                batch = pushed = 0;
            }
        }
    }
    while (head < pos) {  // stream mode: the last slots (batch mode keeps head == pos == 0)
        drain();
        service();
    }
    if constexpr (stream) {
        noc_async_read_set_trid(0);
    }
    while (next_in < num_in) {
        service();
    }
    if (num_in > 0) {
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - num_in);
    }
    noc_async_atomic_barrier();
}
