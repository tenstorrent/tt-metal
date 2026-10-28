# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_combine_l1 — combine level 1 (one row per slot, streamed home), rings at the consumer (prototype).

Every expert output row goes from its expert's chip to its token's home chip, Y first (along the column to the
home's row), then X, on a fixed link. No DRAM on the way: relays hold rows in L1 rings, home sums them into fp32
accumulators in L1. (Prototype source: the rows are read from a DRAM copy of the expert outputs, standing in for the
flat op's down cores; home sums without gate weights for now.)

Cores per chip: a port core per (direction, link) (sparse_all_gather_2d's placement) and A accumulator cores (the
same logical cores on every chip). Rings live at their consumer and each has exactly one producer (the upstream
sender of one link):
  port (d, l):  ring 0 CONTINUE  rows that arrived moving d and keep moving d (its own sender consumes them)
                rings 1 / 2 TURN (X ports only) rows that arrived moving Y+ / Y- and turn into d here
  accumulator:  rings 0..7 ACCUMULATE, one per incoming link (arrival direction x link), rows for its tokens
A sender writes a row (row + 16-byte header: home chip | token << 8) straight into the next chip's ring for what the
row does there, with a fused increment of that ring's arrival counter, if it holds a credit for that ring; a row
entering a continue ring (an injection, or a row that just turned) needs 2 credits (bubble rule), any other 1.
Continuing rows go first. A consumer that frees a slot increments a return counter on the port facing the ring's
producer; that port's sender sends the credits back upstream. Everything is count-based (host tables give each
core its counts); the CPU model (models/demos/deepseek_v3_d_p/tests/perf/scratch_combine_l1_protocol.py,
"consumer") found no deadlock for this scheme on lines and tori.
"""

import os

import torch

import ttnn

from .fabric_all_gather_py import NOC0, NOC1, _SEM_CACHE
from .sparse_all_gather_2d import DIRS, OPP, mesh_plan, place_ports

TILE = 2048  # bf16 tile bytes (rows are handled as flat tiles)
HDR = 64  # header bytes after each row in a slot (16 used)
CNT_OFF, CRED_OFF, RET_OFF, RINGS_OFF = 0, 128, 256, 1024  # per-core L1 area layout (32 counters each)
CB_INJ, CB_ROW, CB_META, CB_TAB, CB_ACC, CB_OUT = 0, 1, 2, 3, 4, 5

_PORT_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t cb_inj = get_compile_time_arg_val(0);
constexpr uint32_t cb_tab = get_compile_time_arg_val(1);
constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
constexpr uint32_t SLOT = get_compile_time_arg_val(3);
constexpr uint32_t INJ_B = __INJB__;
constexpr auto y_args = TensorAccessorArgs<4>();
constexpr auto t_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
// Port reader: the port's injection list (rows of local experts whose first hop leaves through this port), row +
// header into the injection CB, a few rows in flight.
void kernel_main() {
    size_t a = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tab_bytes = get_arg_val<uint32_t>(a++), off = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++);
    const auto y = TensorAccessor(y_args, y_addr, row_bytes);
    const auto tab = TensorAccessor(t_args, tab_addr, tab_bytes);
    cb_reserve_back(cb_tab, 1);
    const uint32_t tl = get_write_ptr(cb_tab);
    constexpr uint32_t TCH = 256;  // table entries per chunk (streamed: a hot port's list can be long)
    const uint32_t* e = nullptr;
    uint32_t chunk0 = 0xFFFFFFFF;
    auto entry = [&](uint32_t i) -> const uint32_t* {
        const uint32_t c0 = i / TCH * TCH;
        if (c0 != chunk0) {
            const uint32_t m = n - c0 < TCH ? n - c0 : TCH;
            const uint32_t start = (off + c0) * 8, al = start & ~63u, delta = start - al;
            noc_async_read(tab.get_noc_addr(0) + al, tl, ((m * 8 + delta + 63) / 64) * 64);
            noc_async_read_barrier();
            e = reinterpret_cast<const uint32_t*>(tl + delta);
            chunk0 = c0;
        }
        return e + 2 * (i - c0);  // (y row, header word)
    };
    constexpr uint32_t B = INJ_B;  // rows per read batch (the CB holds 2 batches)
    for (uint32_t i = 0; i < n; i += B) {
        const uint32_t k = n - i < B ? n - i : B;
        cb_reserve_back(cb_inj, k);
        uint32_t w = get_write_ptr(cb_inj);
        const uint32_t lim = get_local_cb_interface(cb_inj).fifo_limit, fsz = get_local_cb_interface(cb_inj).fifo_size;
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t* ej = entry(i + j);
            const uint32_t yrow = ej[0], hdr = ej[1];
            noc_async_read(y.get_noc_addr(yrow), w, row_bytes);
            volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + row_bytes);
            tr[0] = hdr;
            tr[3] = 1;  // arrival flags: the last 16-byte chunk of each NoC write stream (0 / 1 mod 32)
            tr[7] = 1;
            w += SLOT;
            if (w >= lim) w -= fsz;
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < k; ++j) cb_push_back(cb_inj, 1);
    }
}
"""

_PORT_SENDER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
constexpr uint32_t cb_inj = get_compile_time_arg_val(0);
constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
constexpr uint32_t SLOT = get_compile_time_arg_val(2);
constexpr uint32_t QC = get_compile_time_arg_val(3);  // continue ring slots
constexpr uint32_t QT = get_compile_time_arg_val(4);  // turn ring slots
constexpr uint32_t QA = get_compile_time_arg_val(5);  // accumulate ring slots
constexpr uint32_t A = get_compile_time_arg_val(6);   // accumulator cores per chip
constexpr uint32_t R = get_compile_time_arg_val(7);
constexpr uint32_t C = get_compile_time_arg_val(8);
constexpr uint32_t WRAPX = get_compile_time_arg_val(9);
constexpr uint32_t WRAPY = get_compile_time_arg_val(10);
constexpr uint32_t CNT = __CNT__, CRED = __CRED__, RET = __RET__, RINGS = __RINGS__;
constexpr uint32_t NCRED = 3 + A;
constexpr uint32_t PROF = __PROF__;
constexpr uint32_t CRED_DIV = __CREDDIV__;
constexpr uint32_t ROW_FLUSH = __ROWFLUSH__;
constexpr uint32_t PAIR = __PAIR__;
constexpr uint32_t ISLOT = __ISLOT__;  // injection page stride (64-byte aligned for the DRAM reads)  // two rows per fabric packet when they sit back to back  // probe: 0 = the fused increment does not wait for the row write  // credits go back once 1 / CRED_DIV of a ring is free (or when idle)

// direction index: 0 X+, 1 X-, 2 Y+, 3 Y-
FORCE_INLINE uint32_t opp(uint32_t d) { return d ^ 1; }

void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);  // this core's L1 area (rings + counters), same on every core
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_x = get_arg_val<uint32_t>(a++), ready_y = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t my_r = get_arg_val<uint32_t>(a++), my_c = get_arg_val<uint32_t>(a++);
    const uint32_t dir = get_arg_val<uint32_t>(a++), lnk = get_arg_val<uint32_t>(a++);
    const uint32_t n_inj = get_arg_val<uint32_t>(a++);   // rows this port injects
    const uint32_t n_send = get_arg_val<uint32_t>(a++);  // rows this port sends in all
    const uint32_t n_ret = get_arg_val<uint32_t>(a++);   // credits to send back (rows arriving over its link)
    // the next chip's cores: its continue port (dir, lnk), its X+ / X- ports (turn rings), its accumulator cores
    uint32_t cx[3], cy[3];
    for (uint32_t i = 0; i < 3; ++i) {
        cx[i] = get_arg_val<uint32_t>(a++);
        cy[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t ax[A], ay[A];
    for (uint32_t i = 0; i < A; ++i) {
        ax[i] = get_arg_val<uint32_t>(a++);
        ay[i] = get_arg_val<uint32_t>(a++);
    }
    // the upstream port (the neighbour's port facing back): where returned credits go
    const uint32_t ux = get_arg_val<uint32_t>(a++), uy = get_arg_val<uint32_t>(a++);
    const uint32_t nr = get_arg_val<uint32_t>(a++), nc = get_arg_val<uint32_t>(a++);  // the next chip
    // this chip's ports' return-counter cores, indexed by direction * 2 + link (0 when absent)
    uint32_t px[8], py[8];
    for (uint32_t i = 0; i < 8; ++i) {
        px[i] = get_arg_val<uint32_t>(a++);
        py[i] = get_arg_val<uint32_t>(a++);
    }
    const bool axis_x = dir < 2;
    const uint32_t a_dbg = a;  // dbg address, port index
    a += 2;

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    constexpr uint32_t NH = 8;  // row headers in rotation: flush (and free the sent slots) every NH sends
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hrows[NH];
    for (uint32_t i = 0; i < NH; ++i) {
        hrows[i] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hrows[i], dst_chip, dst_mesh);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hrow = hrows[0];
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hcred = PacketHeaderPool::allocate_header();
    (void)fabric_set_unicast_route(hcred, dst_chip, dst_mesh);
    conn.open();
    hrow->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hrow), sizeof(PACKET_HEADER_TYPE));
    {
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        noc_semaphore_wait_min(ready, 1);
        noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
        noc_async_atomic_barrier();
    }

    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + CNT);
    volatile tt_l1_ptr uint32_t* cred = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + CRED);
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + RET);
    const uint32_t ring0 = area + RINGS, ring1 = ring0 + QC * SLOT, ring2 = ring1 + QT * SLOT;
    uint32_t used[NCRED];
    for (uint32_t i = 0; i < NCRED; ++i) used[i] = 0;
    uint32_t qcap[NCRED];
    qcap[0] = QC;
    qcap[1] = qcap[2] = QT;
    for (uint32_t i = 3; i < NCRED; ++i) qcap[i] = QA;
    uint32_t taken[3] = {0, 0, 0};  // rows consumed from my rings 0..2
    uint32_t ret_done[NCRED];
    for (uint32_t i = 0; i < NCRED; ++i) ret_done[i] = 0;
    uint32_t sent = 0, inj_done = 0, returned = 0;
    uint32_t p_idle = 0, p_cred = 0, p_slot = 0, p_flush = 0, n_credpk = 0, n_blocked = 0, n_empty = 0;
    uint32_t p_send = 0, p_free = 0, n_iter = 0, p_scan = 0;
    const uint32_t p0 = get_timestamp_32b();

    // where a row headed for home (hr, hc) goes at the next chip: ring kind / target core / ring id / credit index
    struct Dest {
        uint32_t x, y, ring_base, cnt_idx, ci, cont;
    };
    auto dest_of = [&](uint32_t hdr) -> Dest {
        const uint32_t home = hdr & 0xFF, t = hdr >> 8;
        const uint32_t hr = home / C, hc = home % C;
        Dest d{};
        if (nr == hr && nc == hc) {  // home: the owner's accumulate ring for my link (arrival dir, link)
            const uint32_t o = t % A;
            d.x = ax[o];
            d.y = ay[o];
            const uint32_t k = dir * 2 + lnk;
            d.ring_base = RINGS + k * QA * SLOT;
            d.cnt_idx = k;
            d.ci = 3 + o;
            d.cont = 0;
            return d;
        }
        if (nr != hr) {  // still on the Y leg: continue (this port is a Y port)
            d.x = cx[0];
            d.y = cy[0];
            d.ring_base = RINGS;
            d.cnt_idx = 0;
            d.ci = 0;
            d.cont = 1;
            return d;
        }
        // on the home's row: X towards hc
        uint32_t xd;
        if (WRAPX) {
            const uint32_t fwd = (hc + C - nc) % C;
            xd = fwd * 2 <= C ? 0 : 1;
        } else {
            xd = hc > nc ? 0 : 1;
        }
        if (!axis_x) {  // turning: that X port's turn ring for my arrival direction (1: Y+, 2: Y-)
            d.x = cx[1 + xd];
            d.y = cy[1 + xd];
            const uint32_t k = dir == 2 ? 1 : 2;
            d.ring_base = RINGS + (k == 1 ? QC : QC + QT) * SLOT;
            d.cnt_idx = k;
            d.ci = 1 + xd;
            d.cont = 0;
            return d;
        }
        d.x = cx[0];
        d.y = cy[0];
        d.ring_base = RINGS;
        d.cnt_idx = 0;
        d.ci = 0;
        d.cont = 1;
        return d;
    };
    auto avail = [&](uint32_t ci) -> uint32_t { return qcap[ci] + cred[ci] - used[ci]; };
    auto landed = [&](uint32_t slot_addr) -> bool {
        volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr + row_bytes);
        return tr[3] != 0 && tr[7] != 0;
    };
    auto clear = [&](uint32_t slot_addr) {
        volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr + row_bytes);
        tr[3] = 0;
        tr[7] = 0;
    };
    auto send_row = [&](uint32_t src, uint32_t hdr, bool entering) -> bool {
        const uint32_t ts0 = get_timestamp_32b();
        const Dest d = dest_of(hdr);
        invalidate_l1_cache();
        if (avail(d.ci) < ((d.cont && entering) ? 2u : 1u)) {
            ++n_blocked;
            return false;
        }
        const uint32_t slot = d.ring_base + (used[d.ci] % qcap[d.ci]) * SLOT;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hrow = hrows[sent % NH];
        // a plain write: the row's trailer carries the arrival flags (written last within each 16-byte stream)
        hrow->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(d.x, d.y, area + slot)}, row_bytes + 32);
        uint32_t t0 = get_timestamp_32b();
        conn.wait_for_empty_write_slot();
        p_slot += get_timestamp_32b() - t0;
        conn.send_current_slot_non_blocking(src, row_bytes + 32, reinterpret_cast<uint32_t>(hrow));
        ++used[d.ci];
        ++sent;
        p_send += get_timestamp_32b() - ts0;
        return true;
    };
    // two rows that sit back to back in this core's L1 (src, src + SLOT): one scatter packet to their two slots
    auto send_two = [&](uint32_t src, uint32_t ha, uint32_t hb, bool entering) -> bool {
        if constexpr (!PAIR) return false;
        const uint32_t ts0 = get_timestamp_32b();
        const Dest da = dest_of(ha), db = dest_of(hb);
        invalidate_l1_cache();
        const uint32_t needa = (da.cont && entering) ? 2u : 1u, needb = (db.cont && entering) ? 2u : 1u;
        if (da.ci == db.ci) {
            if (avail(da.ci) < needa + 1) return false;
        } else if (avail(da.ci) < needa || avail(db.ci) < needb) {
            return false;
        }
        const uint32_t sa = da.ring_base + (used[da.ci] % qcap[da.ci]) * SLOT;
        ++used[da.ci];
        const uint32_t sb = db.ring_base + (used[db.ci] % qcap[db.ci]) * SLOT;
        ++used[db.ci];
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hrow = hrows[sent % NH];
        hrow->to_noc_unicast_scatter_write(
            NocUnicastScatterCommandHeader(
                {get_noc_addr(da.x, da.y, area + sa), get_noc_addr(db.x, db.y, area + sb)},
                {static_cast<uint16_t>(row_bytes + 32)}),
            2 * (row_bytes + 32));
        uint32_t t0 = get_timestamp_32b();
        conn.wait_for_empty_write_slot();
        p_slot += get_timestamp_32b() - t0;
        conn.send_current_slot_non_blocking(src, 2 * (row_bytes + 32), reinterpret_cast<uint32_t>(hrow));
        sent += 2;
        p_send += get_timestamp_32b() - ts0;
        return true;
    };
    // a slot of my ring k freed: one credit to that ring's producer, through the port facing it (only once the
    // row has left this core's L1: the frees wait for a flush)
    auto free_ring_now = [&](uint32_t k, uint32_t n) {
        taken[k] += n;
        uint32_t fd, ci;
        if (k == 0) {
            fd = opp(dir);  // rows arrived moving dir from nb(opp(dir))
            ci = 0;
        } else {
            fd = k == 1 ? 3 : 2;  // arrived moving Y+ comes from nb(Y-); moving Y- from nb(Y+)
            ci = dir == 0 ? 1 : 2;
        }
        const uint32_t pi = fd * 2 + lnk;
        noc_semaphore_inc(get_noc_addr(px[pi], py[pi], area + RET + 4 * ci), n);
        noc_semaphore_inc(get_noc_addr(px[pi], py[pi], area + RET + 4 * 31), n);  // "something to return"
    };
    uint32_t pend[3] = {0, 0, 0}, pend_inj = 0, unflushed = 0;
    auto flush_frees = [&]() {
        const uint32_t t0 = get_timestamp_32b();
        noc_async_writes_flushed();
        p_flush += get_timestamp_32b() - t0;
        const uint32_t tf0 = get_timestamp_32b();
        for (uint32_t k = 0; k < 3; ++k) {
            if (pend[k]) {
                const uint32_t base = k == 0 ? ring0 : (k == 1 ? ring1 : ring2), q = k == 0 ? QC : QT;
                for (uint32_t j = 0; j < pend[k]; ++j) clear(base + ((taken[k] + j) % q) * SLOT);
                free_ring_now(k, pend[k]);
                pend[k] = 0;
            }
        }
        p_free += get_timestamp_32b() - tf0;
        for (; pend_inj; --pend_inj) cb_pop_front(cb_inj, 1);  // page by page: a pop may end at the CB end, never cross it
        unflushed = 0;
    };

    while (sent < n_send || returned < n_ret) {
        bool did = false;
        const uint32_t it0 = get_timestamp_32b();
        ++n_iter;
        invalidate_l1_cache();
        // credits back upstream: in batches of half a ring, or everything when there is nothing else to do
        const bool any_ret = ret[31] != returned;
        auto give_back = [&](bool all) {
          for (uint32_t i = 0; i < NCRED; ++i) {
            const uint32_t dv = ret[i] - ret_done[i];
            if (dv && (all || dv * CRED_DIV >= qcap[i])) {
                const uint32_t t0 = get_timestamp_32b();
                ++n_credpk;
                hcred->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(ux, uy, area + CRED + 4 * i), dv, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hcred), sizeof(PACKET_HEADER_TYPE));
                ret_done[i] += dv;
                returned += dv;
                did = true;
                p_cred += get_timestamp_32b() - t0;
            }
          }
        };
        if (any_ret) give_back(false);
        // continuing rows first
        while (unflushed < NH && landed(ring0 + ((taken[0] + pend[0]) % QC) * SLOT)) {
            const uint32_t i0 = (taken[0] + pend[0]) % QC;
            const uint32_t s = ring0 + i0 * SLOT;
            const uint32_t hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + row_bytes)[0];
            if (i0 + 1 < QC && unflushed + 2 <= NH && landed(s + SLOT)) {
                const uint32_t hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + SLOT + row_bytes)[0];
                if (send_two(s, hdr, hb, false)) {
                    pend[0] += 2;
                    unflushed += 2;
                    did = true;
                    continue;
                }
            }
            if (!send_row(s, hdr, false)) break;
            ++pend[0];
            ++unflushed;
            did = true;
        }
        for (uint32_t k = 1; k <= 2; ++k) {
            while (axis_x && unflushed < NH && landed((k == 1 ? ring1 : ring2) + ((taken[k] + pend[k]) % QT) * SLOT)) {
                const uint32_t i0 = (taken[k] + pend[k]) % QT;
                const uint32_t s = (k == 1 ? ring1 : ring2) + i0 * SLOT;
                const uint32_t hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + row_bytes)[0];
                if (i0 + 1 < QT && unflushed + 2 <= NH && landed(s + SLOT)) {
                    const uint32_t hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + SLOT + row_bytes)[0];
                    if (send_two(s, hdr, hb, true)) {
                        pend[k] += 2;
                        unflushed += 2;
                        did = true;
                        continue;
                    }
                }
                if (!send_row(s, hdr, true)) break;
                ++pend[k];
                ++unflushed;
                did = true;
            }
        }
        while (unflushed < NH && inj_done < n_inj && cb_pages_available_at_front(cb_inj, pend_inj + 1)) {
            // the next injection page: pages are popped only after a flush, so step over the unpopped ones
            uint32_t s = get_read_ptr(cb_inj) + pend_inj * ISLOT;
            const uint32_t ilim = get_local_cb_interface(cb_inj).fifo_limit;
            if (s >= ilim) s -= get_local_cb_interface(cb_inj).fifo_size;
            const uint32_t hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + row_bytes)[0];
            if (!send_row(s, hdr, true)) break;
            ++pend_inj;
            ++inj_done;
            ++unflushed;
            did = true;
        }
        if (unflushed >= NH || (!did && unflushed)) {
            flush_frees();
            did = true;
        }
        if (!did && any_ret) give_back(true);
        if (!did) {
            p_idle += get_timestamp_32b() - it0;
            ++n_empty;
        }
    }
    if constexpr (PROF) {
        const uint32_t dbg = get_arg_val<uint32_t>(a_dbg), pidx = get_arg_val<uint32_t>(a_dbg + 1);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + RET + 256);
        d[0] = 0xDB900000 | pidx; d[1] = get_timestamp_32b() - p0; d[2] = p_idle; d[3] = p_cred; d[4] = p_slot;
        d[5] = p_flush; d[6] = n_credpk; d[7] = n_blocked; d[8] = n_empty; d[9] = n_send; d[10] = n_ret; d[11] = p_send;
        d[12] = p_free; d[13] = n_iter; d[14] = d[15] = 0;
        const InterleavedAddrGen<true> dg = {.bank_base_address = dbg, .page_size = 64};
        noc_async_write(area + RET + 256, get_noc_addr(pidx, dg), 64);
        noc_async_write_barrier();
    }
    flush_frees();
    // re-arm: my rings' arrival counters, the credits I received, the returns I sent
    for (uint32_t i = 0; i < NCRED; ++i) {
        if (used[i]) noc_semaphore_inc(get_noc_addr(area + CRED + 4 * i), 0u - used[i]);
        if (ret_done[i]) noc_semaphore_inc(get_noc_addr(area + RET + 4 * i), 0u - ret_done[i]);
    }
    if (returned) noc_semaphore_inc(get_noc_addr(area + RET + 4 * 31), 0u - returned);
    noc_async_atomic_barrier();
    conn.close();
}
"""

_ACC_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t cb_row = get_compile_time_arg_val(0);
constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
constexpr uint32_t cb_tab = get_compile_time_arg_val(2);
constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
constexpr uint32_t SLOT = get_compile_time_arg_val(4);
constexpr uint32_t QA = get_compile_time_arg_val(5);
constexpr uint32_t A = get_compile_time_arg_val(6);
constexpr uint32_t NT = get_compile_time_arg_val(7);
constexpr uint32_t CNT = __CNT__, RET = __RET__, RINGS = __RINGS__;
constexpr uint32_t PROF = __PROF__;
constexpr uint32_t ABL = __ABL__;  // 2: no row copies into the compute CB
constexpr auto y_args = TensorAccessorArgs<8>();
constexpr auto t_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
// Accumulator reader: its tokens' local rows (experts on the home chip) from DRAM, then whatever arrives in its 8
// accumulate rings; each row (NT tiles) + its token's local index to the compute; a freed ring slot is one credit to
// that ring's producer, through the port facing it.
void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tab_bytes = get_arg_val<uint32_t>(a++), off = get_arg_val<uint32_t>(a++);
    const uint32_t n_local = get_arg_val<uint32_t>(a++), n_rows = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);  // accumulator index
    uint32_t px[8], py[8];
    for (uint32_t i = 0; i < 8; ++i) {
        px[i] = get_arg_val<uint32_t>(a++);
        py[i] = get_arg_val<uint32_t>(a++);
    }
    const auto y = TensorAccessor(y_args, y_addr, row_bytes);
    const auto tab = TensorAccessor(t_args, tab_addr, tab_bytes);
    cb_reserve_back(cb_tab, 1);
    const uint32_t tl = get_write_ptr(cb_tab);
    constexpr uint32_t TCH = 256;
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + CNT);
    uint32_t taken[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    auto push = [&](uint64_t src, uint32_t t) {
        cb_reserve_back(cb_row, NT);
        cb_reserve_back(cb_meta, 1);
        if constexpr ((ABL & 2) == 0) noc_async_read(src, get_write_ptr(cb_row), row_bytes);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = t / A;
        noc_async_read_barrier();
        cb_push_back(cb_row, NT);
        cb_push_back(cb_meta, 1);
    };
    uint32_t done = 0;
    const uint32_t p0 = get_timestamp_32b();
    for (uint32_t c0 = 0; c0 < n_local; c0 += TCH) {
        const uint32_t m = n_local - c0 < TCH ? n_local - c0 : TCH;
        const uint32_t start = (off + c0) * 8, al = start & ~63u, delta = start - al;
        noc_async_read(tab.get_noc_addr(0) + al, tl, ((m * 8 + delta + 63) / 64) * 64);
        noc_async_read_barrier();
        const uint32_t* e = reinterpret_cast<const uint32_t*>(tl + delta);
        for (uint32_t i = 0; i < m; ++i, ++done) {
            push(y.get_noc_addr(e[2 * i]), e[2 * i + 1] >> 8);
        }
    }
    while (done < n_rows) {
        invalidate_l1_cache();
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t s = area + RINGS + (k * QA + taken[k] % QA) * SLOT;
            volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + row_bytes);
            if (tr[3] != 0 && tr[7] != 0) {
                const uint32_t hdr = tr[0];
                push(get_noc_addr(s), hdr >> 8);
                tr[3] = 0;
                tr[7] = 0;
                ++taken[k];
                ++done;
                // ring k = arrival direction * 2 + link: its producer is the neighbour on the opposite side
                const uint32_t pi = ((k >> 1) ^ 1) * 2 + (k & 1);
                noc_semaphore_inc(get_noc_addr(px[pi], py[pi], area + RET + 4 * (3 + me)), 1);
                noc_semaphore_inc(get_noc_addr(px[pi], py[pi], area + RET + 4 * 31), 1);
            }
        }
    }
    const uint32_t p_total = get_timestamp_32b() - p0;
    noc_async_atomic_barrier();
    if constexpr (PROF) {
        const uint32_t dbg = get_arg_val<uint32_t>(a), pidx = get_arg_val<uint32_t>(a + 1);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + RET + 256);
        d[0] = 0xDBA00000 | pidx; d[1] = p_total; d[2] = n_local; d[3] = n_rows;
        const InterleavedAddrGen<true> dg = {.bank_base_address = dbg, .page_size = 64};
        noc_async_write(area + RET + 256, get_noc_addr(pidx, dg), 64);
        noc_async_write_barrier();
    }
}
"""


def acc_compute_config(fidelity=None):
    """The accumulator kernel's compute: fp32 DEST, and the fp32 sums (CB_ACC) unpacked straight to DEST. The default
    unpack narrows a Float32 CB to TF32 on its way to DEST (genfiles: fp32 DEST with every CB fp32 / exponent B), so the
    copy of a sum rounded it twice - to 10 mantissa bits, then to bf16 - a bf16 ulp off at a midpoint."""
    comp = ttnn.ComputeConfigDescriptor()
    if fidelity is not None:
        comp.math_fidelity = fidelity
    comp.fp32_dest_acc_en = True
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[CB_ACC] = ttnn.UnpackToDestMode.UnpackToDestFp32  # (only copy_tile reads it: the out pass)
    comp.unpack_to_dest_mode = modes
    return comp


_ACC_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
// fp32 accumulators in L1 (packer L1 accumulation), then bf16 out
constexpr uint32_t ABL = __ABL__;
constexpr uint32_t DH = 4;  // fp32 tiles per DEST half
void kernel_main() {
    constexpr uint32_t cb_row = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);
    constexpr uint32_t MAXT = get_compile_time_arg_val(5);
    const uint32_t n_rows = get_arg_val<uint32_t>(0), n_tok = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(cb_row, cb_row, cb_acc);
    copy_init(cb_row);
    uint8_t seen[MAXT];
    for (uint32_t i = 0; i < MAXT; ++i) seen[i] = 0;
    cb_reserve_back(cb_acc, n_tok * NT);
    for (uint32_t r = 0; r < n_rows; ++r) {
        cb_wait_front(cb_meta, 1);
        const uint32_t t = read_tile_value(cb_meta, 0, 0);
        cb_pop_front(cb_meta, 1);
        cb_wait_front(cb_row, NT);
        if constexpr (ABL & 1) {  // ablation: no accumulation
            cb_pop_front(cb_row, NT);
            continue;
        }
        pack_reconfig_l1_acc(seen[t] ? 1 : 0);
        for (uint32_t j0 = 0; j0 < NT; j0 += DH) {  // a DEST half holds DH fp32 tiles
            const uint32_t nj = NT - j0 < DH ? NT - j0 : DH;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nj; ++j) copy_tile(cb_row, j0 + j, j);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nj; ++j) pack_tile<true>(j, cb_acc, t * NT + j0 + j);
            tile_regs_release();
        }
        seen[t] = 1;
        cb_pop_front(cb_row, NT);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, n_tok * NT);
    cb_wait_front(cb_acc, n_tok * NT);
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_acc);
    for (uint32_t t = 0; t < n_tok; ++t) {
        cb_reserve_back(cb_out, NT);
        for (uint32_t j0 = 0; j0 < NT; j0 += DH) {
            const uint32_t nj = NT - j0 < DH ? NT - j0 : DH;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nj; ++j) copy_tile(cb_acc, t * NT + j0 + j, j);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nj; ++j) pack_tile(j, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, NT);
    }
}
"""

_ACC_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t cb_out = get_compile_time_arg_val(0);
constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
constexpr uint32_t NT = get_compile_time_arg_val(2);
constexpr uint32_t A = get_compile_time_arg_val(3);
constexpr auto o_args = TensorAccessorArgs<4>();
void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0), n_tok = get_arg_val<uint32_t>(1), me = get_arg_val<uint32_t>(2);
    const auto o = TensorAccessor(o_args, o_addr, row_bytes);
    for (uint32_t i = 0; i < n_tok; ++i) {
        cb_wait_front(cb_out, NT);
        noc_async_write(get_read_ptr(cb_out), o.get_noc_addr(i * A + me), row_bytes);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, NT);
    }
    noc_async_write_barrier();
}
"""


def route(src, home, R, C, wrap_x, wrap_y, link, ytie=False):
    """[(chip, direction)] hops from src to home: Y first, then X (lines: forced; rings: shorter way). Ties as the
    fabric's routing tables (routing_table_generator.cpp: the first of (N, S) / (E, W)): Y- (North, row - 1) and X+
    (East, column + 1) - a multi-hop unicast must leave through its route's first hop. ytie (the combine's SPC_YTIE,
    its senders re-encode these routes): a Y tie goes Y+ when the source's and home's columns differ in parity - half
    the opposite row's chips each way."""
    hops = []
    r, c = src

    def step(n, a, b, wrap, tie_plus):
        if wrap:
            f = (b - a) % n
            plus = f * 2 < n or (f * 2 == n and tie_plus)
            return (1 if plus else -1), (f if plus else n - f)
        return (1 if b > a else -1), abs(b - a)

    s, k = step(R, r, home[0], wrap_y, ytie and (c + home[1]) % 2 == 1)
    for _ in range(k):
        hops.append(((r, c), "Y+" if s > 0 else "Y-"))
        r = (r + s) % R
    s, k = step(C, c, home[1], wrap_x, True)
    for _ in range(k):
        hops.append(((r, c), "X+" if s > 0 else "X-"))
        c = (c + s) % C
    return hops


def sparse_combine_l1_prepare(
    mesh_device, S, W, rows_by_chip, *, num_links=2, torus=(False, False), n_acc=16, qc=64, qt=32, qa=16, acc_cores=None
):
    """rows_by_chip[coord] = list of (y row on that chip, home coord, home token t, link). Builds tables / programs;
    returns run(y, out) -> out ([1, 1, S, W] bf16 per chip: home token t's sum)."""
    R, C = tuple(mesh_device.shape)
    row_bytes = W * 2
    NT = (row_bytes + TILE - 1) // TILE
    SLOT = ((row_bytes + 32 + 31) // 32) * 32  # ring slot: row + 32-byte trailer; two slots = one max-size packet
    ISLOT = ((row_bytes + 32 + 63) // 64) * 64  # injection page (DRAM reads land 64-byte aligned)
    chips, nb, (wrap_x, wrap_y) = mesh_plan(mesh_device, num_links, torus)
    port_core = place_ports(mesh_device, chips)
    used = {(c.x, c.y) for c in port_core.values()}
    grid = mesh_device.compute_with_storage_grid_size()
    if acc_cores is None:  # the same logical cores on every chip, away from every chip's port cores
        acc_cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in used][:n_acc]
    A = len(acc_cores)
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    # host tables: per port its injection list and counts, per accumulator its local list and counts
    inj = {k: [] for k in port_core}
    n_send = {k: 0 for k in port_core}
    n_ret = {k: 0 for k in port_core}
    loc = {(ch, i): [] for ch in chips for i in range(A)}
    n_rows = {(ch, i): 0 for ch in chips for i in range(A)}
    for ch, rows in rows_by_chip.items():
        for yrow, home, t, l in rows:
            hdr = (home[0] * C + home[1]) | (t << 8)
            o = t % A
            n_rows[(home, o)] += 1
            hops = route(ch, home, R, C, wrap_x, wrap_y, l)
            if not hops:
                loc[(home, o)].append((yrow, hdr))
                continue
            inj[(ch, hops[0][1], l)].append((yrow, hdr))
            for hc, d in hops:
                n_send[(hc, d, l)] += 1
                # the receiving chip's port facing back returns this row's credit
                n_ret[(nb(hc, d), OPP[d], l)] += 1
    NCRED = 3 + A
    assert NCRED < 31, "credit counters per sender: 3 + accumulators < 31"
    area_bytes = RINGS_OFF + max((qc + 2 * qt) * SLOT, 8 * qa * SLOT)
    all_cores = sorted({(c.x, c.y) for c in port_core.values()} | {(c.x, c.y) for c in acc_cores})
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in all_cores])
    key = (id(mesh_device), tuple(all_cores), "combine_l1", area_bytes)
    if key not in _SEM_CACHE:
        ready = ttnn.create_global_semaphore(mesh_device, core_set, 0)
        words = area_bytes // 4
        shard = ttnn.ShardSpec(core_set, [1, words], ttnn.ShardOrientation.ROW_MAJOR)
        area_t = ttnn.from_torch(
            torch.zeros(1, 1, len(all_cores), words, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.synchronize_device(mesh_device)
        _SEM_CACHE[key] = (mesh_device, ready, area_t)
    _, ready, area_t = _SEM_CACHE[key]
    ready_addr = int(ttnn.get_global_semaphore_address(ready))
    area = int(area_t.buffer_address())

    # one table tensor per chip: the ports' injection lists then the accumulators' local lists ((y row, header) pairs)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    tabs, offs = {}, {}
    for ch in chips:
        lst = []
        for k in sorted(k for k in port_core if k[0] == ch):
            offs[("p",) + k] = len(lst)
            lst += inj[k]
        for i in range(A):
            offs[("a", ch, i)] = len(lst)
            lst += loc[(ch, i)]
        tabs[ch] = lst
    n_tab = max(max(len(v) for v in tabs.values()), 1)
    th = torch.zeros(R, C, 1, 2 * n_tab, dtype=torch.int64)
    for (r, c), lst in tabs.items():
        if lst:
            th[r, c, 0, : 2 * len(lst)] = torch.tensor(lst, dtype=torch.int64).reshape(-1)
    tab_t = ttnn.from_torch(
        th.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    tab_bytes = 8 * n_tab
    MAXT = (S + A - 1) // A
    prof = os.environ.get("SPC_PROF") == "1"
    inj_b = int(os.environ.get("SPC_INJB", "8"))
    subs = {
        "__CNT__": str(CNT_OFF),
        "__CRED__": str(CRED_OFF),
        "__RET__": str(RET_OFF),
        "__RINGS__": str(RINGS_OFF),
        "__PROF__": str(int(prof)),
        "__ABL__": os.environ.get("SPC_ABL", "0"),
        "__INJB__": str(inj_b),
        "__CREDDIV__": os.environ.get("SPC_CREDDIV", "2"),
        "__ROWFLUSH__": os.environ.get("SPC_ROWFLUSH", "1"),
        "__PAIR__": os.environ.get("SPC_PAIR", "1"),
        "__ISLOT__": str(ISLOT),
    }
    dbg_t = ttnn.from_torch(
        torch.zeros(R, C, 128, 16, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1)),
    )
    srcs = {}
    for name, ksrc in (
        ("pr", _PORT_READER),
        ("ps", _PORT_SENDER),
        ("ar", _ACC_READER),
        ("ac", _ACC_COMPUTE),
        ("aw", _ACC_WRITER),
    ):
        for k, v in subs.items():
            ksrc = ksrc.replace(k, v)
        srcs[name] = ksrc
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    state = {}

    def build(y, out):
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        t_ct = list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
        o_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        mesh_desc = ttnn.MeshProgramDescriptor()
        for ch, ports in chips.items():
            r, c = ch
            program = ttnn.ProgramDescriptor()
            my = [(d, li, port_core[(ch, d, li)]) for d, p in ports.items() for li in range(len(p["links"]))]
            pset = ttnn.CoreRangeSet([ttnn.CoreRange(k, k) for _, _, k in my])
            aset = ttnn.CoreRangeSet([ttnn.CoreRange(k, k) for k in acc_cores])
            pxy = []
            for i in range(8):
                d, li = DIRS[i // 2], i % 2
                k = (ch, d, li)
                v = virt(port_core[k]) if k in port_core else None
                pxy += [v.x, v.y] if v else [0, 0]
            tab_cb = 256 * 8 + 128  # each core streams its own entries, 256 at a time
            program.cbs = [
                ttnn.CBDescriptor(
                    total_size=2 * inj_b * ISLOT,
                    core_ranges=pset,
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_INJ, ttnn.uint32, ISLOT)],
                ),
                ttnn.CBDescriptor(
                    total_size=tab_cb,
                    core_ranges=pset.merge(aset),
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_TAB, ttnn.uint32, tab_cb)],
                ),
                ttnn.CBDescriptor(
                    total_size=2 * NT * TILE,
                    core_ranges=aset,
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_ROW, ttnn.bfloat16, TILE)],
                ),
                ttnn.CBDescriptor(
                    total_size=2 * 32,
                    core_ranges=aset,
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_META, ttnn.uint32, 32)],
                ),
                ttnn.CBDescriptor(
                    total_size=MAXT * NT * 4096,
                    core_ranges=aset,
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_ACC, ttnn.float32, 4096)],
                ),
                ttnn.CBDescriptor(
                    total_size=2 * NT * TILE,
                    core_ranges=aset,
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_OUT, ttnn.bfloat16, TILE)],
                ),
            ]
            pr_rt, ps_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for d, li, core in my:
                k = (ch, d, li)
                pr_rt[core.x][core.y] = [
                    int(y.buffer_address()),
                    int(tab_t.buffer_address()),
                    tab_bytes,
                    offs[("p",) + k],
                    len(inj[k]),
                ]
                peer = ports[d]["peer"]
                rc = virt(port_core[(peer, OPP[d], li)])
                pn = node(peer)
                nxt = []
                for kk in ((peer, d, li), (peer, "X+", li), (peer, "X-", li)):
                    v = virt(port_core[kk]) if kk in port_core else None
                    nxt += [v.x, v.y] if v else [0, 0]
                args = [
                    area,
                    ready_addr,
                    rc.x,
                    rc.y,
                    int(pn.mesh_id),
                    int(pn.chip_id),
                    r,
                    c,
                    DIRS.index(d),
                    li,
                    len(inj[k]),
                    n_send[k],
                    n_ret[k],
                ] + nxt
                for ac in acc_cores:
                    v = virt(ac)
                    args += [v.x, v.y]
                args += [rc.x, rc.y, peer[0], peer[1]] + pxy
                args += [int(dbg_t.buffer_address()), DIRS.index(d) * 2 + li]
                args += list(ttnn.setup_fabric_connection(node(ch), pn, ports[d]["links"][li], program, core))
                ps_rt[core.x][core.y] = args
            ar_rt, ac_rt, aw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, core in enumerate(acc_cores):
                n_tok = len(range(i, S, A))
                ar_rt[core.x][core.y] = (
                    [
                        area,
                        int(y.buffer_address()),
                        int(tab_t.buffer_address()),
                        tab_bytes,
                        offs[("a", ch, i)],
                        len(loc[(ch, i)]),
                        n_rows[(ch, i)],
                        i,
                    ]
                    + pxy
                    + [int(dbg_t.buffer_address()), 8 + i]
                )
                ac_rt[core.x][core.y] = [n_rows[(ch, i)], n_tok]
                aw_rt[core.x][core.y] = [int(out.buffer_address()), n_tok, i]
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=srcs["pr"],
                    source_type=st,
                    core_ranges=pset,
                    compile_time_args=[CB_INJ, CB_TAB, row_bytes, ISLOT] + y_ct + t_ct,
                    runtime_args=pr_rt,
                    config=dm("RISCV_1", NOC0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["ps"],
                    source_type=st,
                    core_ranges=pset,
                    compile_time_args=[CB_INJ, row_bytes, SLOT, qc, qt, qa, A, R, C, int(wrap_x), int(wrap_y)],
                    runtime_args=ps_rt,
                    config=dm("RISCV_0", NOC1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["ar"],
                    source_type=st,
                    core_ranges=aset,
                    compile_time_args=[CB_ROW, CB_META, CB_TAB, row_bytes, SLOT, qa, A, NT] + y_ct + t_ct,
                    runtime_args=ar_rt,
                    config=dm("RISCV_1", NOC0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["ac"],
                    source_type=st,
                    core_ranges=aset,
                    compile_time_args=[CB_ROW, CB_META, CB_ACC, CB_OUT, NT, MAXT],
                    runtime_args=ac_rt,
                    config=acc_compute_config(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["aw"],
                    source_type=st,
                    core_ranges=aset,
                    compile_time_args=[CB_OUT, row_bytes, NT, A] + o_ct,
                    runtime_args=aw_rt,
                    config=dm("RISCV_0", NOC1),
                ),
            ]
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*ch), ttnn.MeshCoordinate(*ch))] = program
        return mesh_desc

    def run(y, out):
        k = (int(y.buffer_address()), int(out.buffer_address()))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(y, out)
        return ttnn.generic_op([y, tab_t, out], state["desc"])

    run.counts = dict(n_send=n_send, n_ret=n_ret, n_rows=n_rows)
    run.dbg = dbg_t
    return run
