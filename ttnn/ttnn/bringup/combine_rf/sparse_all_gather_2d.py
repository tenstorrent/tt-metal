# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_all_gather_2d — the plan-free sparse all-gather on a 2D mesh (prototype).

Every chip (r, c) of an R x C mesh (each axis a ring or a line) owns S rows, already in place in its output at slots
idx * S + t (idx = r * C + c; the producer, e.g. the latent projection, writes them there). Each own row t comes with a
32-bit route word: bits 0-15 the destination chips (bit idx), bit 16 the order (0: X first, i.e. along the row; 1: Y
first), bit 17 the link (route_words(N) = 1); on more than 16 chips two words per row (route_words(N) = 2): the
destination mask, then order | link << 1. The word can come from a planner (order / link chosen by link load) or from the stateless
rule; either way the trees are dimension-order trees with stateless spans (ring: minimal span, ties by bit 30 of
t * 2654435761; line: forced), the same as the CPU model in models/demos/deepseek_v3_d_p/tests/perf/
scratch_sparse_ag_2d_protocol.py, which verified the protocol below.

Ports: one port core per (direction X+ / X- / Y+ / Y-, link) that has a neighbour. Its reader merges own rows and up to
three input manifest rings (0: rows continuing in its direction, 1 / 2: rows turning into its axis that arrived moving
+ / - along the other axis); its sender sends each row one hop (two rows per scatter packet when they fit) and stages,
for the next chip, an entry in each of up to three streams: continue (same direction), turn + and turn - (the next
chip's two ports of the other axis, when the row is on its first leg and that chip's line needs it). Entries are 16
bytes (word 0: slot | hops left << 16 | generation << 20 | order << 24; word 1: the destination mask); a flushed fused
write + increment vouches for the rows sent before it on the same link. END(g) = no more rows that had travelled g
hops before leaving; a port emits END(0) after its own rows, END(g) after every input delivered END(g - 1), and is done
when END(GMAX) is out and every input delivered END(GMAX). Rings hold a whole call (no credits, nothing waits for
space): ring k holds S entries per origin chip whose trees can reach a ring k of some port (manifest_layout), not N S.
Every link ends in a ring of the chip it leads to, so a chip's ports are done only after every incoming link's rows
landed: a link into a chip with no port to continue or turn into (a 1-D line's end chip) carries a terminal stream
instead - only its END(GMAX), into ring 0 of the receiving port of that link (free there: nothing is behind it), which
gates that port's completion but not its END order (_wiring).
"""

import os
import functools

import torch

import ttnn

from .fabric_all_gather_py import (
    NOC0,
    NOC1,
    _SEM_CACHE,
    _worker_nearest_noc0,
    _worker_nearest_noc1,
    allowed_cores,
    eth_noc_coord,
    probe_ethernet_cores,
)

CB_ROWS, CB_META, CB_PLAN, CB_STAGE, CB_DBG = 0, 1, 2, 3, 4
META_PAGE = 16
DIRS = ("X+", "X-", "Y+", "Y-")
OPP = {"X+": "X-", "X-": "X+", "Y+": "Y-", "Y-": "Y+"}


def hbit(t, k):
    return ((t * 2654435761) & 0xFFFFFFFF) >> k & 1


def span_entry(n, wrap, pos, tmask, tie):
    """(a, b) packed a | b << 4: hops forward / backward from pos on an n-line / n-ring covering the set bits."""
    offs = {(x - pos) % n if wrap else x - pos for x in range(n) if (tmask >> x) & 1 and x != pos}
    if not offs:
        return 0
    if not wrap:
        a = max([o for o in offs if o > 0], default=0)
        b = max([-o for o in offs if o < 0], default=0)
        return a | b << 4
    cands = [(a, b) for a in range(n) for b in range(n - a) if all(o <= a or n - o <= b for o in offs)]
    m = min(a + b for a, b in cands)
    cands = [c for c in cands if c[0] + c[1] == m]
    mm = min(max(c) for c in cands)
    cands = sorted(c for c in cands if max(c) == mm)
    a, b = cands[tie if len(cands) > 1 else 0]
    return a | b << 4


@functools.lru_cache(maxsize=None)  # (pure in its integer arguments: per-call host time otherwise)
def span_table(n, wrap):
    return [span_entry(n, wrap, pos, tm, tie) for pos in range(n) for tm in range(1 << n) for tie in range(2)]


def ring_table(n, wrap):
    """The kernels' ring span table: span_entry from position 0 by rotated mask and tie ([0] for a line)."""
    return [span_entry(n, True, 0, rm, tie) for rm in range(1 << n) for tie in range(2)] if wrap else [0]


@functools.lru_cache(maxsize=None)  # (pure in its integer arguments: per-call host time otherwise)
def span_subs(R, C, wrap_x, wrap_y):
    """The _COMMON substitutions of the span helpers."""
    return {
        "__WRAPX__": "true" if wrap_x else "false",
        "__WRAPY__": "true" if wrap_y else "false",
        "__XRING__": ", ".join(map(str, ring_table(C, wrap_x))),
        "__YRING__": ", ".join(map(str, ring_table(R, wrap_y))),
    }


_COMMON = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t R = __R__;
constexpr uint32_t C = __C__;
constexpr uint32_t S = __S__;
constexpr uint32_t GMAX = __GMAX__;
constexpr bool WRAP_X = __WRAPX__, WRAP_Y = __WRAPY__;
// ring spans by the destination mask rotated to the position (a ring looks the same from every chip: 2^n x 2 bytes, not
// n x 2^n x 2, which an 8-ring would put at 4 KB of the RISC's local memory); a line's span needs no table
constexpr uint8_t XRING[WRAP_X ? (1u << C) * 2 : 1] = {__XRING__};
constexpr uint8_t YRING[WRAP_Y ? (1u << R) * 2 : 1] = {__YRING__};
constexpr uint32_t END = 1u << 31;
constexpr uint32_t SKIP = 1u << 30;
constexpr uint32_t META_WORDS = 4;

constexpr uint32_t RW = __RW__;  // route words per own row (1: mask | order << 16 | link << 17; 2: mask, order | link << 1)
FORCE_INLINE uint32_t route_mask(const uint32_t* w, uint32_t t) { return RW == 2 ? w[2 * t] : w[t] & 0xFFFF; }
FORCE_INLINE uint32_t route_flags(const uint32_t* w, uint32_t t) { return RW == 2 ? w[2 * t + 1] : w[t] >> 16; }
// ring k's first entry in the manifest (ring_offs: ring 1's | ring 2's << 16, in entries)
FORCE_INLINE uint32_t ring_off(uint32_t ring_offs, uint32_t k) { return k == 0 ? 0 : (ring_offs >> (16 * (k - 1))) & 0xFFFF; }
FORCE_INLINE uint32_t tie_of(uint32_t t) { return ((t * 2654435761u) >> 30) & 1; }
// (a, b) packed a | b << 4: hops forward / backward from pos covering mask m's set positions (pos itself ignored)
template <uint32_t n, bool wrap>
FORCE_INLINE uint32_t span_of(const uint8_t* ring, uint32_t pos, uint32_t m, uint32_t t) {
    m &= ~(1u << pos) & ((1u << n) - 1);
    if (m == 0) return 0;
    if constexpr (wrap) {
        return ring[(((m >> pos) | (m << (n - pos))) & ((1u << n) - 1)) * 2 + tie_of(t)];
    } else {
        const uint32_t hi = 31 - __builtin_clz(m), lo = __builtin_ctz(m);
        return (hi > pos ? hi - pos : 0) | ((lo < pos ? pos - lo : 0) << 4);
    }
}
FORCE_INLINE uint32_t xspan(uint32_t pos, uint32_t cm, uint32_t t) { return span_of<C, WRAP_X>(XRING, pos, cm, t); }
FORCE_INLINE uint32_t yspan(uint32_t pos, uint32_t rm, uint32_t t) { return span_of<R, WRAP_Y>(YRING, pos, rm, t); }
// columns holding a destination (any row)
FORCE_INLINE uint32_t cols_of(uint32_t m) {
    uint32_t cm = 0;
    for (uint32_t r = 0; r < R; ++r) cm |= (m >> (r * C)) & ((1u << C) - 1);
    return cm;
}
// rows holding a destination (any column)
FORCE_INLINE uint32_t rows_of(uint32_t m) {
    uint32_t rm = 0;
    for (uint32_t r = 0; r < R; ++r) rm |= (((m >> (r * C)) & ((1u << C) - 1)) != 0) << r;
    return rm;
}
// destinations in column c, as a row mask
FORCE_INLINE uint32_t rows_in_col(uint32_t m, uint32_t c) {
    uint32_t rm = 0;
    for (uint32_t r = 0; r < R; ++r) rm |= ((m >> (r * C + c)) & 1) << r;
    return rm;
}
// destinations in row r, as a column mask
FORCE_INLINE uint32_t cols_in_row(uint32_t m, uint32_t r) { return (m >> (r * C)) & ((1u << C) - 1); }
"""

_READER = (
    _COMMON
    + r"""
constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
constexpr uint32_t cb_plan = get_compile_time_arg_val(2);
constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
constexpr uint32_t group = get_compile_time_arg_val(4);
constexpr uint32_t PLAN_BYTES = get_compile_time_arg_val(5);
constexpr auto out_args = TensorAccessorArgs<6>();
constexpr auto plan_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
constexpr auto dbg_args = TensorAccessorArgs<plan_args.next_compile_time_args_offset()>();

// Port reader: merge the input rings (entries whose rows have landed) and the own rows this port starts, hand them to
// the sender `group` at a time with a 16-byte metadata page each (word 0 as the manifest entry, word 1 the mask).
void kernel_main() {
    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    const uint32_t man_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ring_offs = get_arg_val<uint32_t>(a++);
    uint32_t cnt_addr[3];
    for (uint32_t k = 0; k < 3; ++k) cnt_addr[k] = get_arg_val<uint32_t>(a++);
    // which inputs exist (bit k: ring k); bit 3 + k: ring k is a terminal stream (only the link's END(GMAX): it gates
    // this port's completion - the link's rows landed - not its END order)
    const uint32_t ring_mask = get_arg_val<uint32_t>(a++);
    const uint32_t my_r = get_arg_val<uint32_t>(a++);
    const uint32_t my_c = get_arg_val<uint32_t>(a++);
    const uint32_t dir = get_arg_val<uint32_t>(a++);  // 0 X+, 1 X-, 2 Y+, 3 Y-
    const uint32_t lnk = get_arg_val<uint32_t>(a++);
    const uint32_t dbg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t port_idx = get_arg_val<uint32_t>(a++);
    const uint32_t spin_limit = get_arg_val<uint32_t>(a++);
    const uint32_t me = my_r * C + my_c;
    const bool axis_x = dir < 2;
    const bool plus = (dir & 1) == 0;
    const auto out = TensorAccessor(out_args, out_addr, row_bytes);
    const auto plan = TensorAccessor(plan_args, plan_addr, PLAN_BYTES);

    cb_reserve_back(cb_plan, 1);
    const uint32_t plan_l1 = get_write_ptr(cb_plan);
    noc_async_read(plan.get_noc_addr(0), plan_l1, PLAN_BYTES);
    noc_async_read_barrier();
    const uint32_t* words = reinterpret_cast<const uint32_t*>(plan_l1);

    volatile tt_l1_ptr uint32_t* cnt[3];
    volatile tt_l1_ptr uint32_t* ring[3];
    for (uint32_t k = 0; k < 3; ++k) {
        cnt[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_addr[k]);
        ring[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(man_addr + ring_off(ring_offs, k) * 16);
    }
    uint32_t mi[3] = {0, 0, 0};
    int32_t end_in[3] = {-1, -1, -1};
    int32_t end_out = -1;
    uint32_t own_t = 0, batch = 0, rows_w = 0, meta_w = 0;

    // hops this port gives own row t (0: not this port's)
    auto own_hops = [&](uint32_t t, uint32_t& m, uint32_t& order) -> uint32_t {
        const uint32_t f = route_flags(words, t);
        if (((f >> 1) & 1) != lnk) return 0;
        m = route_mask(words, t) & ~(1u << me);
        if (m == 0) return 0;
        order = f & 1;  // 0: X first
        const bool first_x = order == 0;
        uint32_t ab;
        if (axis_x == first_x) {  // my axis is the first leg
            ab = axis_x ? xspan(my_c, cols_of(m), t) : yspan(my_r, rows_of(m), t);
        } else {  // the turn at the origin into my axis
            ab = axis_x ? xspan(my_c, cols_in_row(m, my_r), t) : yspan(my_r, rows_in_col(m, my_c), t);
        }
        return plus ? (ab & 15) : (ab >> 4);
    };
    auto flush = [&]() {
        if (batch) {
            for (uint32_t i = batch; i < group; ++i) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w)[i * META_WORDS] = SKIP;
            }
            noc_async_read_barrier();
            cb_push_back(cb_rows, group);
            cb_push_back(cb_meta, group);
            batch = 0;
        }
    };
    auto slot_page = [&]() {
        if (batch == 0) {
            cb_reserve_back(cb_rows, group);
            rows_w = get_write_ptr(cb_rows);
            cb_reserve_back(cb_meta, group);
            meta_w = get_write_ptr(cb_meta);
        }
    };
    auto put = [&](uint32_t slot, uint32_t w0, uint32_t w1) {
        slot_page();
        noc_async_read(out.get_noc_addr(slot), rows_w + batch * row_bytes, row_bytes);
        volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w) + batch * META_WORDS;
        mp[0] = w0;
        mp[1] = w1;
        if (++batch == group) flush();
    };
    auto put_end = [&](uint32_t g) {
        slot_page();
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_w)[batch * META_WORDS] = END | (g << 20);
        if (++batch == group) flush();
    };
    auto inputs_reached = [&](int32_t g, uint32_t mask) {
        for (uint32_t k = 0; k < 3; ++k) {
            if (((mask >> k) & 1) && end_in[k] < g) return false;
        }
        return true;
    };
    const uint32_t order_mask = ring_mask & ~(ring_mask >> 3) & 7;  // the inputs END(g) waits on

    uint32_t idle = 0;
    bool bailed = false;
    while (true) {
        bool did = false;
        invalidate_l1_cache();
        for (uint32_t k = 0; k < 3; ++k) {
            if (((ring_mask >> k) & 1) && mi[k] < *cnt[k]) {
                const uint32_t e0 = ring[k][4 * mi[k]], e1 = ring[k][4 * mi[k] + 1];
                ++mi[k];
                if (e0 & END) {
                    end_in[k] = static_cast<int32_t>((e0 >> 20) & 0xF);
                } else {
                    put(e0 & 0xFFFF, e0, e1);
                }
                did = true;
            }
        }
        if (!did) {
            while (own_t < S) {
                const uint32_t t = own_t++;
                uint32_t m = 0, order = 0;
                const uint32_t h = own_hops(t, m, order);
                if (h) {
                    const uint32_t slot = me * S + t;
                    put(slot, slot | (h << 16) | (order << 24), m);
                    did = true;
                    break;
                }
            }
        }
        if (!did && own_t >= S) {
            const int32_t g = end_out + 1;
            if (g <= static_cast<int32_t>(GMAX) && (g == 0 || inputs_reached(g - 1, order_mask))) {
                put_end(g);
                end_out = g;
                did = true;
            } else if (g > static_cast<int32_t>(GMAX) && inputs_reached(GMAX, ring_mask)) {
                break;
            }
        }
        if (!did && batch) flush();
        idle = did ? 0 : idle + 1;
        if (spin_limit && idle > spin_limit) {
            bailed = true;
            break;
        }
    }
    flush();
    if (spin_limit) {  // debug: state of this reader
        cb_reserve_back(4, 1);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));
        invalidate_l1_cache();
        d[0] = 0xDB600000 | port_idx; d[1] = me; d[2] = dir; d[3] = lnk; d[4] = own_t; d[5] = end_out;
        d[6] = mi[0]; d[7] = mi[1]; d[8] = mi[2]; d[9] = *cnt[0]; d[10] = *cnt[1]; d[11] = *cnt[2];
        d[12] = end_in[0]; d[13] = end_in[1]; d[14] = end_in[2]; d[15] = bailed | (ring_mask << 4);
        const auto dbg = TensorAccessor(dbg_args, dbg_addr, 64);
        noc_async_write(get_write_ptr(4), dbg.get_noc_addr(port_idx), 64);
        noc_async_write_barrier();
        if (bailed) {  // release the sender
            for (uint32_t g = end_out + 1; g <= GMAX; ++g) put_end(g);
            flush();
            return;
        }
    }
    for (uint32_t k = 0; k < 3; ++k) {
        if (mi[k]) noc_semaphore_inc(get_noc_addr(cnt_addr[k]), 0u - mi[k]);
    }
    noc_async_atomic_barrier();
}
"""
)

_SENDER = (
    _COMMON
    + r"""
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
constexpr uint32_t group = get_compile_time_arg_val(4);
constexpr uint32_t MB = get_compile_time_arg_val(5);
constexpr uint32_t ROWS_PER_PKT = get_compile_time_arg_val(6);
constexpr auto out_args = TensorAccessorArgs<7>();

// Port sender: the fence with the neighbour, then every row one hop into the same slot of the neighbour's output,
// and for each the entries the neighbour's ports need (continue / turn + / turn -), batched per stream.
void kernel_main() {
    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t man_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ring_offs = get_arg_val<uint32_t>(a++);
    uint32_t cnt_addr[3];
    for (uint32_t k = 0; k < 3; ++k) cnt_addr[k] = get_arg_val<uint32_t>(a++);
    uint32_t s_on[3], s_x[3], s_y[3], s_k[3];
    for (uint32_t s = 0; s < 3; ++s) {
        s_on[s] = get_arg_val<uint32_t>(a++);
        s_x[s] = get_arg_val<uint32_t>(a++);
        s_y[s] = get_arg_val<uint32_t>(a++);
        s_k[s] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t nb_r = get_arg_val<uint32_t>(a++);  // the neighbour's coordinates
    const uint32_t nb_c = get_arg_val<uint32_t>(a++);
    const uint32_t dir = get_arg_val<uint32_t>(a++);
    const bool axis_x = dir < 2;
    const auto out = TensorAccessor(out_args, out_addr, row_bytes);

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[group];
    for (uint32_t h = 0; h < group; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdrs[h], dst_chip_id, dst_mesh_id);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* mhdr = PacketHeaderPool::allocate_header();
    (void)fabric_set_unicast_route(mhdr, dst_chip_id, dst_mesh_id);
    conn.open();
    // fence: tell the neighbour's opposite port this chip has started; wait for the same from it
    hdrs[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
    {
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        noc_semaphore_wait_min(ready, 1);
        noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
        noc_async_atomic_barrier();
    }

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage_l1 = get_write_ptr(cb_stage);  // 3 streams x 2 halves x MB entries of 16 bytes
    uint32_t half[3] = {0, 0, 0}, staged[3] = {0, 0, 0}, sent[3] = {0, 0, 0};
    auto send_manifest = [&](uint32_t s) {
        if (staged[s] == 0) return;
        const uint32_t src = stage_l1 + (s * 2 + half[s]) * MB * 16;
        const uint64_t dst = get_noc_addr(s_x[s], s_y[s], man_addr + (ring_off(ring_offs, s_k[s]) + sent[s]) * 16);
        const uint64_t cnt_noc = get_noc_addr(s_x[s], s_y[s], cnt_addr[s_k[s]]);
        mhdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, cnt_noc, staged[s], true},
                                                     staged[s] * 16);
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src, staged[s] * 16, reinterpret_cast<uint32_t>(mhdr));
        noc_async_writes_flushed();
        sent[s] += staged[s];
        staged[s] = 0;
        half[s] ^= 1;
    };
    auto stage = [&](uint32_t s, uint32_t w0, uint32_t w1) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_l1 + (s * 2 + half[s]) * MB * 16) + 4 * staged[s];
        p[0] = w0;
        p[1] = w1;
        if (++staged[s] == MB) send_manifest(s);
    };
    // what the neighbour does with a row this port just sent it
    auto follow = [&](uint32_t w0, uint32_t m) {
        const uint32_t slot = w0 & 0xFFFF, h = (w0 >> 16) & 0xF, g = (w0 >> 20) & 0xF, order = (w0 >> 24) & 1;
        const uint32_t nxt = (g + 1) << 20 | (order << 24) | slot;
        if (h > 1) stage(0, nxt | ((h - 1) << 16), m);
        if (axis_x == (order == 0)) {  // first leg: the neighbour turns into the other axis where its line needs it
            const uint32_t t = slot % S;
            const uint32_t ab = axis_x ? yspan(nb_r, rows_in_col(m, nb_c), t) : xspan(nb_c, cols_in_row(m, nb_r), t);
            if (ab & 15) stage(1, nxt | ((ab & 15) << 16), m);
            if (ab >> 4) stage(2, nxt | ((ab >> 4) << 16), m);
        }
    };

    uint32_t ends = 0;
    bool done = false;
#ifdef SA_ZGATE_CTR
    // (the zero gate: the call's scratch zeroing (other kernels of this program) counts in here, SA_ZGATE_N a call -
    // the counter only grows, my call count after it; no END leaves before it: a chip that has not zeroed holds every
    // other chip's dispatch from completing, so no combine writes into its L1 before)
    volatile tt_l1_ptr uint32_t* zep = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SA_ZGATE_EP);
    const uint32_t zneed = (*zep + 1) * SA_ZGATE_N;
    *zep = *zep + 1;
    bool zok = false;
#endif
    while (!done) {
        if (!cb_pages_available_at_front(cb_meta, group)) {
            for (uint32_t s = 0; s < 3; ++s) send_manifest(s);  // about to wait: nothing may sit staged
        }
        cb_wait_front(cb_meta, group);
        cb_wait_front(cb_rows, group);
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
        const uint32_t rows_l1 = get_read_ptr(cb_rows);
        for (uint32_t i = 0; i < group; ++i) {
            const uint32_t w0 = meta[i * META_WORDS], m = meta[i * META_WORDS + 1];
            if (w0 & END) {
#ifdef SA_ZGATE_CTR
                if (!zok) {
                    volatile tt_l1_ptr uint32_t* zc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SA_ZGATE_CTR);
                    do {
                        invalidate_l1_cache();
                    } while (*zc < zneed);
                    zok = true;
                }
#endif
                const bool last = ((w0 >> 20) & 0xF) == GMAX;
                for (uint32_t s = 0; s < 3; ++s) {  // (s_on 2: a terminal stream - END(GMAX) only)
                    if (s_on[s] == 1 || (s_on[s] == 2 && last)) stage(s, w0, 0);
                }
                if (last) done = true;
                ++ends;
                continue;
            }
            if (w0 & SKIP) continue;
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[i];
            if constexpr (ROWS_PER_PKT == 2) {
                const uint32_t w0b = i + 1 < group ? meta[(i + 1) * META_WORDS] : END;
                if ((w0b & (END | SKIP)) == 0) {
                    const uint32_t mb = meta[(i + 1) * META_WORDS + 1];
                    hdr->to_noc_unicast_scatter_write(
                        NocUnicastScatterCommandHeader(
                            {out.get_noc_addr(w0 & 0xFFFF, 0, 0), out.get_noc_addr(w0b & 0xFFFF, 0, 0)},
                            {static_cast<uint16_t>(row_bytes)}),
                        2 * row_bytes);
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(rows_l1 + i * row_bytes, 2 * row_bytes, reinterpret_cast<uint32_t>(hdr));
                    follow(w0, m);
                    follow(w0b, mb);
                    ++i;
                    continue;
                }
            }
            hdr->to_noc_unicast_write(NocUnicastCommandHeader{out.get_noc_addr(w0 & 0xFFFF, 0, 0)}, row_bytes);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(rows_l1 + i * row_bytes, row_bytes, reinterpret_cast<uint32_t>(hdr));
            follow(w0, m);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_rows, group);
        cb_pop_front(cb_meta, group);
    }
    for (uint32_t s = 0; s < 3; ++s) send_manifest(s);
    conn.close();
}
"""
)


def _axis_dirs(d):
    """The other axis's directions (+, -) of direction d."""
    return ("Y+", "Y-") if d[0] == "X" else ("X+", "X-")


def _wiring(chips, nb, coord, d):
    """Port (coord, d)'s inputs and output streams: (ring_mask, streams). ring_mask bit k: input ring k exists (0: rows
    continuing in d from the chip behind, its port d; 1 / 2: rows turning into d's axis that arrived moving + / - along
    the other axis, from the chips on either side, their ports + / -); bit 3 + k: ring k is a terminal stream.
    streams: per output stream s (0: continue, 1 / 2: turn + / -) (on, target port's direction at the neighbour, ring);
    on 2 = terminal. A link into a chip that has no port in d nor on the other axis (a 1-D line's end) would deliver no
    END there, and that chip's dispatch could finish (and its experts read x) before the link's rows land: it carries
    a terminal stream instead - END(GMAX) only, into ring 0 of the neighbour's port OPP[d] (the port of the same link;
    its ring 0 is free: no chip is behind it)."""
    pa, ma = _axis_dirs(d)
    behind = nb(coord, OPP[d])
    ring_mask = 0
    if behind is not None and d in chips[behind]:
        ring_mask |= 1
    elif d in chips[coord] and not any(x in chips[coord] for x in (OPP[d], pa, ma)):
        ring_mask |= 1 | 8  # (the link from the port d's neighbour ends here)
    src_plus = nb(coord, ma)  # a row moving + arrives from the - side
    if src_plus is not None and pa in chips[src_plus]:
        ring_mask |= 2
    src_minus = nb(coord, pa)
    if src_minus is not None and ma in chips[src_minus]:
        ring_mask |= 4
    peer = chips[coord][d]["peer"]
    k_turn = 1 if d[1] == "+" else 2  # (ring 1 at the neighbour if this port moves rows +, else ring 2)
    streams = [(1 if t in chips[peer] else 0, t, k) for t, k in ((d, 0), (pa, k_turn), (ma, k_turn))]
    if not any(on for on, _, _ in streams):
        streams[0] = (2, OPP[d], 0)
    return ring_mask, streams


def route_words(N):
    """Route words per own row: 1 up to 16 chips, 2 beyond (the 32-bit mask, then order | link << 1)."""
    return 1 if N <= 16 else 2


_PLAN_CACHE = {}  # mesh_plan / manifest_spec by (mesh, its shape, options); the mesh kept alive so its id stays its own


def mesh_plan(mesh_device, num_links, torus=(False, False)):
    """Ports of every chip: {coord: {dir: dict(peer, links)}}; torus = (wrap X, wrap Y) where cabled. Memoized per mesh
    and options (the fabric's links do not change; ~120 us of host a call otherwise)."""
    key = ("plan", id(mesh_device), tuple(mesh_device.shape), num_links, tuple(torus))
    hit = _PLAN_CACHE.get(key)
    if hit is not None and hit[0] is mesh_device:
        return hit[1]
    out = _mesh_plan(mesh_device, num_links, torus)
    _PLAN_CACHE[key] = (mesh_device, out)
    return out


def _mesh_plan(mesh_device, num_links, torus):
    R, C = tuple(mesh_device.shape)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    wrap_x = torus[0] and C > 2
    wrap_y = torus[1] and R > 2

    def nb(coord, d):
        r, c = coord
        if d[0] == "X":
            c2 = c + (1 if d[1] == "+" else -1)
            if wrap_x:
                c2 %= C
            return (r, c2) if 0 <= c2 < C else None
        r2 = r + (1 if d[1] == "+" else -1)
        if wrap_y:
            r2 %= R
        return (r2, c) if 0 <= r2 < R else None

    chips = {}
    for r in range(R):
        for c in range(C):
            ports = {}
            for d in DIRS:
                peer = nb((r, c), d)
                if peer is None:
                    continue
                if not ttnn.get_neighbor_eth_directions(mesh_device, node((r, c)), node(peer)):
                    raise ValueError(f"sparse_all_gather_2d: {(r, c)} -> {peer} is not a direct link")
                links = ttnn.get_forwarding_link_indices(node((r, c)), node(peer))
                if len(links) < num_links:
                    raise ValueError(f"sparse_all_gather_2d: {(r, c)} -> {peer} has {len(links)} links")
                ports[d] = dict(peer=peer, links=links[:num_links])
            chips[(r, c)] = ports
    return chips, nb, (wrap_x, wrap_y)


_PORTS_CACHE = {}
_DESC_CACHE = {}  # sparse_all_gather_2d's descriptors by buffers and options


def place_ports(mesh_device, chips, allowed=None):
    """{(coord, dir, link index): logical core}: one port core per direction and link, next to its Ethernet core
    (from `allowed`, logical (x, y) cells, if given: e.g. the cells another op leaves free). Memoized per mesh, chips
    and allowed cells (a per-call op's host time: ~8 ms a placement)."""
    key = (
        id(mesh_device),
        repr(
            sorted(
                (c, sorted((d, p["peer"], tuple(p["links"])) for d, p in ports.items())) for c, ports in chips.items()
            )
        ),
        None if allowed is None else tuple(sorted(allowed)),
        os.environ.get("SPC_PLACE", "noc1"),
    )
    hit = _PORTS_CACHE.get(key)
    if hit is not None and hit[0] is mesh_device:
        return dict(hit[1])
    out = _place_ports(mesh_device, chips, allowed)
    _PORTS_CACHE[key] = (mesh_device, dict(out))
    return out


def _place_ports(mesh_device, chips, allowed=None):
    allowed = allowed_cores(mesh_device) if allowed is None else set(allowed)
    connections = {
        coord: [(p["peer"], l) for p in ports.values() for l in p["links"]] for coord, ports in chips.items()
    }
    eth = probe_ethernet_cores(mesh_device, connections, allowed_cores(mesh_device))  # (a setup-time probe)
    port_core = {}
    mode = os.environ.get("SPC_PLACE", "noc1")
    if mode == "auto":  # (probe: the nearest-port placement when it leaves >= 8 cells of `allowed` to the accumulators, else
        # uniform on NoC0 - the per-chip placement's union of port cells costs the accumulators' cells where cells are few)
        os.environ["SPC_PLACE"] = "noc1"
        try:
            near = _place_ports(mesh_device, chips, allowed)
        finally:
            os.environ["SPC_PLACE"] = "auto"
        left = len(set(allowed) - {(c.x, c.y) for c in near.values()})
        return near if left >= 8 else _place_ports_uniform(mesh_device, chips, allowed, eth, noc=0)
    if mode in ("uni0", "uni1"):  # (probe: one logical cell per (direction, link) on every chip - the fewest NoC hops summed
        # over the chips, NoC0 / NoC1 - so the ports take 8 cells of the shared cell list, not their union over chips)
        return _place_ports_uniform(mesh_device, chips, allowed, eth, noc=int(mode[-1]))
    for coord, ports in chips.items():
        taken = set()
        for d, p in ports.items():
            for li, l in enumerate(p["links"]):
                exy = eth_noc_coord(mesh_device, coord, eth[(coord, p["peer"], l)])
                # (probe SPC_PLACE=noc0: nearest on NoC0, for the sender on NoC0 - SPC_SNOC=0)
                near = _worker_nearest_noc0 if os.environ.get("SPC_PLACE", "noc1") == "noc0" else _worker_nearest_noc1
                core = near(mesh_device, coord, exy, taken, allowed)
                if core is None:  # (no Ethernet core reported - the mock cluster runs no probe: the first free core)
                    x, y = min((xy for xy in allowed if xy not in taken), key=lambda xy: (xy[1], xy[0]))
                    core = ttnn.CoreCoord(x, y)
                taken.add((core.x, core.y))
                port_core[(coord, d, li)] = core
    return port_core


def _place_ports_uniform(mesh_device, chips, allowed, eth, noc):
    from .fabric_all_gather_py import _chip_maps, _noc_grid

    gx, gy = _noc_grid(mesh_device)
    slots = sorted({(d, li) for ports in chips.values() for d, p in ports.items() for li in range(len(p["links"]))})
    phys = {}  # (coord, logical cell) -> physical NoC (x, y)
    for coord in chips:
        t2p = _chip_maps(mesh_device, coord)[2]
        for x, y in allowed:
            v = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            phys[(coord, (x, y))] = (t2p.get(v.x, v.x), v.y)

    def hops(coord, cell, exy):
        px, py = phys[(coord, cell)]
        ex, ey = exy
        if noc == 0:  # (+X then +Y, wrapping)
            return (ex - px) % gx + (ey - py) % gy
        return (py - ey) % gy + (px - ex) % gx  # (NoC1: -Y then -X, wrapping)

    cost = {}
    for d, li in slots:
        for cell in allowed:
            t = 0
            for coord, ports in chips.items():
                if d not in ports or li >= len(ports[d]["links"]):
                    continue
                exy = eth_noc_coord(mesh_device, coord, eth[(coord, ports[d]["peer"], ports[d]["links"][li])])
                t += hops(coord, cell, exy)
            cost[((d, li), cell)] = t
    taken, pick = set(), {}
    left = set(slots)
    while left:  # (greedy: the slot whose best free cell is cheapest first)
        best = min(
            ((cost[(sl, cell)], sl, cell) for sl in left for cell in allowed if cell not in taken),
            key=lambda e: (e[0], e[2][1], e[2][0]),
        )
        _, sl, cell = best
        pick[sl] = cell
        taken.add(cell)
        left.discard(sl)
    return {
        (coord, d, li): ttnn.CoreCoord(*pick[(d, li)])
        for coord, ports in chips.items()
        for d, p in ports.items()
        for li in range(len(p["links"]))
    }


def ensure_semaphores(mesh_device, *, num_links=2, torus=(False, False)):
    """Create the op's global semaphores now (persistent L1: create them before any per-call L1 allocation, or they
    land under it and fragment the bank once it is freed)."""
    chips, _, _ = mesh_plan(mesh_device, num_links, torus)
    cores = sorted({(c.x, c.y) for c in place_ports(mesh_device, chips).values()})
    key = (id(mesh_device), tuple(cores), "sparse_2d_sems")
    if key not in _SEM_CACHE:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
        _SEM_CACHE[key] = (
            mesh_device,
            tuple(ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(4)),
            None,
        )
    return _SEM_CACHE[key][1]


@functools.lru_cache(maxsize=None)  # (pure in its integer arguments: per-call host time otherwise)
def axis_reach(n, wrap):
    """Per position: the most hops a tree's span goes forward / backward from it (over every mask and tie)."""
    t = span_table(n, wrap)
    per = 2 << n
    return [max(e & 15 for e in t[p * per : (p + 1) * per]) for p in range(n)], [
        max(e >> 4 for e in t[p * per : (p + 1) * per]) for p in range(n)
    ]


@functools.lru_cache(maxsize=None)  # (pure in its integer arguments: per-call host time otherwise)
def manifest_layout(R, C, wrap_x, wrap_y, S):
    """(ring 1's offset, ring 2's offset, total) in 16-byte entries of a port core's manifest. A row enters a given
    ring of a given port at most once, so ring k needs S entries per origin chip whose trees can put a row there (max
    over the ports) plus the END markers. Ring 0 of port (chip, d): rows that started their leg along d's axis at an
    entry chip k hops back with a span of at least k + 1 hops; their origins are the entry chip and, for the second
    leg, the chips of its line on the other axis that reach it. Rings 1 / 2: first-leg rows of the other axis that
    arrived moving + / -: one origin per chip of that line within reach."""
    fx, bx = axis_reach(C, wrap_x)
    fy, by = axis_reach(R, wrap_y)

    def hops(n, wrap, p, q, fwd):  # hops from p to q moving + (fwd) or -, None if that way does not lead there
        k = (q - p if fwd else p - q) % n if wrap else (q - p if fwd else p - q)
        return k if k > 0 else None

    def reaches(n, wrap, f, b, p, q):
        kf, kb = hops(n, wrap, p, q, True), hops(n, wrap, p, q, False)
        return (kf is not None and kf <= f[p]) or (kb is not None and kb <= b[p])

    axes = {"X": (C, wrap_x, fx, bx, R, wrap_y, fy, by), "Y": (R, wrap_y, fy, by, C, wrap_x, fx, bx)}
    n0 = n1 = n2 = 0
    for r in range(R):
        for c in range(C):
            for ax, (n, w, f, b, no, wo, fo, bo) in axes.items():
                pos, opos = (c, r) if ax == "X" else (r, c)
                for fwd in (True, False):
                    span = f if fwd else b
                    cont = 0
                    for e in range(n):
                        k = hops(n, w, e, pos, fwd)
                        if k is not None and k + 1 <= span[e]:
                            cont += 1 + sum(reaches(no, wo, fo, bo, q, opos) for q in range(no) if q != opos)
                    n0 = max(n0, cont)
                for fwd, k_ in ((True, 1), (False, 2)):
                    span = fo if fwd else bo
                    cnt = sum(1 for q in range(no) if (k := hops(no, wo, q, opos, fwd)) is not None and k <= span[q])
                    if k_ == 1:
                        n1 = max(n1, cnt)
                    else:
                        n2 = max(n2, cnt)
    GMAX = (C - 1) + (R - 1) - 1
    e0, e1, e2 = (n * S + GMAX + 4 for n in (n0, n1, n2))
    assert e0 + e1 < (1 << 16), "manifest ring offsets are 16-bit entry counts"
    return e0, e0 + e1, e0 + e1 + e2


def manifest_spec(mesh_device, S, *, num_links=2, torus=(False, False)):
    """(shape, dtype, layout, device, memory config) of the port cores' manifest rings for S own rows per chip:
    allocate it per call (ttnn.allocate_tensor_on_device) and pass it as ``manifest`` (its contents need no zeroing:
    the rings are consumed by count, and the counters are global semaphores the reader resets), instead of the
    persistent one (~123 KB of every core's L1 at K3's 2560 rows). Memoized per mesh, S and options."""
    key = ("manifest", id(mesh_device), tuple(mesh_device.shape), S, num_links, tuple(torus))
    hit = _PLAN_CACHE.get(key)
    if hit is not None and hit[0] is mesh_device:
        return hit[1]
    R, C = tuple(mesh_device.shape)
    chips, _, (wrap_x, wrap_y) = mesh_plan(mesh_device, num_links, torus)
    cores = sorted({(c.x, c.y) for c in place_ports(mesh_device, chips).values()})
    man_words = manifest_layout(R, C, wrap_x, wrap_y, S)[2] * 4
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [1, man_words], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out = (ttnn.Shape([1, 1, len(cores), man_words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, mc)
    _PLAN_CACHE[key] = (mesh_device, out)
    return out


def static_cb_bytes(N, S, row_width, *, group=8, manifest_batch=16, dtype_bytes=2):
    """L1 the port cores' circular buffers take (sparse_all_gather_2d's descriptor, no extra op): N chips, S own rows,
    rows row_width wide. Each CB rounded to 64 B; for sizing the L1 below a buffer that stays allocated over the call.
    """
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    row_bytes = al(row_width * dtype_bytes)
    plan_bytes = al(S * route_words(N) * 4)
    cbs = (2 * group * row_bytes, 2 * group * META_PAGE, plan_bytes, 3 * 2 * manifest_batch * 16, 64)
    return sum(al(v) for v in cbs)


def sparse_all_gather_2d(
    output,
    plan_tensor,
    *,
    manifest=None,
    num_links=2,
    torus=(False, False),
    group=8,
    manifest_batch=16,
    pack=None,
    debug=None,
    spin_limit=0,
    extra=None,
    zero=None,
):
    """output: [1, 1, N * S, W] ROW_MAJOR, DRAM interleaved, replicated layout per chip (each chip's own rows already at
    slots idx * S + t, idx = r * C + c); plan_tensor: [1, 1, 1, S] uint32 per chip, the route words of its own rows.
    Fills every destination's slots; returns output. extra = (tensors, fn((r, c)) -> (cbs, kernels)): another op's
    CBs / kernels appended to each chip's program, on cores off port_cores_all(..) (it runs under the dispatch)."""
    # (the same buffers, options and no merged op: the descriptor of the last such call - its runtime args are these
    # buffers' addresses; ~2 ms of host per call otherwise. Each buffer by its address and what the descriptor compiled
    # from it - shape, dtype, layout, page size, memory config (sharding: the accessors' compile-time args): a new
    # buffer at a freed one's address with another width must not reuse the old row bytes / CB sizes / accessors)
    dkey = None
    if extra is None:
        a_ = lambda t: (  # noqa: E731
            None
            if t is None
            else (
                int(t.buffer_address()),
                tuple(t.shape),
                t.dtype,
                t.layout,
                int(t.buffer_aligned_page_size()),
                t.memory_config(),
            )
        )
        dkey = (
            id(output.device()),
            a_(output),
            a_(plan_tensor),
            a_(manifest),
            a_(debug),
            num_links,
            tuple(torus),
            group,
            manifest_batch,
            pack,
            spin_limit,
            ttnn.get_tt_fabric_max_payload_size_bytes(),
            None if zero is None else zero["key"],
        )
        hit = _DESC_CACHE.get(dkey)
        if hit is not None and hit[0] is output.device():
            # (prepared dispatch: only the io tensors' address slots patched per call, not every runtime arg)
            from ttnn.bringup.prepared_program import generic_op as prepared_generic_op

            return prepared_generic_op([plan_tensor, output], hit[1])
    mesh_device = output.device()
    R, C = tuple(mesh_device.shape)
    N = R * C
    RW = route_words(N)
    S = int(plan_tensor.shape[-1]) // RW
    W = int(output.shape[-1])
    assert int(output.shape[-2]) == N * S and N * S < (1 << 16) and N <= 32
    row_bytes = int(output.buffer_aligned_page_size())
    plan_bytes = int(plan_tensor.buffer_aligned_page_size())
    if pack is None:
        pack = 2 if 2 * row_bytes <= ttnn.get_tt_fabric_max_payload_size_bytes() else 1
    chips, nb, (wrap_x, wrap_y) = mesh_plan(mesh_device, num_links, torus)
    GMAX = (C - 1) + (R - 1) - 1
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    port_core = place_ports(mesh_device, chips)
    all_cores = sorted({(c.x, c.y) for c in port_core.values()})
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in all_cores])
    off1, off2, man_entries = manifest_layout(R, C, wrap_x, wrap_y, S)
    ring_offs = off1 | off2 << 16
    man_words = man_entries * 4
    key = (id(mesh_device), tuple(all_cores), "sparse_2d", man_words if manifest is None else "own")
    sems = ensure_semaphores(mesh_device, num_links=num_links, torus=torus)
    if key not in _SEM_CACHE and manifest is not None:
        _SEM_CACHE[key] = (mesh_device, sems, None)
    if key not in _SEM_CACHE:
        shard = ttnn.ShardSpec(core_set, [1, man_words], ttnn.ShardOrientation.ROW_MAJOR)
        man_t = ttnn.from_torch(
            torch.zeros(1, 1, len(all_cores), man_words, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.synchronize_device(mesh_device)
        _SEM_CACHE[key] = (mesh_device, sems, man_t)
    _, sems, man_t = _SEM_CACHE[key]
    cnt_addrs = [int(ttnn.get_global_semaphore_address(x)) for x in sems[:3]]
    ready_addr = int(ttnn.get_global_semaphore_address(sems[3]))
    if manifest is not None:
        sh = manifest.memory_config().shard_spec
        assert int(sh.shape[1]) >= man_words and sh.grid == core_set, "manifest: manifest_spec(...) of this call"
        man_t = manifest
    man_addr = int(man_t.buffer_address())

    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    plan_ct = list(ttnn.TensorAccessorArgs(plan_tensor).get_compile_time_args())
    out_addr, plan_addr = int(output.buffer_address()), int(plan_tensor.buffer_address())
    if debug is None:
        debug = plan_tensor
    dbg_ct = list(ttnn.TensorAccessorArgs(debug).get_compile_time_args())
    dbg_addr = int(debug.buffer_address())
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    common = {
        "__R__": str(R),
        "__C__": str(C),
        "__S__": str(S),
        "__GMAX__": str(GMAX),
        "__RW__": str(RW),
        **span_subs(R, C, wrap_x, wrap_y),
    }
    reader_src, sender_src = _READER, _SENDER
    for k, v in common.items():
        reader_src = reader_src.replace(k, v)
        sender_src = sender_src.replace(k, v)
    if zero is not None:  # (the zero gate's counter / call-count words: persistent, on every port core)
        sender_src = (
            f"#define SA_ZGATE_CTR {zero['ctr']}u\n#define SA_ZGATE_EP {zero['ep']}u\n#define SA_ZGATE_N {zero['n']}u\n"
            + sender_src
        )
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ports in chips.items():
        r, c = coord
        program = ttnn.ProgramDescriptor()
        my_ports = [(d, li, port_core[(coord, d, li)]) for d, p in ports.items() for li in range(len(p["links"]))]
        port_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for _, _, core in my_ports])
        program.cbs = [
            ttnn.CBDescriptor(
                total_size=2 * group * row_bytes,
                core_ranges=port_set,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_ROWS, output.dtype, row_bytes)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * group * META_PAGE,
                core_ranges=port_set,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_META, ttnn.uint32, META_PAGE)],
            ),
            ttnn.CBDescriptor(
                total_size=plan_bytes,
                core_ranges=port_set,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_PLAN, ttnn.uint32, plan_bytes)],
            ),
            ttnn.CBDescriptor(
                total_size=3 * 2 * manifest_batch * 16,
                core_ranges=port_set,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_STAGE, ttnn.uint32, 3 * 2 * manifest_batch * 16)],
            ),
            ttnn.CBDescriptor(
                total_size=64,
                core_ranges=port_set,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_DBG, ttnn.uint32, 64)],
            ),
        ]
        reader_rt, sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for d, li, core in my_ports:
            ring_mask, wiring = _wiring(chips, nb, coord, d)
            port_idx = (r * C + c) * 8 + DIRS.index(d) * 2 + li
            reader_rt[core.x][core.y] = (
                [out_addr, plan_addr, man_addr, ring_offs]
                + cnt_addrs
                + [ring_mask, r, c, DIRS.index(d), li, dbg_addr, port_idx, spin_limit]
            )
            peer = ports[d]["peer"]
            rc = virt(port_core[(peer, OPP[d], li)])
            pn = node(peer)
            streams = []
            for on, target, k in wiring:
                v = virt(port_core[(peer, target, li)]) if on else None
                streams += [on, v.x, v.y, k] if on else [0, 0, 0, 0]
            args = [out_addr, ready_addr, rc.x, rc.y, int(pn.mesh_id), int(pn.chip_id), man_addr, ring_offs]
            args += cnt_addrs + streams + [peer[0], peer[1], DIRS.index(d)]
            args += list(ttnn.setup_fabric_connection(node(coord), pn, ports[d]["links"][li], program, core))
            sender_rt[core.x][core.y] = args
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=reader_src,
                source_type=src,
                core_ranges=port_set,
                compile_time_args=[CB_ROWS, CB_META, CB_PLAN, row_bytes, group, plan_bytes] + out_ct + plan_ct + dbg_ct,
                runtime_args=reader_rt,
                config=dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=sender_src,
                source_type=src,
                core_ranges=port_set,
                compile_time_args=[CB_ROWS, CB_META, CB_STAGE, row_bytes, group, manifest_batch, pack] + out_ct,
                runtime_args=sender_rt,
                config=dm("RISCV_0", NOC1),
            ),
        ]
        if extra is not None:
            cbs_x, kernels_x = extra[1](coord)
            program.cbs = list(program.cbs) + list(cbs_x)
            program.kernels = list(program.kernels) + list(kernels_x)
        if zero is not None:
            program.kernels = list(program.kernels) + list(zero["kernels"](coord))
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = program
    if dkey is not None:
        if len(_DESC_CACHE) >= 256:  # (bounded: callers with per-call buffers at ever new addresses)
            _DESC_CACHE.pop(next(iter(_DESC_CACHE)))
        ttnn.freeze_program_hashes(mesh_desc)  # (hash stored once: not re-hashed per call)
        _DESC_CACHE[dkey] = (mesh_device, mesh_desc)
    return ttnn.generic_op(([] if extra is None else list(extra[0])) + [plan_tensor, output], mesh_desc)


def port_cores_all(mesh_device, *, num_links=2, torus=(False, False)):
    """Every chip's port cores (logical CoreCoords, the union): the cores an extra op must stay off."""
    chips, _, _ = mesh_plan(mesh_device, num_links, torus)
    return sorted({(c.x, c.y) for c in place_ports(mesh_device, chips).values()})


# ---- the dispatch as a phase of another op's port kernels (the MoE exchange: the combine's ports) ----
# The reader's and the sender's bodies become functions in namespace sa (sa::read_phase(a0), sa::send_phase(a0, conn))
# that the host kernel calls first: their runtime args at [a0, a0 + n), their compile-time args from ct0, their CBs at
# the given indices; the sender phase sends on the host kernel's open fabric connection (one sender a link). After the
# reader phase every row sent to this chip has landed (each incoming link delivers END(GMAX) to one of its ports after
# its rows): the reader then counts in on each x relay's ready word (+1, the first port also the pad to XREADY_N).
XREADY_N = 8  # (ports a chip at most: 4 directions x 2 links - each chip's count is padded to it)


def _phase_src(src, name, ct0, cbs, fabric):
    """A kernel source as a phase function in namespace sa (see above)."""
    import re

    head, body = src.split("void kernel_main() {", 1)
    inc = "\n".join(l for l in head.splitlines() if l.startswith("#include")) + "\n"
    rest = "\n".join(l for l in head.splitlines() if not l.startswith("#include"))
    rest = re.sub(
        r"get_compile_time_arg_val\((\d+)\)", lambda m: f"get_compile_time_arg_val({int(m.group(1)) + ct0})", rest
    )
    rest = re.sub(r"TensorAccessorArgs<(\d+)>", lambda m: f"TensorAccessorArgs<{int(m.group(1)) + ct0}>", rest)
    rest = rest.replace("using namespace tt::tt_fabric;", "")
    if fabric:
        body = body.replace(
            "    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);\n", ""
        )
        body = body.replace("    conn.open();\n", "").replace("    conn.close();\n", "")
        sig = f"template <typename CONN>\nvoid {name}(size_t a, CONN& conn) {{"
    else:
        sig = f"void {name}(size_t a) {{"
    body = body.replace("    size_t a = 0;\n", "", 1)
    body = body.replace("cb_reserve_back(4, 1)", f"cb_reserve_back({cbs['dbg']}, 1)").replace(
        "get_write_ptr(4)", f"get_write_ptr({cbs['dbg']})"
    )
    pre = "using namespace tt::tt_fabric;\n" if fabric else ""
    return inc, pre + "namespace sa {\n" + rest + "\n" + sig + body + "\n}  // namespace sa\n"


def dispatch_phase(
    mesh_device,
    output,
    plan_tensor,
    manifest,
    port_core,
    sems,
    *,
    rd_ct0,
    sd_ct0,
    cbs,
    relays,
    words_addr,
    ready_off,
    num_links=2,
    torus=(False, False),
    group=8,
    manifest_batch=16,
    pack=None,
):
    """The dispatch's pieces for the host op's port kernels (see above), its ports at port_core ({(coord, dir, link):
    logical core}, as place_ports): per chip coord dict(cbs, rd={(x, y): args}, sd=...), and the sources:
    (reader includes, reader phase, sender includes, sender phase, reader CT, sender CT). sems: the four global
    semaphores on those cores (the rings' three counters, the fence's ready word); cbs: dict(rows, meta, plan, stage, dbg)
    CB indices; relays: logical cores whose ready word (words_addr + ready_off) the readers count in on."""
    R, C = tuple(mesh_device.shape)
    N = R * C
    RW = route_words(N)
    S = int(plan_tensor.shape[-1]) // RW
    assert int(output.shape[-2]) == N * S and N * S < (1 << 16) and N <= 32
    row_bytes = int(output.buffer_aligned_page_size())
    plan_bytes = int(plan_tensor.buffer_aligned_page_size())
    if pack is None:
        pack = 2 if 2 * row_bytes <= ttnn.get_tt_fabric_max_payload_size_bytes() else 1
    chips, nb, (wrap_x, wrap_y) = mesh_plan(mesh_device, num_links, torus)
    GMAX = (C - 1) + (R - 1) - 1
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))  # noqa: E731
    off1, off2, man_entries = manifest_layout(R, C, wrap_x, wrap_y, S)
    assert int(manifest.memory_config().shard_spec.shape[1]) >= man_entries * 4, "manifest: too small"
    ring_offs = off1 | off2 << 16
    cnt_addrs = [int(ttnn.get_global_semaphore_address(x)) for x in sems[:3]]
    ready_addr = int(ttnn.get_global_semaphore_address(sems[3]))
    man_addr, out_addr, plan_addr = (
        int(manifest.buffer_address()),
        int(output.buffer_address()),
        int(plan_tensor.buffer_address()),
    )
    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    plan_ct = list(ttnn.TensorAccessorArgs(plan_tensor).get_compile_time_args())
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)  # noqa: E731
    subs = {"__R__": str(R), "__C__": str(C), "__S__": str(S), "__GMAX__": str(GMAX), "__RW__": str(RW),
            **span_subs(R, C, wrap_x, wrap_y)}  # fmt: skip
    rsrc, ssrc = _READER, _SENDER
    for k, v in subs.items():
        rsrc, ssrc = rsrc.replace(k, v), ssrc.replace(k, v)
    # (the reader phase ends counting in on the relays: after its last ring was consumed and re-armed)
    rsrc = rsrc.replace(
        "    noc_async_atomic_barrier();\n}\n",
        "    noc_async_atomic_barrier();\n"
        "    {  // every row this port's inputs brought has landed: count in on each x relay's ready word\n"
        "        const uint32_t nrel = get_arg_val<uint32_t>(a++), inc = get_arg_val<uint32_t>(a++);\n"
        "        const uint32_t rw = get_arg_val<uint32_t>(a++);\n"
        "        for (uint32_t i = 0; i < nrel; ++i, a += 2)\n"
        "            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1), rw), inc);\n"
        "        noc_async_atomic_barrier();\n"
        "    }\n}\n",
    )
    rinc, rph = _phase_src(rsrc, "read_phase", rd_ct0, cbs, False)
    sinc, sph = _phase_src(ssrc, "send_phase", sd_ct0, cbs, True)
    rd_ct = [cbs["rows"], cbs["meta"], cbs["plan"], row_bytes, group, plan_bytes] + out_ct + plan_ct + plan_ct
    sd_ct = [cbs["rows"], cbs["meta"], cbs["stage"], row_bytes, group, manifest_batch, pack] + out_ct

    rel = [virt(c) for c in relays]
    out = {}
    for coord, ports in chips.items():
        r, c = coord
        my_ports = [(d, li, port_core[(coord, d, li)]) for d, p in ports.items() for li in range(len(p["links"]))]
        assert len(my_ports) <= XREADY_N
        port_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for _, _, core in my_ports])
        mk = lambda i, size, dt, page: ttnn.CBDescriptor(  # noqa: E731
            total_size=size, core_ranges=port_set, format_descriptors=[ttnn.CBFormatDescriptor(i, dt, page)]
        )
        pieces = dict(
            cbs=[
                mk(cbs["rows"], 2 * group * row_bytes, output.dtype, row_bytes),
                mk(cbs["meta"], 2 * group * META_PAGE, ttnn.uint32, META_PAGE),
                mk(cbs["plan"], plan_bytes, ttnn.uint32, plan_bytes),
                mk(cbs["stage"], 3 * 2 * manifest_batch * 16, ttnn.uint32, 3 * 2 * manifest_batch * 16),
                mk(cbs["dbg"], 64, ttnn.uint32, 64),
            ],
            rd={},
            sd={},
        )
        for i, (d, li, core) in enumerate(my_ports):
            ring_mask, wiring = _wiring(chips, nb, coord, d)
            port_idx = (r * C + c) * 8 + DIRS.index(d) * 2 + li
            inc = 1 + (XREADY_N - len(my_ports) if i == 0 else 0)
            pieces["rd"][(core.x, core.y)] = (
                [out_addr, plan_addr, man_addr, ring_offs]
                + cnt_addrs
                + [ring_mask, r, c, DIRS.index(d), li, plan_addr, port_idx, 0]
                + [len(rel), inc, words_addr + ready_off]
                + [v for x_ in rel for v in (x_.x, x_.y)]
            )
            peer = ports[d]["peer"]
            rc = virt(port_core[(peer, OPP[d], li)])
            pn = node(peer)
            # (x landed rests on every link's END reaching a port of the chip it leads to: _wiring's terminal stream
            # where the link's rows go no further)
            streams = []
            for on, target, k in wiring:
                v = virt(port_core[(peer, target, li)]) if on else None
                streams += [on, v.x, v.y, k] if on else [0, 0, 0, 0]
            pieces["sd"][(core.x, core.y)] = (
                [out_addr, ready_addr, rc.x, rc.y, int(pn.mesh_id), int(pn.chip_id), man_addr, ring_offs]
                + cnt_addrs
                + streams
                + [peer[0], peer[1], DIRS.index(d)]
            )
        out[coord] = pieces
    return out, (rinc, rph, sinc, sph, rd_ct, sd_ct)
