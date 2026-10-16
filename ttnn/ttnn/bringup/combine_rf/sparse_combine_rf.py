# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_combine_rf — combine level 1 with the fabric routers doing the relaying (prototype).

Same traffic as sparse_combine_l1 (every expert output row from its expert's chip to its token's home, Y first, then
X, on a fixed link), but a row is injected once, as a multi-hop 2D unicast: the routers forward it (the 2D fabric's
route tables are dimension-ordered, Y then X, the same paths), and it lands straight in the home accumulator's L1.

Cores per chip: a port core per (direction, link) (one fabric connection each) and A accumulator cores (the same
logical cores on every chip).
  accumulator o:  a ring of D slots per (source chip, link). Rows from source s to home h always leave s through the
                  same port (its first hop), so each ring has exactly one producer: port (s, first hop(s -> h), l).
  port (s, d, l): injects its rows; holds a credit counter per (home, accumulator) ring it feeds; returns the credits
                  of rows its own chip's accumulators consumed from sources whose first hop back is d (atomic incs,
                  multi-hop through the fabric as well).
Sinks never wait on anything but their own compute, so every row in the fabric has a slot reserved at its sink and the
fabric always drains. Counters are cumulative across calls (ring positions continue), so calls need no barrier: an
accumulator only takes its expected count from each ring, a later call's rows wait in the ring.
"""

import os

import torch

import ttnn

from .fabric_all_gather_py import NOC0, NOC1, _SEM_CACHE
from .sparse_all_gather_2d import DIRS, OPP, mesh_plan, place_ports
from .sparse_combine_l1 import _ACC_COMPUTE, _ACC_WRITER, _PORT_READER, TILE, route

CB_INJ, CB_ROW, CB_META, CB_TAB, CB_ACC, CB_OUT, CB_W = 0, 1, 2, 3, 4, 5, 6
PIECE_MAX = int(
    os.environ.get("SPC_PIECE", "7168")
)  # row bytes per fabric packet at most: wider rows go as P equal pieces (64-byte multiples), each
# with its own 32-byte trailer, so a 7232 B fabric payload (more router slots) serves every model's rows; pieces sit
# PS apart (64-byte aligned for the DRAM reads)

_RF_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t cb_inj = get_compile_time_arg_val(0);
constexpr uint32_t cb_tab = get_compile_time_arg_val(1);
constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
constexpr uint32_t SLOT = get_compile_time_arg_val(3);
constexpr uint32_t INJ_B = __INJB__;
constexpr uint32_t MISC = __MISC__;
constexpr uint32_t P = __P__, PD = __PD__, PS = __PS__;
constexpr auto y_args = TensorAccessorArgs<4>();
constexpr auto t_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
// Port reader: the port's injection list (rows of local experts whose first hop leaves through this port, in expert
// order), each row as P pieces with their trailers into the injection CB, a batch in flight. With a collector (W y
// writers, overlapped with the flat expert): a row of local expert e is read only once counts[e] (cumulative, +W per
// call) covers this call.
void kernel_main() {
    size_t a = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tab_bytes = get_arg_val<uint32_t>(a++), off = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++);
    const uint32_t area = get_arg_val<uint32_t>(a++), cx = get_arg_val<uint32_t>(a++), cy = get_arg_val<uint32_t>(a++);
    const uint32_t counts_addr = get_arg_val<uint32_t>(a++), W = get_arg_val<uint32_t>(a++);
    const uint32_t tdbg = get_arg_val<uint32_t>(a++), tpage = get_arg_val<uint32_t>(a++);
    constexpr uint32_t TPROF = __TPROF__;  // (timeline: when each local expert was released, start / end)
    uint32_t tl_[128];
    if constexpr (TPROF) {
        for (uint32_t i = 0; i < 128; ++i) tl_[i] = 0;
        tl_[0] = 0x7E000000 | tpage;
        tl_[1] = get_timestamp_32b();
    }
    const auto y = TensorAccessor(y_args, y_addr, row_bytes);
    const auto tab = TensorAccessor(t_args, tab_addr, tab_bytes);
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    uint32_t target = 0;
    if (W) {
        const uint32_t calls = misc[100] + 1;
        misc[100] = calls;
        target = calls * W;
    }
    uint32_t e_ok = 0xFFFFFFFF;  // (slot order: once local expert e is reported, every earlier one is)
    const uint32_t sc = area + MISC + 448;
    auto wait_e = [&](uint32_t e) {
        if (W == 0 || (e_ok != 0xFFFFFFFF && e <= e_ok)) return;
        const uint64_t src = get_noc_addr(cx, cy, counts_addr + (e & ~3u) * 4);
        while (true) {
            noc_async_read(src, sc, 16);
            noc_async_read_barrier();
            invalidate_l1_cache();
            if (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[e & 3] >= target) break;
        }
        e_ok = e;
        if constexpr (TPROF) {
            if (e < 124) tl_[4 + e] = get_timestamp_32b();
        }
    };
    cb_reserve_back(cb_tab, 1);
    const uint32_t tl = get_write_ptr(cb_tab);
    constexpr uint32_t TCH = 256;
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
    constexpr uint32_t B = INJ_B;
    for (uint32_t i = 0; i < n; i += B) {
        const uint32_t k = n - i < B ? n - i : B;
        cb_reserve_back(cb_inj, k);
        uint32_t w = get_write_ptr(cb_inj);
        const uint32_t lim = get_local_cb_interface(cb_inj).fifo_limit, fsz = get_local_cb_interface(cb_inj).fifo_size;
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t* ej = entry(i + j);
            const uint32_t yrow = ej[0], hdr = ej[1];
            wait_e(hdr >> 24);
            for (uint32_t pc = 0; pc < P; ++pc) {
                const uint32_t pb = row_bytes - pc * PD < PD ? row_bytes - pc * PD : PD;
                if constexpr (!__NOYREAD__) noc_async_read(y.get_noc_addr(yrow) + pc * PD, w + pc * PS, pb);
                volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + pc * PS + pb);
                tr[0] = hdr;
                tr[3] = 1;  // arrival flags: the last 16-byte chunk of each NoC write stream (0 / 1 mod 32)
                tr[7] = 1;
            }
            w += SLOT;
            if (w >= lim) w -= fsz;
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < k; ++j) cb_push_back(cb_inj, 1);
    }
    if constexpr (TPROF) {
        tl_[2] = get_timestamp_32b();
        tl_[3] = n;
        const uint32_t buf = tl;  // (the table CB: free by now; misc is shared with the sender)
        for (uint32_t i = 0; i < 128; ++i) reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf)[i] = tl_[i];
        const InterleavedAddrGen<true> dg = {.bank_base_address = tdbg, .page_size = 512};
        noc_async_write(buf, get_noc_addr(tpage, dg), 512);
        noc_async_write_barrier();
    }
}
"""
_RF_READER_L1 = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t cb_inj = get_compile_time_arg_val(0);
constexpr uint32_t MISC = __MISC__;
constexpr uint32_t GRANT_B = 8;  // grants go out in batches of this many slots (or the rest)
// Port reader, L1 hand-off from the flat expert's y writers (se_yrm.hpp SE_Y_L1): the writers put every row of this
// port straight into its injection ring (this CB) in the order they produce them; this reader grants ring slots
// (grant[d] on every writer, cumulative), and pushes a row once every writer's landed counter here covers it (the
// column-0 writer also wrote the trailer: header word and flags). Counters are cumulative across calls (misc[101]: this
// port's rows before this call).
void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);
    uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t W = get_arg_val<uint32_t>(a++);
    const uint32_t did = get_arg_val<uint32_t>(a++), Q = get_arg_val<uint32_t>(a++), words = get_arg_val<uint32_t>(a++);
    if constexpr (__DEVPLAN__) {  // (device plan: n is the plan's address)
        cb_reserve_back(__CBTAB__, 1);
        const uint32_t sc = (get_write_ptr(__CBTAB__) + 63) & ~63u;
        const InterleavedAddrGen<true> pg = {.bank_base_address = n, .page_size = 4096};
        noc_async_read(get_noc_addr(0, pg), sc, 64);
        noc_async_read_barrier();
        n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[did];
    }
    const uint32_t tdbg = get_arg_val<uint32_t>(a++), tpage = get_arg_val<uint32_t>(a++);
    uint32_t wx[32], wy[32];
    for (uint32_t w = 0; w < W; ++w) {
        wx[w] = get_arg_val<uint32_t>(a++);
        wy[w] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t t0 = get_timestamp_32b();
    uint32_t t_land = 0;
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* landed = misc + 128;
    const uint32_t base = misc[101];
    uint32_t granted = 0, pushed = 0;
    auto grant_to = [&](uint32_t g) {
        const uint32_t dlt = g - granted;
        for (uint32_t w = 0; w < W; ++w) noc_semaphore_inc(get_noc_addr(wx[w], wy[w], words + 4 * did), dlt);
        granted = g;
    };
    while (pushed < n) {
        invalidate_l1_cache();
        const uint32_t want = pushed + Q < n ? pushed + Q : n;
        uint32_t g = granted;
        while (g < want && cb_pages_reservable_at_back(cb_inj, g - pushed + 1)) ++g;
        if (g > granted && (g - granted >= GRANT_B || g == want)) grant_to(g);
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < W; ++w) {
            const uint32_t v = landed[w] - base;
            m = v < m ? v : m;
        }
        if (m >= n && t_land == 0) t_land = get_timestamp_32b();
        while (pushed < m && pushed < granted) {
            cb_push_back(cb_inj, 1);
            ++pushed;
        }
    }
    misc[101] = base + n;
    noc_async_atomic_barrier();
    if constexpr (__TPROF__) {  // (timeline: start, every row landed, end)
        cb_reserve_back(__CBTAB__, 1);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(__CBTAB__));
        d[0] = 0x7E300000 | tpage; d[1] = t0; d[2] = t_land ? t_land : get_timestamp_32b(); d[3] = get_timestamp_32b(); d[4] = n;
        const InterleavedAddrGen<true> dg = {.bank_base_address = tdbg, .page_size = 512};
        noc_async_write(get_write_ptr(__CBTAB__), get_noc_addr(tpage, dg), 64);
        noc_async_write_barrier();
    }
}
"""

# the weighted accumulate: each row times its gate weight (HiFi4, scalar broadcast from the weight CB's element 0) into
# DEST (fp32), packed onto the token's fp32 sum with packer L1 accumulation (scratch_l1acc_precision.py: exact)
_ACC_COMPUTE_W = (
    _ACC_COMPUTE.replace(
        "        const uint32_t t = read_tile_value(cb_meta, 0, 0);\n        cb_pop_front(cb_meta, 1);",
        "        const uint32_t t = read_tile_value(cb_meta, 0, 0);\n        cb_pop_front(cb_meta, 1);\n"
        "        if (__DEVPLAN__ && t == 0xFFFFFFFF) break;  // (device plan: the reader's end marker)",
    )
    .replace(
        "    for (uint32_t r = 0; r < n_rows; ++r) {",
        "    for (uint32_t r = 0; __DEVPLAN__ || r < n_rows; ++r) {",
    )
    .replace(
        '#include "api/compute/reconfig_data_format.h"',
        '#include "api/compute/reconfig_data_format.h"\n#include "api/compute/bcast.h"',
    )
    .replace(
        "    constexpr uint32_t MAXT = get_compile_time_arg_val(5);",
        "    constexpr uint32_t MAXT = get_compile_time_arg_val(5);\n    constexpr uint32_t cb_w = get_compile_time_arg_val(6);",
    )
    .replace(
        "    compute_kernel_hw_startup(cb_row, cb_row, cb_acc);\n    copy_init(cb_row);",
        "    compute_kernel_hw_startup(cb_row, cb_w, cb_acc);\n    mul_bcast_scalar_init(cb_row, cb_w);",
    )
    .replace(
        "        cb_wait_front(cb_row, NT);\n        if constexpr (ABL & 1) {  // ablation: no accumulation\n            cb_pop_front(cb_row, NT);",
        "        cb_wait_front(cb_row, NT);\n        cb_wait_front(cb_w, 1);\n        if constexpr (ABL & 1) {  // ablation: no accumulation\n            cb_pop_front(cb_row, NT);\n            cb_pop_front(cb_w, 1);",
    )
    .replace(
        "            for (uint32_t j = 0; j < nj; ++j) copy_tile(cb_row, j0 + j, j);",
        "            for (uint32_t j = 0; j < nj; ++j) mul_tiles_bcast_scalar(cb_row, cb_w, j0 + j, 0, j);",
    )
    .replace(
        "        seen[t] = 1;\n        cb_pop_front(cb_row, NT);",
        "        seen[t] = 1;\n        cb_pop_front(cb_row, NT);\n        cb_pop_front(cb_w, 1);",
    )
)
assert _ACC_COMPUTE_W.count("cb_pop_front(cb_w, 1)") == 2 and "mul_tiles_bcast_scalar" in _ACC_COMPUTE_W
assert "end marker" in _ACC_COMPUTE_W and "__DEVPLAN__ || r < n_rows" in _ACC_COMPUTE_W

# per-core L1 area: counters (cumulative), headers, then the accumulators' rings
CRED, USED, RET, MISC, HDRS, RINGS = 0, 1024, 2048, 4096, 5120, 22528
# MISC words: 1 returned so far; 16.. rows consumed per source chip (accumulators increment), 32.. rows covered by the
# credit snapshots sent; 64.. acc taken[] (per ring); 192.. profile. HDRS: N x NH routed headers, then snapshots

_RF_SENDER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
constexpr uint32_t cb_inj = get_compile_time_arg_val(0);
constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
constexpr uint32_t SLOT = get_compile_time_arg_val(2);
constexpr uint32_t D = get_compile_time_arg_val(3);  // ring slots per (source chip, link) at each accumulator
constexpr uint32_t A = get_compile_time_arg_val(4);
constexpr uint32_t AP = (A + 3) / 4 * 4;  // credit words per home (16-byte aligned arrays: they travel as NoC writes)
constexpr uint32_t N = get_compile_time_arg_val(5);
constexpr uint32_t L = get_compile_time_arg_val(6);
constexpr uint32_t ISLOT = get_compile_time_arg_val(7);
constexpr uint32_t CRED = __CRED__, USED = __USED__, RET = __RET__, MISC = __MISC__, HDRS = __HDRS__,
                   RINGS = __RINGS__;
constexpr uint32_t PROF = __PROF__;
constexpr uint32_t CRED_THR = __CREDTHR__;  // rows consumed from a source before its credits go back (or when idle)
constexpr uint32_t PACE = __PACE__;
constexpr uint32_t P = __P__, PD = __PD__, PS = __PS__;  // pieces per row, bytes per piece, piece stride
constexpr uint32_t LASTPB = (P - 1) * PS + (row_bytes - (P - 1) * PD);  // the last piece's trailer
constexpr uint32_t NH = 8;  // rotating row headers: flush (and pop the sent pages) every NH sends
constexpr uint32_t HSZ = 128;
static_assert(sizeof(PACKET_HEADER_TYPE) <= HSZ, "header stride");
constexpr uint32_t HW = sizeof(PACKET_HEADER_TYPE) / 4;

void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);
    const uint16_t mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t me = get_arg_val<uint32_t>(a++), lnk = get_arg_val<uint32_t>(a++);
    uint32_t n_inj = get_arg_val<uint32_t>(a++), n_ret = get_arg_val<uint32_t>(a++);
    if constexpr (__DEVPLAN__) {  // (device plan: the args are the plan's address and this port's id)
        const uint32_t sc = (area + __MISC__ + 832 + 63) & ~63u, pid = n_ret;
        const InterleavedAddrGen<true> pg = {.bank_base_address = n_inj, .page_size = 4096};
        noc_async_read(get_noc_addr(0, pg), sc, 64);  // n_inj[8], n_ret[8]
        noc_async_read_barrier();
        n_inj = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[pid];
        n_ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[8 + pid];
    }
    uint16_t chip_id[N];
    for (uint32_t i = 0; i < N; ++i) chip_id[i] = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    uint32_t ax[A], ay[A];
    for (uint32_t i = 0; i < A; ++i) {
        ax[i] = get_arg_val<uint32_t>(a++);
        ay[i] = get_arg_val<uint32_t>(a++);
    }
    // per source chip s: the port on s that feeds my chip on my link (where s's credits go); 0 if not mine to return
    uint32_t sx[N], sy[N];
    for (uint32_t i = 0; i < N; ++i) {
        sx[i] = get_arg_val<uint32_t>(a++);
        sy[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t pace = get_arg_val<uint32_t>(a++);  // cycles per injected row (0: as fast as possible)
    const uint32_t a_dbg = a;
    a += 4;
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    // headers per (destination chip, rotation), routed once; a send only sets the NoC command
    auto hdr_at = [&](uint32_t chip, uint32_t rot) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(area + HDRS + (chip * NH + rot) * HSZ);
    };
    for (uint32_t i = 0; i < N; ++i) {
        if (i == me) continue;
        for (uint32_t r = 0; r < NH; ++r) (void)fabric_set_unicast_route(hdr_at(i, r), chip_id[i], mesh);
    }
    const uint32_t cstage = area + HDRS + N * NH * HSZ;  // credit snapshots, one per rotation
    conn.open();

    volatile tt_l1_ptr uint32_t* cred = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + CRED);
    volatile tt_l1_ptr uint32_t* used = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + USED);
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + RET);
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* rtot = misc + 16;   // rows consumed per source chip (accumulators increment)
    volatile tt_l1_ptr uint32_t* rdone = misc + 32;  // rows covered by the last snapshot sent per source chip
    const uint32_t ret0 = misc[1];
    uint32_t returned = ret0;
    const uint32_t ring_base = RINGS + (me * L + lnk) * D * SLOT;  // my ring at every accumulator
    uint32_t inj_done = 0, sent = 0, pend_inj = 0, unflushed = 0;
    uint32_t p_idle = 0, p_cred = 0, p_slot = 0, p_flush = 0, n_credpk = 0, n_blocked = 0, n_empty = 0, p_send = 0;
    const uint32_t p0 = get_timestamp_32b();

    auto flush = [&]() {
        const uint32_t t0 = get_timestamp_32b();
        noc_async_writes_flushed();
        p_flush += get_timestamp_32b() - t0;
        for (; pend_inj; --pend_inj) cb_pop_front(cb_inj, 1);
        unflushed = 0;
    };
    // credits: a snapshot of the cumulative consumed counts of source s's rings (one per accumulator) written over
    // s's credit array for my chip: one small packet returns any number of rows
    auto give_back = [&](bool all) {
        for (uint32_t s = 0; s < N; ++s) {
            if (sx[s] == 0) continue;
            const uint32_t pend = rtot[s] - rdone[s];
            if (pend == 0 || (!all && pend < CRED_THR)) continue;
            if (unflushed >= NH) flush();
            const uint32_t t0 = get_timestamp_32b();
            const uint32_t rot = sent % NH;
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cstage + rot * 4 * AP);
            uint32_t cov = 0;
            for (uint32_t o = 0; o < A; ++o) {
                const uint32_t v = ret[s * AP + o];
                st[o] = v;
                cov += v;
            }
            auto* h = hdr_at(s, rot);
            h->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(sx[s], sy[s], area + CRED + 4 * me * AP)}, 4 * AP);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(cstage + rot * 4 * AP, 4 * AP, reinterpret_cast<uint32_t>(h));
            ++sent;
            ++unflushed;
            returned += cov - rdone[s];
            rdone[s] = cov;
            ++n_credpk;
            p_cred += get_timestamp_32b() - t0;
        }
    };
    auto any_ret = [&]() {
        uint32_t t = 0;
        for (uint32_t s = 0; s < N; ++s) t += rtot[s];
        return t;
    };
    const uint32_t tstart = get_timestamp_32b();

    while (inj_done < n_inj || returned - ret0 < n_ret) {
        bool did = false;
        const uint32_t it0 = get_timestamp_32b();
        invalidate_l1_cache();
        const uint32_t tot = any_ret();
        const bool pending = tot != returned;
        if (pending && unflushed < NH) give_back(false);
        while (unflushed + P <= NH && inj_done < n_inj && cb_pages_available_at_front(cb_inj, pend_inj + 1)) {
            if (PACE && get_timestamp_32b() - tstart < inj_done * pace) break;
            uint32_t src = get_read_ptr(cb_inj) + pend_inj * ISLOT;
            const uint32_t ilim = get_local_cb_interface(cb_inj).fifo_limit;
            if (src >= ilim) src -= get_local_cb_interface(cb_inj).fifo_size;
            const uint32_t hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + LASTPB)[0];
            const uint32_t home = hdr & 0xFF, t = (hdr >> 8) & 0xFFFF, o = t % A, ci = home * AP + o;
            const uint32_t u = used[ci];
            if (D + cred[ci] - u < 1) {
                ++n_blocked;
                break;
            }
            const uint32_t ts0 = get_timestamp_32b();
            const uint32_t dst = area + ring_base + (u % D) * SLOT;
            for (uint32_t pc = 0; pc < P; ++pc) {
                const uint32_t pb = (pc + 1 == P ? row_bytes - pc * PD : PD) + 32;
                auto* h = hdr_at(home, sent % NH);
                h->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(ax[o], ay[o], dst + pc * PS)}, pb);
                const uint32_t t0 = get_timestamp_32b();
                conn.wait_for_empty_write_slot();
                p_slot += get_timestamp_32b() - t0;
                conn.send_current_slot_non_blocking(src + pc * PS, pb, reinterpret_cast<uint32_t>(h));
                ++sent;
                ++unflushed;
            }
            used[ci] = u + 1;
            ++pend_inj;
            ++inj_done;
            did = true;
            p_send += get_timestamp_32b() - ts0;
        }
        if (unflushed >= NH || (!did && unflushed)) {
            flush();
            did = true;
        }
        if (!did && pending) {
            give_back(true);
            did = true;
        }
        if (!did) {
            p_idle += get_timestamp_32b() - it0;
            ++n_empty;
        }
    }
    flush();
    misc[1] = returned;
    // drain: every row this port injected is consumed (its credit snapshot landed), so nothing of this call reaches
    // this core after the program ends (the arena may be per call: its L1 is another op's after this)
    for (bool open_ = __DRAIN__; open_;) {
        invalidate_l1_cache();
        open_ = false;
        for (uint32_t ci = 0; ci < N * AP; ++ci) open_ |= cred[ci] != used[ci];
    }
    if constexpr (__TPROF__) {
        const uint32_t tdbg = get_arg_val<uint32_t>(a_dbg + 2), tpage = get_arg_val<uint32_t>(a_dbg + 3);
        volatile tt_l1_ptr uint32_t* d = misc + 192;
        d[0] = 0x7E100000 | tpage; d[1] = p0; d[2] = get_timestamp_32b(); d[3] = p_slot; d[4] = p_idle; d[5] = p_cred;
        d[6] = n_blocked; d[7] = n_inj; d[8] = p_send; d[9] = n_empty; d[10] = d[11] = d[12] = d[13] = d[14] = d[15] = 0;
        const InterleavedAddrGen<true> dg = {.bank_base_address = tdbg, .page_size = 512};
        noc_async_write(area + MISC + 192 * 4, get_noc_addr(tpage, dg), 64);
        noc_async_write_barrier();
    }
    if constexpr (PROF) {
        const uint32_t dbg = get_arg_val<uint32_t>(a_dbg), pidx = get_arg_val<uint32_t>(a_dbg + 1);
        volatile tt_l1_ptr uint32_t* d = misc + 192;
        d[0] = 0xDB900000 | pidx; d[1] = get_timestamp_32b() - p0; d[2] = p_idle; d[3] = p_cred; d[4] = p_slot;
        d[5] = p_flush; d[6] = n_credpk; d[7] = n_blocked; d[8] = n_empty; d[9] = n_inj; d[10] = n_ret; d[11] = p_send;
        d[12] = d[13] = d[14] = d[15] = 0;
        const InterleavedAddrGen<true> dg = {.bank_base_address = dbg, .page_size = 64};
        noc_async_write(area + MISC + 192 * 4, get_noc_addr(pidx, dg), 64);
        noc_async_write_barrier();
    }
    conn.close();
}
"""

_RF_ACC_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t cb_row = get_compile_time_arg_val(0);
constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
constexpr uint32_t cb_tab = get_compile_time_arg_val(2);
constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
constexpr uint32_t SLOT = get_compile_time_arg_val(4);
constexpr uint32_t D = get_compile_time_arg_val(5);
constexpr uint32_t A = get_compile_time_arg_val(6);
constexpr uint32_t AP = (A + 3) / 4 * 4;
constexpr uint32_t NT = get_compile_time_arg_val(7);
constexpr uint32_t N = get_compile_time_arg_val(8);
constexpr uint32_t L = get_compile_time_arg_val(9);
constexpr uint32_t RET = __RET__, MISC = __MISC__, RINGS = __RINGS__;
constexpr uint32_t PROF = __PROF__;
constexpr uint32_t ABL = __ABL__;
constexpr uint32_t P = __P__, PD = __PD__, PS = __PS__;
constexpr uint32_t LASTPB = (P - 1) * PS + (row_bytes - (P - 1) * PD);
constexpr auto y_args = TensorAccessorArgs<10>();
constexpr auto t_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
// Accumulator reader: local rows from DRAM, then its N * L rings (each up to this call's expected count); a consumed
// slot is one credit, counted on the port that faces the row's source (it sends the credits back).
void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tab_bytes = get_arg_val<uint32_t>(a++), off = get_arg_val<uint32_t>(a++);
    uint32_t n_local = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t cx = get_arg_val<uint32_t>(a++), cy = get_arg_val<uint32_t>(a++);
    const uint32_t counts_addr = get_arg_val<uint32_t>(a++), W = get_arg_val<uint32_t>(a++);
    uint32_t want[N * L], px[N * L], py[N * L];
    for (uint32_t k = 0; k < N * L; ++k) {
        want[k] = get_arg_val<uint32_t>(a++);
        px[k] = get_arg_val<uint32_t>(a++);
        py[k] = get_arg_val<uint32_t>(a++);
    }
    const auto y = TensorAccessor(y_args, y_addr, row_bytes);
    const auto tab = TensorAccessor(t_args, tab_addr, tab_bytes);
    cb_reserve_back(cb_tab, 1);
    const uint32_t tl = get_write_ptr(cb_tab);
    if constexpr (__DEVPLAN__) {  // (device plan: n_local is the plan's address; want[] from it too)
        const uint32_t sc = (tl + 63) & ~63u;
        const InterleavedAddrGen<true> pg = {.bank_base_address = n_local, .page_size = 4096};
        noc_async_read(get_noc_addr(0, pg) + 64, sc, 64);  // n_local[16]
        noc_async_read(get_noc_addr(0, pg) + (32 + me * 32) * 4, sc + 64, 128);  // want[me][32]
        noc_async_read_barrier();
        n_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[me];
        for (uint32_t k = 0; k < N * L; ++k) want[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc + 64)[k];
    }
    uint32_t n_rows = n_local;
    for (uint32_t k = 0; k < N * L; ++k) n_rows += want[k];
    constexpr uint32_t TCH = 256;
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* taken = misc + 64;  // per ring, cumulative
    constexpr uint32_t WGT = __L1__;  // (gate weights: with the L1 hand-off, from each row's trailer)
    auto push_w = [&](uint32_t wb) {
        if constexpr (WGT) {
            cb_reserve_back(__CBW__, 1);
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(__CBW__))[0] = wb & 0xFFFF;  // element 0
            cb_push_back(__CBW__, 1);
        }
    };
    auto push = [&](uint64_t src, uint32_t t) {
        push_w(0x3F80);  // (1.0)
        cb_reserve_back(cb_row, NT);
        cb_reserve_back(cb_meta, 1);
        if constexpr ((ABL & 2) == 0) noc_async_read(src, get_write_ptr(cb_row), row_bytes);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = t / A;
        noc_async_read_barrier();
        cb_push_back(cb_row, NT);
        cb_push_back(cb_meta, 1);
    };
    auto push_pieces = [&](uint32_t slot, uint32_t t) {  // a ring row: its pieces back to back into the CB
        push_w(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + LASTPB)[1]);
        cb_reserve_back(cb_row, NT);
        cb_reserve_back(cb_meta, 1);
        if constexpr ((ABL & 2) == 0) {
            for (uint32_t pc = 0; pc < P; ++pc) {
                noc_async_read(get_noc_addr(slot + pc * PS), get_write_ptr(cb_row) + pc * PD, pc + 1 == P ? row_bytes - pc * PD : PD);
            }
        }
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = t / A;
        noc_async_read_barrier();
        cb_push_back(cb_row, NT);
        cb_push_back(cb_meta, 1);
    };
    uint32_t done = 0;
    const uint32_t p0 = get_timestamp_32b();
    // local rows (experts on this chip), taken between ring polls as their experts finish (with a collector)
    uint32_t target = 0;
    if (W) {
        const uint32_t calls = misc[100] + 1;
        misc[100] = calls;
        target = calls * W;
    }
    uint32_t e_ok = 0xFFFFFFFF, li = 0, chunk0 = 0xFFFFFFFF, polls = 0;
    const uint32_t* le = nullptr;
    const uint32_t sc = area + MISC + 448;
    auto lentry = [&](uint32_t i) -> const uint32_t* {
        const uint32_t c0 = i / TCH * TCH;
        if (c0 != chunk0) {
            const uint32_t m = n_local - c0 < TCH ? n_local - c0 : TCH;
            const uint32_t start = (off + c0) * 8, al = start & ~63u, delta = start - al;
            noc_async_read(tab.get_noc_addr(0) + al, tl, ((m * 8 + delta + 63) / 64) * 64);
            noc_async_read_barrier();
            le = reinterpret_cast<const uint32_t*>(tl + delta);
            chunk0 = c0;
        }
        return le + 2 * (i - c0);
    };
    // L1 hand-off: local rows land in a ring of QL slots at LB (the y writers, se_yrm.hpp SE_Y_L1), granted from here
    constexpr uint32_t L1M = __L1__;
    const uint32_t QL = get_arg_val<uint32_t>(a + 4), LB = get_arg_val<uint32_t>(a + 5), WL = get_arg_val<uint32_t>(a + 6);
    const uint32_t ldid = get_arg_val<uint32_t>(a + 7), lwords = get_arg_val<uint32_t>(a + 8);
    volatile tt_l1_ptr uint32_t* landed = misc + 128;
    const uint32_t lbase = L1M ? misc[101] : 0;
    uint32_t lgranted = 0;
    auto lgrant_to = [&](uint32_t g) {
        for (uint32_t w = 0; w < WL; ++w) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(a + 9 + 2 * w), get_arg_val<uint32_t>(a + 10 + 2 * w), lwords + 4 * ldid),
                g - lgranted);
        }
        lgranted = g;
    };
    if (L1M && n_local) lgrant_to(n_local < QL ? n_local : QL);
    auto expert_done = [&](uint32_t e) -> bool {
        if (W == 0 || (e_ok != 0xFFFFFFFF && e <= e_ok)) return true;
        if ((polls++ & 15) != 0) return false;  // (a remote read every 16th look)
        noc_async_read(get_noc_addr(cx, cy, counts_addr + (e & ~3u) * 4), sc, 16);
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[e & 3] < target) return false;
        e_ok = e;
        return true;
    };
    uint32_t t_local = 0;
    while (done < n_rows) {
        invalidate_l1_cache();
        if (li == n_local && t_local == 0) t_local = get_timestamp_32b();
        while (L1M && li < n_local) {
            uint32_t m = 0xFFFFFFFF;
            for (uint32_t w = 0; w < WL; ++w) {
                const uint32_t v = landed[w] - lbase;
                m = v < m ? v : m;
            }
            if (li >= m) break;
            const uint32_t sl = LB + (li % QL) * SLOT;
            push_pieces(sl, (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl + LASTPB)[0] >> 8) & 0xFFFF);
            ++li;
            ++done;
            const uint32_t g = li + QL < n_local ? li + QL : n_local;
            if (g > lgranted && (g - lgranted >= 4 || g == n_local)) lgrant_to(g);
        }
        while (!L1M && li < n_local) {
            const uint32_t* en = lentry(li);
            if (!expert_done(en[1] >> 24)) break;
            push(__NOYREAD__ ? get_noc_addr(tl) : y.get_noc_addr(en[0]), (en[1] >> 8) & 0xFFFF);  // (probe: no y reads)
            ++li;
            ++done;
        }
        for (uint32_t k = 0; k < N * L; ++k) {
            if (!want[k]) continue;
            const uint32_t tk = taken[k];
            const uint32_t s = area + RINGS + (k * D + tk % D) * SLOT;
            bool all = true;
            for (uint32_t pc = 0; pc < P && all; ++pc) {
                volatile tt_l1_ptr uint32_t* tp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + pc * PS + (pc + 1 == P ? row_bytes - pc * PD : PD));
                all = tp[3] != 0 && tp[7] != 0;
            }
            if (all) {
                volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB);
                push_pieces(s, (tr[0] >> 8) & 0xFFFF);
                for (uint32_t pc = 0; pc < P; ++pc) {
                    volatile tt_l1_ptr uint32_t* tp =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + pc * PS + (pc + 1 == P ? row_bytes - pc * PD : PD));
                    tp[3] = 0;
                    tp[7] = 0;
                }
                taken[k] = tk + 1;
                --want[k];
                ++done;
                noc_semaphore_inc(get_noc_addr(px[k], py[k], area + RET + 4 * ((k / L) * AP + me)), 1);
                noc_semaphore_inc(get_noc_addr(px[k], py[k], area + MISC + 4 * (16 + k / L)), 1);
            }
        }
    }
    if constexpr (__DEVPLAN__) {  // (the compute's end marker: it does not know the row count)
        cb_reserve_back(cb_meta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = 0xFFFFFFFF;
        cb_push_back(cb_meta, 1);
    }
    const uint32_t p_total = get_timestamp_32b() - p0;
    if (L1M) misc[101] = lbase + n_local;
    noc_async_atomic_barrier();
    if constexpr (__TPROF__) {
        const uint32_t tdbg = get_arg_val<uint32_t>(a + 2), tpage = get_arg_val<uint32_t>(a + 3);
        volatile tt_l1_ptr uint32_t* d = misc + 192;
        d[0] = 0x7E200000 | tpage; d[1] = p0; d[2] = t_local; d[3] = p0 + p_total; d[4] = n_local; d[5] = n_rows;
        for (uint32_t i = 6; i < 16; ++i) d[i] = 0;
        const InterleavedAddrGen<true> dg = {.bank_base_address = tdbg, .page_size = 512};
        noc_async_write(area + MISC + 192 * 4, get_noc_addr(tpage, dg), 64);
        noc_async_write_barrier();
    }
    if constexpr (PROF) {
        const uint32_t dbg = get_arg_val<uint32_t>(a), pidx = get_arg_val<uint32_t>(a + 1);
        volatile tt_l1_ptr uint32_t* d = misc + 192;
        d[0] = 0xDBA00000 | pidx; d[1] = p_total; d[2] = n_local; d[3] = n_rows;
        const InterleavedAddrGen<true> dg = {.bank_base_address = dbg, .page_size = 64};
        noc_async_write(area + MISC + 192 * 4, get_noc_addr(pidx, dg), 64);
        noc_async_write_barrier();
    }
}
"""


def first_hop(src, dst, R, C, wrap_x, wrap_y):
    h = route(src, dst, R, C, wrap_x, wrap_y, 0)
    return h[0][1] if h else None


def sparse_combine_rf_prepare(
    mesh_device,
    S,
    W,
    rows_by_chip,
    *,
    num_links=2,
    torus=(False, False),
    n_acc=16,
    depth=8,
    cells=None,
    arena=None,
    writers=0,
    counts_off=None,
    l1=None,
    devplan=None,
):
    """rows_by_chip[coord] = list of (y row on that chip, home coord, home token t, link[, local expert]). cells: the
    logical (x, y) cells the combine may use (ports nearest their Ethernet cores, accumulators the rest), else the
    whole grid. arena: an L1 tensor sharded over (at least) the combine's cells, zero at creation and kept by the
    caller (e.g. the flat expert's arena, flat_routed_expert_with_program): the combine's state, rings and CBs live
    in it. writers > 0: overlapped with the flat expert, whose `writers` y writers report each local expert to the
    collector (the first accumulator, counts at area + counts_off): rows of expert e are read once it is reported.
    l1: dict(writers=[logical (x, y) of the flat y writers], words=address of their L1 words): the L1 hand-off (no y
    in DRAM; flat_routed_expert_with_program l1_args = run.l1_args). Returns run(y, out) (standalone) with
    run.build(y, out) -> the MeshProgramDescriptor and run.collector. devplan (with l1): dict(plan=, table=) from
    combine_plan (the planner's per-call counts and hand-off table, on the device): the kernels read their counts from
    the plan, rows_by_chip only places the cores."""
    R, C = tuple(mesh_device.shape)
    N, L, A0 = R * C, num_links, n_acc
    row_bytes = W * 2
    NT = (row_bytes + TILE - 1) // TILE
    NP = (row_bytes + PIECE_MAX - 1) // PIECE_MAX
    PD = ((row_bytes + NP - 1) // NP + 63) // 64 * 64
    PS = ((PD + 32 + 63) // 64) * 64
    SLOT = ISLOT = NP * PS
    D = depth
    chips, nb, (wrap_x, wrap_y) = mesh_plan(mesh_device, num_links, torus)
    port_core = place_ports(mesh_device, chips, cells)
    used_cores = {(c.x, c.y) for c in port_core.values()}
    grid = mesh_device.compute_with_storage_grid_size()
    pool = cells if cells is not None else [(x, y) for y in range(grid.y) for x in range(grid.x)]
    acc_cores = [ttnn.CoreCoord(x, y) for x, y in pool if (x, y) not in used_cores][:A0]
    A = len(acc_cores)
    assert N * ((A + 3) // 4 * 4) <= 256 and N <= 16
    run_cells = dict(ports={k: (c.x, c.y) for k, c in port_core.items()}, accs=[(c.x, c.y) for c in acc_cores])
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    idx = lambda ch: ch[0] * C + ch[1]
    fh = lambda s, d: first_hop(s, d, R, C, wrap_x, wrap_y)

    inj = {k: [] for k in port_core}
    n_ret = {k: 0 for k in port_core}
    loc = {(ch, i): [] for ch in chips for i in range(A)}
    want = {(ch, i): [0] * (N * L) for ch in chips for i in range(A)}
    for ch, rows in rows_by_chip.items():
        for row in rows:
            yrow, home, t, l = row[:4]
            assert t < (1 << 16)
            hdr = idx(home) | (t << 8) | ((row[4] if len(row) > 4 else 0) << 24)
            o = t % A
            if home == ch:
                loc[(home, o)].append((yrow, hdr))
                continue
            inj[(ch, fh(ch, home), l)].append((yrow, hdr))
            want[(home, o)][idx(ch) * L + l] += 1
            n_ret[(home, fh(home, ch), l)] += 1
    area_bytes = RINGS + N * L * D * SLOT
    all_cores = sorted(used_cores | {(c.x, c.y) for c in acc_cores})
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in all_cores])
    if arena is None:
        key = (id(mesh_device), tuple(all_cores), "combine_rf", area_bytes)
        if key not in _SEM_CACHE:
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
            _SEM_CACHE[key] = (mesh_device, None, area_t)
        area_t = _SEM_CACHE[key][2]
        arena_bytes = None
    else:
        area_t = arena
        sh = arena.memory_config().shard_spec
        arena_bytes = int(sh.shape[0]) * int(sh.shape[1]) * (4 if arena.dtype == ttnn.uint32 else 2)
    area = int(area_t.buffer_address())
    cnt_off = RET + 1024 if counts_off is None else counts_off
    assert cnt_off % 16 == 0 and RET + 4 * N * ((A + 3) // 4 * 4) <= cnt_off and cnt_off + 512 <= MISC
    coll = acc_cores[0]
    coll_v = virt(coll)

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
    if l1 is not None:  # destination ids per chip: its ports (sorted), then its accumulators
        port_ids = {ch: {k: j for j, k in enumerate(sorted(k for k in port_core if k[0] == ch))} for ch in chips}
        n_y = max((max((r[0] for r in rows), default=-1) for rows in rows_by_chip.values()), default=-1) + 1
        n_tiles = max((n_y + 31) // 32, 1)
        lt = torch.full((R, C, n_tiles, 64), 0, dtype=torch.int64)
        for ch, rows in rows_by_chip.items():
            for row in rows:
                yrow, home, t, l = row[:4]
                hdr = idx(home) | (t << 8) | ((row[4] if len(row) > 4 else 0) << 24)
                dd = len(port_ids[ch]) + t % A if home == ch else port_ids[ch][(ch, fh(ch, home), l)]
                wb = row[5] if len(row) > 5 else 0x3F80  # (gate weight, bf16 bits; 1.0 without)
                lt[ch[0], ch[1], yrow // 32, 2 * (yrow % 32)] = dd | (wb << 16)
                lt[ch[0], ch[1], yrow // 32, 2 * (yrow % 32) + 1] = hdr
        l1_tab = ttnn.from_torch(
            lt.to(torch.int32).reshape(R, C, n_tiles, 64),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        wv = [virt(ttnn.CoreCoord(*c)) for c in l1["writers"]]
        wxy = [v for c in wv for v in (c.x, c.y)]
        assert len(wv) <= 32
        QP = min(64, (N * L * D * SLOT) // ISLOT)  # a port's injection ring: in its (unused) ring region
    l1_args = []
    prof = os.environ.get("SPC_PROF") == "1"
    pace_us = float(os.environ.get("SPC_PACE", "0"))  # pace every port's injections to finish at this time
    inj_b = int(os.environ.get("SPC_INJB", "8"))
    subs = {
        "__CRED__": str(CRED),
        "__USED__": str(USED),
        "__RET__": str(RET),
        "__MISC__": str(MISC),
        "__HDRS__": str(HDRS),
        "__RINGS__": str(RINGS),
        "__PROF__": str(int(prof)),
        "__ABL__": os.environ.get("SPC_ABL", "0"),
        "__INJB__": str(inj_b),
        "__CREDTHR__": os.environ.get("SPC_CREDTHR", "8"),
        "__PACE__": str(int(pace_us > 0)),
        "__P__": str(NP),
        "__TPROF__": os.environ.get("SPC_TPROF", "0"),
        "__L1__": str(int(l1 is not None)),
        "__CBTAB__": str(CB_TAB),
        "__CBW__": str(CB_W),
        "__DEVPLAN__": str(int(devplan is not None)),
        "__NOYREAD__": os.environ.get("SPC_NOYREAD", "0"),  # (probe: rows not read from y: garbage)
        "__PD__": str(PD),
        "__PS__": str(PS),
        "__DRAIN__": "false" if os.environ.get("SPC_NODRAIN") == "1" else "true",  # (probe: no end-of-call credit drain)
    }
    dbg_t = ttnn.from_torch(
        torch.zeros(R, C, 128, 16, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1)),
    )
    tdbg_t = ttnn.from_torch(  # (SPC_TPROF timeline: pages 0-7 port readers, 8-15 senders, 16.. accumulators)
        torch.zeros(R, C, 48, 128, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1)),
    )
    srcs = {}
    for name, ksrc in (
        ("pr", _RF_READER),
        ("pr1", _RF_READER_L1),
        ("ps", _RF_SENDER),
        ("ar", _RF_ACC_READER),
        ("ac", _ACC_COMPUTE_W if l1 is not None else _ACC_COMPUTE),
        ("aw", _ACC_WRITER),
    ):
        for k, v in subs.items():
            ksrc = ksrc.replace(k, v)
        srcs[name] = ksrc
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    chip_ids = [int(node((i // C, i % C)).chip_id) for i in range(N)]
    mesh_id = int(node((0, 0)).mesh_id)
    acc_xy = []
    for ac in acc_cores:
        v = virt(ac)
        acc_xy += [v.x, v.y]
    state = {}

    def arena_cbs(placed, arena_t):
        cbs = []
        for cb, crs, off, total, fmt, page in placed:
            d_ = ttnn.cb_descriptor_from_sharded_tensor(cb, arena_t, address_offset=off, total_size=total, core_ranges=crs)
            d_.format_descriptors = [ttnn.CBFormatDescriptor(cb, fmt, page)]
            cbs.append(d_)
        return cbs

    def rebind(arena_t):
        """The last build's descriptor with its CBs on arena_t (a per-call arena at the same address): what a
        program-cache miss needs (the CBs bind the buffer, a freed one is no use)."""
        assert int(arena_t.buffer_address()) == area, "the arena moved: the kernels bake its address"
        mesh_desc = ttnn.MeshProgramDescriptor()
        for ch, program, placed in state["placed"]:
            program.cbs = arena_cbs(placed, arena_t)
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*ch), ttnn.MeshCoordinate(*ch))] = program
        return mesh_desc

    def build(y, out, arena_t=None):
        nonlocal area_t
        if arena_t is not None:  # (a per-call arena: the same address, its buffer bound by the CBs)
            assert int(arena_t.buffer_address()) == area, "the arena moved: the kernels bake its address"
            area_t = arena_t
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        t_ct = list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
        o_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        mesh_desc = ttnn.MeshProgramDescriptor()
        state["placed"] = []
        for ch, ports in chips.items():
            program = ttnn.ProgramDescriptor()
            placed = []  # (the arena-backed CBs: rebind(arena) rebuilds them on another arena tensor)
            state["placed"].append((ch, program, placed))
            my = [(d, li, port_core[(ch, d, li)]) for d, p in ports.items() for li in range(len(p["links"]))]
            pset = ttnn.CoreRangeSet([ttnn.CoreRange(k, k) for _, _, k in my])
            aset = ttnn.CoreRangeSet([ttnn.CoreRange(k, k) for k in acc_cores])
            tab_cb = 256 * 8 + 128
            specs = [  # (cb, cores, total, format, page): CB_TAB first (both kinds), then each kind's own
                (CB_TAB, pset.merge(aset), tab_cb, ttnn.uint32, tab_cb),
                (CB_INJ, pset, 2 * inj_b * ISLOT, ttnn.uint32, ISLOT),
                (CB_ROW, aset, 2 * NT * TILE, ttnn.bfloat16, TILE),
                (CB_META, aset, 2 * 32, ttnn.uint32, 32),
                (CB_ACC, aset, MAXT * NT * 4096, ttnn.float32, 4096),
                (CB_OUT, aset, 2 * NT * TILE, ttnn.bfloat16, TILE),
            ] + ([(CB_W, aset, 2 * TILE, ttnn.bfloat16, TILE)] if l1 is not None else [])
            cbs = []  # (assigned at the end: program.cbs is a copy through the binding, appends to it are lost)
            if arena is None:
                for cb, crs, total, fmt, page in specs:
                    cbs.append(
                        ttnn.CBDescriptor(
                            total_size=total,
                            core_ranges=crs,
                            format_descriptors=[ttnn.CBFormatDescriptor(cb, fmt, page)],
                        )
                    )
            else:
                al = lambda v: (v + 63) // 64 * 64
                base = al(area_bytes)
                ends = {"p": base, "a": base}
                for cb, crs, total, fmt, page in specs:
                    kinds = ("p", "a") if cb == CB_TAB else (("p",) if cb == CB_INJ else ("a",))
                    if l1 is not None and cb == CB_INJ:  # (the hand-off ring: the writers address it directly)
                        placed.append((cb, crs, RINGS, QP * ISLOT, fmt, page))
                        continue
                    off = max(ends[k] for k in kinds)
                    for k in kinds:
                        ends[k] = al(off + total)
                    placed.append((cb, crs, off, total, fmt, page))
                cbs = arena_cbs(placed, area_t)
                assert (
                    max(ends.values()) <= arena_bytes
                ), f"combine needs {max(ends.values()) >> 10} KB per core, the arena has {arena_bytes >> 10} KB"
                if l1 is not None:  # the accumulators' local ring: what is left of the arena
                    loc_off = ends["a"]
                    QL = min(16, (arena_bytes - loc_off) // ISLOT)
                    assert QL >= 2, f"no room for the accumulators' local ring ({(arena_bytes - loc_off) >> 10} KB)"
            program.cbs = cbs
            if l1 is not None:
                pids = port_ids[ch]
                tab_dev = devplan["table"] if devplan is not None else l1_tab
                la = [len(pids) + A, ISLOT, int(tab_dev.buffer_address()), int(l1["words"]), 0, 0, 0, 0]
                for k in sorted(pids, key=pids.get):
                    v = virt(port_core[k])
                    la += [v.x, v.y, area + RINGS, QP, area + MISC + 512]
                for i_, ac in enumerate(acc_cores):
                    v = virt(ac)
                    la += [v.x, v.y, area + loc_off, QL, area + MISC + 512]
                l1_args.append(la)
            pr_rt, ps_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for d, li, core in my:
                k = (ch, d, li)
                if l1 is not None:
                    pr_rt[core.x][core.y] = [
                        area,
                        int(devplan["plan"].buffer_address()) if devplan is not None else len(inj[k]),
                        len(wv),
                        port_ids[ch][k],
                        QP,
                        int(l1["words"]),
                        int(tdbg_t.buffer_address()),
                        DIRS.index(d) * 2 + li,
                    ] + wxy
                else:
                    pr_rt[core.x][core.y] = [
                        int(y.buffer_address()),
                        int(tab_t.buffer_address()),
                        tab_bytes,
                        offs[("p",) + k],
                        len(inj[k]),
                        area,
                        coll_v.x,
                        coll_v.y,
                        area + cnt_off,
                        writers,
                        int(tdbg_t.buffer_address()),
                        DIRS.index(d) * 2 + li,
                    ]
                peer = ports[d]["peer"]
                cnt_args = (
                    [int(devplan["plan"].buffer_address()), port_ids[ch][k]]
                    if devplan is not None
                    else [len(inj[k]), n_ret[k]]
                )
                args = [area, mesh_id, idx(ch), li] + cnt_args + chip_ids + acc_xy
                for i in range(N):
                    s = (i // C, i % C)
                    if s != ch and fh(ch, s) == d:
                        v = virt(port_core[(s, fh(s, ch), li)])
                        args += [v.x, v.y]
                    else:
                        args += [0, 0]
                args += [int(pace_us * 1350 / len(inj[k])) if pace_us > 0 and inj[k] else 0]
                args += [int(dbg_t.buffer_address()), DIRS.index(d) * 2 + li]
                args += [int(tdbg_t.buffer_address()), 8 + DIRS.index(d) * 2 + li]
                args += list(ttnn.setup_fabric_connection(node(ch), node(peer), ports[d]["links"][li], program, core))
                ps_rt[core.x][core.y] = args
            ar_rt, ac_rt, aw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, core in enumerate(acc_cores):
                n_tok = len(range(i, S, A))
                args = [
                    area,
                    int(y.buffer_address()),
                    int(tab_t.buffer_address()),
                    tab_bytes,
                    offs[("a", ch, i)],
                    int(devplan["plan"].buffer_address()) if devplan is not None else len(loc[(ch, i)]),
                    i,
                    coll_v.x,
                    coll_v.y,
                    area + cnt_off,
                    writers,
                ]
                for kk in range(N * L):
                    s, li = (kk // L // C, kk // L % C), kk % L
                    w = want[(ch, i)][kk]
                    if devplan is not None and s != ch:  # (the plan decides the counts: every ring's producer port)
                        v = virt(port_core[(ch, fh(ch, s), li)])
                        args += [0, v.x, v.y]
                    elif w:
                        v = virt(port_core[(ch, fh(ch, s), li)])
                        args += [w, v.x, v.y]
                    else:
                        args += [0, 0, 0]
                args += [int(dbg_t.buffer_address()), 8 + i, int(tdbg_t.buffer_address()), 16 + i]
                if l1 is not None:
                    args += [QL, area + loc_off, len(wv), len(port_ids[ch]) + i, int(l1["words"])] + wxy
                else:
                    args += [0, 0, 0, 0, 0]
                ar_rt[core.x][core.y] = args
                n_rows = len(loc[(ch, i)]) + sum(want[(ch, i)])
                ac_rt[core.x][core.y] = [0 if devplan is not None else n_rows, n_tok]
                aw_rt[core.x][core.y] = [int(out.buffer_address()), n_tok, i]
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=srcs["pr1"] if l1 is not None else srcs["pr"],
                    source_type=st,
                    core_ranges=pset,
                    compile_time_args=[CB_INJ] if l1 is not None else [CB_INJ, CB_TAB, row_bytes, ISLOT] + y_ct + t_ct,
                    runtime_args=pr_rt,
                    config=dm("RISCV_1", NOC0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["ps"],
                    source_type=st,
                    core_ranges=pset,
                    compile_time_args=[CB_INJ, row_bytes, SLOT, D, A, N, L, ISLOT],
                    runtime_args=ps_rt,
                    config=dm("RISCV_0", NOC1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["ar"],
                    source_type=st,
                    core_ranges=aset,
                    compile_time_args=[CB_ROW, CB_META, CB_TAB, row_bytes, SLOT, D, A, NT, N, L] + y_ct + t_ct,
                    runtime_args=ar_rt,
                    config=dm("RISCV_1", NOC0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["ac"],
                    source_type=st,
                    core_ranges=aset,
                    compile_time_args=[CB_ROW, CB_META, CB_ACC, CB_OUT, NT, MAXT] + ([CB_W] if l1 is not None else []),
                    runtime_args=ac_rt,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
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

    def build_l1(y, out, arena_t=None):  # (the descriptor and, per chip in its order, the y writers' hand-off args)
        l1_args.clear()
        desc = build(y, out, arena_t)
        return desc, [list(v) for v in l1_args]

    # what a fresh (uninitialized) arena needs zeroed for a call to start from scratch: per core (logical x, y), runs of
    # (offset in the arena, bytes, stride, count): the state block, and at the accumulators every ring slot's piece
    # trailers (a slot is taken when its trailer flags are set)
    trl = [pc * PS + (row_bytes - pc * PD if pc + 1 == NP else PD) for pc in range(NP)]
    zero_runs = {c: [(0, RINGS, 0, 1)] for c in used_cores}
    for c in acc_cores:
        zero_runs[(c.x, c.y)] = [(0, RINGS, 0, 1)] + [(RINGS + t, 32, SLOT, N * L * D) for t in trl]

    run.build = build
    run.build_l1 = build_l1
    run.rebind = rebind
    run.tdbg = tdbg_t
    run.tab = tab_t
    run.collector = dict(core=coll, counts_addr=area + cnt_off, writers=writers)
    run.counts = dict(n_inj={k: len(v) for k, v in inj.items()}, n_ret=n_ret)
    run.cells = run_cells
    run.zero_runs = zero_runs
    run.dbg = dbg_t
    return run
