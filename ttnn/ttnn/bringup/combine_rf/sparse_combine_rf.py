# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_combine_rf — combine level 1 with the fabric routers doing the relaying (prototype).

Same traffic as sparse_combine_l1 (every expert output row from its expert's chip to its token's home, Y first, then
X, on a fixed link), but a row is injected once, as a multi-hop 2D unicast: the routers forward it (the 2D fabric's
route tables are dimension-ordered, Y then X, the same paths), and it lands straight in the home accumulator's L1.

Cores per chip: a port core per (direction, link) (one fabric connection each) and A accumulator cores (the same
logical cores on every chip).
  accumulator o:  a landing pool of PSL slots (the arena left after the accumulator's CBs, capped at depth x (N - 1)),
                  carved per call into one ring per source chip by that call's demand: a source with rows here gets one
                  slot, the rest in proportion to its rows (at most its rows). The pool's L1 does not grow with N (a
                  Galaxy's 31 sources at a useful fixed depth would not fit), and a hot source gets the slots. A row of
                  home token t goes to accumulator o = t % A over link (o + s + h) % L: fixed per (accumulator, source,
                  home), so each ring has exactly one producer, port (s, first hop(s -> h), that link).
  port (s, d, l): injects its rows; holds a credit counter per (home, accumulator) ring it feeds; returns the credits
                  of rows its own chip's accumulators consumed from sources whose first hop back is d (atomic incs,
                  multi-hop through the fabric as well), and forwards those sources their rings' layouts (base, depth)
                  for the call, which its accumulators publish at their start.
Sinks never wait on anything but their own compute, so every row in the fabric has a slot reserved at its sink and the
fabric always drains. Counters are cumulative across calls; a ring's slots restart each call (every call drains, and a
source's rows wait for this call's layout from the home, so a call never lands in another call's carving).
"""

import collections
import math
import os
import struct
from dataclasses import dataclass

import torch

import ttnn

from .fabric_all_gather_py import NOC0, NOC1, _SEM_CACHE
from .sparse_all_gather_2d import DIRS, OPP, mesh_plan, place_ports
from .combine_plan import link_of
from .sparse_combine_l1 import _ACC_COMPUTE, _ACC_WRITER, _PORT_READER, TILE, acc_compute_config, route, tie_mode

CB_INJ, CB_ROW, CB_META, CB_TAB, CB_ACC, CB_OUT, CB_W = 0, 1, 2, 3, 4, 5, 6
DISP_CBS = dict(rows=24, meta=25, plan=26, stage=27, dbg=28)  # (the dispatch phase's, on the port cores)
PIECE_MAX = 7168  # row bytes per fabric packet at most (default; piece_max): wider rows go as P equal pieces (64-B multiples), each
# with its own 32-byte trailer, so a 7232 B fabric payload (more router slots) serves every model's rows; pieces sit
# PS apart (64-byte aligned for the DRAM reads)


@dataclass(frozen=True)
class CombineProbes:
    """Perf / debug probes of the combine: never set by model code, read from the environment in this one place
    (SPC_* names). Production choices are sparse_combine_rf_prepare's arguments (piece_max, inj_slots, depth /
    ring_depth, n_acc, acc_on_ports, ...)."""

    prof: bool = False  # SPC_PROF=1: profile words
    tprof: str = "0"  # SPC_TPROF: per-call timeline (scratch_flat_combine_rf.py TPROF lines)
    pace_us: float = 0.0  # SPC_PACE: pace every port's injections to finish at this time
    inj_b: int = 8  # SPC_INJB: injection batch
    abl: str = "0"  # SPC_ABL: ablation mask
    credthr: str = "8"  # SPC_CREDTHR: credit return threshold
    credidle: int = 0  # SPC_CREDIDLE: idle credit sends wait for this many rows ... (0: the combine's choice)
    credage: int = 0  # SPC_CREDAGE: ... or for their oldest to be this many cycles old
    credauto: bool = False  # SPC_CREDAUTO=1: credits only at the end when every pool holds its call (see the prepare)
    hcache: bool = True  # SPC_HCACHE=0: route the senders' headers every call (no DRAM header cache)
    far: int = 0  # SPC_FAR=h: within the look-ahead window, rows going >= h hops first (0: queue order)
    credfin: bool = False  # SPC_CREDFIN=1: a source's credits go back at once when final (every row of it consumed here)
    credbusy: int = 0  # SPC_CREDBUSY: busy passes too send a source's credits under the threshold this many cycles old
    noyread: str = "0"  # SPC_NOYREAD: rows not read from y (garbage)
    nodrain: bool = False  # SPC_NODRAIN=1: no end-of-call credit drain
    piggy: str = ""  # SPC_PIGGY: 1 / 0 overrides piggyback_credits
    hw: bool = False  # SPC_HW=1: hang words - a sparse snapshot of each port's / accumulator's loop state in MISC 480..
    fill_y: float = 0.0  # SPC_FILLY: filler bytes a Y-bound row byte, to the Y neighbour's sink (fill_sink): a Y link's
    # load as on the Galaxy 8x4 (its Y edges carry 2-2.5 x the 2x4's: fill 1.0 mean, 1.5 the busiest)
    bk: int = 1  # SPC_BK: bookkeeping load x bk - every credit snapshot / ring layout packet sent bk times (the copies
    # idempotent: the same cumulative counts / layouts), every ring polled bk times a pass - the per-source bookkeeping of
    # bk x the sources (the Galaxy's 31 from the LoudBox's 7 at 4); the data path unchanged
    la: int = 0  # SPC_LA=w: the sender looks up to w rows past a row whose ring is full (0 / 1: strict queue order)
    gbgate: bool = False  # SPC_GBGATE=1: the sender's busy-pass credit scan only when the total pending reaches CRED_THR
    ytie: int = 0  # SPC_YTIE=1 (sparse_combine_l1.tie_mode; =fab: the fabric's parity tables): Y ties (the opposite row of a ring) split Y+ / Y- by column parity (route(ytie=)):
    # only the first hop's port changes - the 2D action map routes each later row by the destination's table entry
    snoc: str = "1"  # SPC_SNOC=0: the sender on NoC0 by running it on RISCV_1 (its readers move to RISCV_0 / NoC1)
    se: str = ""  # SPC_SE: shared_rows probes - off (no rows, both sides: wrong sums) / noread (no DRAM reads: garbage)
    # / noown (the accumulators skip the own block)

    @staticmethod
    def from_env(env=None):
        env = os.environ if env is None else env
        return CombineProbes(
            prof=env.get("SPC_PROF") == "1",
            tprof=env.get("SPC_TPROF", "0"),
            pace_us=float(env.get("SPC_PACE", "0")),
            inj_b=int(env.get("SPC_INJB", "8")),
            abl=env.get("SPC_ABL", "0"),
            credthr=env.get("SPC_CREDTHR", "8"),
            credidle=int(env.get("SPC_CREDIDLE", "0")),
            credage=int(env.get("SPC_CREDAGE", "0")),
            credbusy=int(env.get("SPC_CREDBUSY", "0")),
            credfin=env.get("SPC_CREDFIN") == "1",
            far=int(env.get("SPC_FAR", "0")),
            ytie=tie_mode(env),
            gbgate=env.get("SPC_GBGATE") == "1",
            hcache=env.get("SPC_HCACHE", "1") != "0",
            credauto=env.get("SPC_CREDAUTO") == "1",
            noyread=env.get("SPC_NOYREAD", "0"),
            nodrain=env.get("SPC_NODRAIN") == "1",
            se=env.get("SPC_SE", ""),
            piggy=env.get("SPC_PIGGY", ""),
            hw=env.get("SPC_HW") == "1",
            bk=int(env.get("SPC_BK", "1")),
            fill_y=float(env.get("SPC_FILLY", "0")),
            snoc=env.get("SPC_SNOC", "1"),
            la=int(env.get("SPC_LA", "0")),
        )


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
# DEST (fp32), packed onto the token's fp32 sum with packer L1 accumulation (scratch_l1acc_precision.py: exact). snap_q
# (SNAP): each product first rounded on the SFPU to a multiple of q, so every fp32 add is exact while the partial sums
# stay below 2^24 q - the sum is then the same in any landing order
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
        '#include "api/compute/reconfig_data_format.h"\n#include "api/compute/bcast.h"\n'
        '#include "api/compute/eltwise_unary/binop_with_scalar.h"\n'
        "constexpr uint32_t SNAP = __SNAP__;  // (fp32 bits of C; 0: off)",
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
        "            if constexpr (SNAP) mul_bcast_scalar_init(cb_row, cb_w);\n"
        "            for (uint32_t j = 0; j < nj; ++j) mul_tiles_bcast_scalar(cb_row, cb_w, j0 + j, 0, j);\n"
        "            if constexpr (SNAP) {  // every term to a multiple of q: (v + C) - C, C = 1.5 x 2^23 q\n"
        "                binop_with_scalar_tile_init();\n"
        "                for (uint32_t j = 0; j < nj; ++j) {\n"
        "                    add_unary_tile(j, SNAP);\n"
        "                    sub_unary_tile(j, SNAP);\n"
        "                }\n"
        "            }",
    )
    .replace(
        "        seen[t] = 1;\n        cb_pop_front(cb_row, NT);",
        "        seen[t] = 1;\n        cb_pop_front(cb_row, NT);\n        cb_pop_front(cb_w, 1);",
    )
)
assert _ACC_COMPUTE_W.count("add_unary_tile(j, SNAP)") == 1
assert _ACC_COMPUTE_W.count("cb_pop_front(cb_w, 1)") == 2 and "mul_tiles_bcast_scalar" in _ACC_COMPUTE_W
assert "end marker" in _ACC_COMPUTE_W and "__DEVPLAN__ || r < n_rows" in _ACC_COMPUTE_W

# per-core L1 area (state_layout): credits / used / returned per (home, accumulator), the ring layouts received (at a
# source's ports) and to forward (at a home's ports), the rings' first count this call (N x AP words each), the collector
# counts (512 B), MISC (2 KB), HDRS (N x NH routed headers, then NH credit snapshots), then the landing pool
NH_, HSZ_ = 8, 128
MISC_BYTES = 2048
# MISC words: 1 returned so far; 100 calls; 101 base; 112.. scratch; 128.. landed (writers); 192.. profile; 208..
# devplan scratch; 256.. rows consumed per source chip (accumulators increment); 288.. rows covered by the credit
# snapshots sent; 320.. acc taken[] (per ring = source chip); 477..511 the hang words (CombineProbes.hw)
M_RTOT, M_RDONE, M_TAKEN = 256, 288, 320
GRANTS = 8  # (landing "shared": grant ids outstanding a (source, accumulator) at most - its grant ring)


def state_layout(N, A, G=0):
    """Byte offsets of the state block: CRED, USED, RET, CNT (collector counts), MISC, HDRS, RINGS (its end; the
    landing pool starts there), and LAYR / LAYO / U0 (state_layout_ext). G: the shared landing's grant ring size (0:
    rings - the layout unchanged)."""
    CRED, USED, RET, _, _, _, CNT, MISC, HDRS, RINGS, _, _ = _state(N, A, G)[:12]
    return CRED, USED, RET, CNT, MISC, HDRS, RINGS


def state_layout_ext(N, A, G=0):
    """LAYR (layouts received per (home, accumulator) at a source port), LAYO (per (source, accumulator), written by the
    home's accumulators, forwarded by its port), U0 (a source port's used count per ring at this call's start)."""
    return _state(N, A, G)[3:6]


def state_layout_grants(N, A, G):
    """The shared landing's GIDH (at a home port: per (source, accumulator) its G grant ids) and GREC (at a source port:
    per home its snapshot - AP grant counts, then AP x G ids)."""
    return _state(N, A, G)[12:14]


def _state(N, A, G=0):
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    AP = (A + 3) // 4 * 4
    rw = al(4 * N * AP)
    CRED, USED, RET, LAYR, LAYO, U0, CNT = (i * rw for i in range(7))
    # (SHW: a port's monotonic credit count per ring, merged from its snapshots and the piggybacked words; PIG: per
    # (home chip, receiving accumulator of this port's link class: 8 at most), the 16-bit consumed counts the rows from
    # that chip carried - a slot per accumulator, so each only moves forward: one path, one ring, in order)
    SHW = 7 * rw
    PIG = SHW + rw
    CNT = al(PIG + 16 * 8 * N)
    GIDH = GREC = 0
    if G:  # (shared landing: the grant rings, before the counts)
        GIDH = CNT
        GREC = al(GIDH + 4 * N * AP * G)
        CNT = al(GREC + 4 * N * AP * (1 + G))
    MISC = CNT + 512
    HDRS = MISC + MISC_BYTES
    RINGS = al(HDRS + N * NH_ * HSZ_ + NH_ * 4 * AP * (1 + G))
    return CRED, USED, RET, LAYR, LAYO, U0, CNT, MISC, HDRS, RINGS, SHW, PIG, GIDH, GREC


def state_layout_pig(N, A):
    """SHW (a port's monotonic credit count per ring) and PIG (per home chip, the piggybacked consumed counts)."""
    return _state(N, A)[10:12]  # (the same with grants: they sit after PIG)


_RF_SENDER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
constexpr uint32_t cb_inj = get_compile_time_arg_val(0);
constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
constexpr uint32_t SLOT = get_compile_time_arg_val(2);
constexpr uint32_t PSL = get_compile_time_arg_val(3);  // landing pool slots at each accumulator (carved per call)
constexpr uint32_t A = get_compile_time_arg_val(4);
constexpr uint32_t AP = (A + 3) / 4 * 4;  // credit words per home (16-byte aligned arrays: they travel as NoC writes)
constexpr uint32_t N = get_compile_time_arg_val(5);
constexpr uint32_t L = get_compile_time_arg_val(6);
constexpr uint32_t ISLOT = get_compile_time_arg_val(7);
constexpr uint32_t CRED = __CRED__, USED = __USED__, RET = __RET__, MISC = __MISC__, HDRS = __HDRS__,
                   RINGS = __RINGS__, LAYR = __LAYR__, LAYO = __LAYO__, U0 = __U0__;
constexpr uint32_t PROF = __PROF__;
#define TSX() ((PROF || __TPROF__) ? get_timestamp_32b() : 0u)  // (loop timers: only when profiling)
constexpr uint32_t CRED_THR = __CREDTHR__;  // rows consumed from a source before its credits go back (or when idle)
constexpr uint32_t CRED_IDLE = __CREDIDLE__, CRED_AGE = __CREDAGE__;
constexpr uint32_t CRED_BUSY = __CREDBUSY__;
constexpr uint32_t CREDFIN = __CREDFIN__;
constexpr uint32_t FAR = __FAR__;  // (look-ahead: rows going >= FAR hops first)  // (a source's credits go back at once when final: all its rows here consumed)  // (busy passes: credits under CRED_THR go once this many cycles old; 0 off)
constexpr uint32_t FILLY = __FILLY__, FILL_SINK = __FILLSINK__, FC_ = __FILLC__;  // (probe SPC_FILLY: Y filler x 256)  // (idle: CRED_IDLE rows, or CRED_AGE cycles old)
constexpr uint32_t PACE = __PACE__;
constexpr uint32_t P = __P__, PD = __PD__, PS = __PS__;  // pieces per row, bytes per piece, piece stride
constexpr uint32_t LASTPB = (P - 1) * PS + (row_bytes - (P - 1) * PD);  // the last piece's trailer
constexpr uint32_t NH = __NH__;  // rotating row headers: flush (and pop the sent pages) every NH sends
constexpr uint32_t HSZ = __HSZ__;
static_assert(sizeof(PACKET_HEADER_TYPE) <= HSZ, "header stride");
constexpr uint32_t HW = sizeof(PACKET_HEADER_TYPE) / 4;
constexpr uint32_t LA = __LA__;  // look-ahead: rows past a blocked one (its ring full / no layout yet) the sender may send
constexpr uint32_t NL = __NL__;  // fabric connections: 1 (a port per link) or L (a port per direction, all its links)
// shared-expert partial rows (shared_rows): the chip's partial [C S, W] row-major in DRAM (block j: owner column
// c0 + j of my mesh row, j 0 = mine, not sent); this port's rows are read into NSE staging slots and sent before the
// flat's rows, on the same rings and credits (static counts)
constexpr uint32_t SE = __SE__, NSE = __NSE__;
// barrier (the arena zeroed right before this program, no other cross-chip sync before it): every port core counts one
// atomic increment from every other chip in a persistent word (bar[0], cumulative; bar[1] its calls) and sends
// nothing remote before all N - 1 of this call are in; each chip's port facing a chip increments that chip's ports
constexpr uint32_t BAR = __BAR__;
// piggybacked credits (PIGGY): a row to home k carries, in its trailer's spare words, this port's consumed counts of
// k's rows on this link class (up to 8 accumulators, 16 bits each, cumulative); k's accumulator copies them to its
// port facing me on that link (PIG words), which merges them with its snapshots (SHW: monotonic, so late or reordered
// copies never take credits back). Standalone snapshots remain for when no row goes that way.
constexpr uint32_t PIGGY = __PIGGY__, PIG = __PIG__, SHW = __SHW__;
// landing "shared": a home's accumulators grant slot ids (GIDH at this port, per (source, accumulator) G words, id |
// grant seq << 24; RET counts them); this port forwards each source a snapshot of its validated grant counts and ids
// (GREC at the source's port, per home: AP counts then AP x G ids); a row goes into its grant's slot
constexpr uint32_t SHARED = __SHARED__, G = __G__, GIDH = __GIDH__, GREC = __GREC__;
constexpr uint32_t SNAPW = SHARED ? AP * (1 + G) : AP;  // (a snapshot's words)
#define GTAG(q) (0x80u | ((q) & 0x7Fu))  // (a grant's tag: never 0 - a zeroed ring entry is no grant)
#if __FUSED__
// (a port that is also an accumulator, acc_on_ports: this RISC also runs the port's L1 hand-off - slot grants to the y
// writers, pushes of landed rows - and the accumulator's output writes; RISCV_1 runs the accumulator's reader)
constexpr uint32_t cb_out = __CBOUT__;
constexpr uint32_t NT_ = __NT__;
constexpr auto o_args = TensorAccessorArgs<8>();
#endif

void kernel_main() {
    const uint32_t t_k0 = get_timestamp_32b();
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
#ifdef ACC_LUT
    // (per-chip roles: a home chip's accumulator cells differ - its table entry home * A + o, x | y << 8, copied here)
    const uint32_t lut_dram = get_arg_val<uint32_t>(a++), lut_l1 = get_arg_val<uint32_t>(a++);
    {
        const InterleavedAddrGen<true> lg = {.bank_base_address = lut_dram, .page_size = LUT_BYTES};
        noc_async_read(get_noc_addr(0, lg), lut_l1, LUT_BYTES);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint16_t* acc_lut = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lut_l1);
#define ACC_X(h, o) (acc_lut[(h) * A + (o)] & 0xFFu)
#define ACC_Y(h, o) (acc_lut[(h) * A + (o)] >> 8)
#else
    uint32_t ax[A], ay[A];
    for (uint32_t i = 0; i < A; ++i) {
        ax[i] = get_arg_val<uint32_t>(a++);
        ay[i] = get_arg_val<uint32_t>(a++);
    }
#define ACC_X(h, o) ax[o]
#define ACC_Y(h, o) ay[o]
#endif
    // per source chip s: the port on s that feeds my chip on my link (where s's credits go); 0 if not mine to return
    uint32_t sx[N], sy[N];
    for (uint32_t i = 0; i < N; ++i) {
        sx[i] = get_arg_val<uint32_t>(a++);
        sy[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t rfin[N];  // (CREDFIN: rows of source s whose credits this port returns this call)
    for (uint32_t i = 0; i < N; ++i) rfin[i] = CREDFIN ? get_arg_val<uint32_t>(a++) : 0;
    uint8_t hops[N];
    for (uint32_t i = 0; i < N; ++i) hops[i] = FAR ? static_cast<uint8_t>(get_arg_val<uint32_t>(a++)) : 0;
    const uint32_t pace = get_arg_val<uint32_t>(a++);  // cycles per injected row (0: as fast as possible)
    const uint32_t a_dbg = a;
    a += 6;  // (dbg, page, tdbg, page, header cache, page)
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
#if __NL__ > 1
    auto conn1 = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
#endif
    // a row's connection: its link (accumulator + source + home) % L with a port per direction, else the port's one
    auto conn_of = [&](uint32_t l) -> decltype(conn)& {
#if __NL__ > 1
        if (l) return conn1;
#endif
        (void)l;
        return conn;
    };

    // headers per (destination chip, rotation), routed once; a send only sets the NoC command
    auto hdr_at = [&](uint32_t chip, uint32_t rot) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(area + HDRS + (chip * NH + rot) * HSZ);
    };
    // headers, cached in DRAM: a 2D route costs ~1.7 us (decode_route_to_buffer) - N x NH of them were ~90 us of every
    // call's start, and a home port forwards its layouts only after them. The first call routes one header per
    // destination, copies it to the other rotations (same route: a send rewrites only the NoC command) and stores the
    // block with a key in this port's DRAM page; later calls read the block back
    {
        constexpr uint32_t HBLK = N * NH * HSZ;
        const uint32_t hc_addr = get_arg_val<uint32_t>(a_dbg + 4), hc_page = get_arg_val<uint32_t>(a_dbg + 5);
        const InterleavedAddrGen<true> hcg = {.bank_base_address = hc_addr, .page_size = HBLK + 64};
        volatile tt_l1_ptr uint32_t* kw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC) + 224;
        const uint32_t key = 0x48C00000u ^ (me << 8) ^ (lnk << 16) ^ N;
        noc_async_read(get_noc_addr(hc_page, hcg, HBLK), reinterpret_cast<uint32_t>(kw), 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (__HCACHE__ && kw[0] == key) {
            noc_async_read(get_noc_addr(hc_page, hcg), area + HDRS, HBLK);
            noc_async_read_barrier();
        } else {
            for (uint32_t i = 0; i < N; ++i) {
                if (i == me) continue;
                auto* h0 = hdr_at(i, 0);
                (void)fabric_set_unicast_route(h0, chip_id[i], mesh);
                const volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(h0);
                for (uint32_t r = 1; r < NH; ++r) {
                    volatile tt_l1_ptr uint32_t* hr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_at(i, r));
                    for (uint32_t w = 0; w < HW; ++w) hr[w] = s0[w];
                }
            }
            noc_async_write(area + HDRS, get_noc_addr(hc_page, hcg), HBLK);
            kw[0] = key;
            noc_async_write(reinterpret_cast<uint32_t>(kw), get_noc_addr(hc_page, hcg, HBLK), 64);
            noc_async_write_barrier();
        }
    }
    const uint32_t cstage = area + HDRS + N * NH * HSZ;  // credit snapshots, one per rotation
    const uint32_t t_hdr = get_timestamp_32b();
    conn.open();
#if __NL__ > 1
    conn1.open();
#endif
#if __FUSED__
    const uint32_t hW = get_arg_val<uint32_t>(a++), hdid = get_arg_val<uint32_t>(a++), hQ = get_arg_val<uint32_t>(a++);
    const uint32_t hwords = get_arg_val<uint32_t>(a++), o_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_tok = get_arg_val<uint32_t>(a++), acc_me = get_arg_val<uint32_t>(a++);
    uint32_t hwx[32], hwy[32];
    for (uint32_t w = 0; w < hW; ++w) {
        hwx[w] = get_arg_val<uint32_t>(a++);
        hwy[w] = get_arg_val<uint32_t>(a++);
    }
    const auto ob = TensorAccessor(o_args, o_addr, row_bytes);
    volatile tt_l1_ptr uint32_t* hland = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC) + 128;
    const uint32_t hbase = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC)[101];
    uint32_t hgranted = 0, hpushed = 0, wr_i = 0;
    // the hand-off receiver's step (as _RF_READER_L1): grant ring slots, push rows every writer has landed
    auto hand_off = [&]() {
        if (hpushed >= n_inj) return;
        const uint32_t want = hpushed + hQ < n_inj ? hpushed + hQ : n_inj;
        uint32_t g = hgranted;
        while (g < want && cb_pages_reservable_at_back(cb_inj, g - hpushed + 1)) ++g;
        if (g > hgranted && (g - hgranted >= 8 || g == want)) {
            for (uint32_t w = 0; w < hW; ++w) noc_semaphore_inc(get_noc_addr(hwx[w], hwy[w], hwords + 4 * hdid), g - hgranted);
            hgranted = g;
        }
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < hW; ++w) {
            const uint32_t v = hland[w] - hbase;
            m = v < m ? v : m;
        }
        while (hpushed < m && hpushed < hgranted) {
            cb_push_back(cb_inj, 1);
            ++hpushed;
        }
    };
    // the accumulator writer's step (as _ACC_WRITER): one finished token row to the output
    auto write_out = [&]() {
        if (wr_i < n_tok && cb_pages_available_at_front(cb_out, NT_)) {
            noc_async_write(get_read_ptr(cb_out), ob.get_noc_addr(wr_i * A + acc_me), row_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, NT_);
            ++wr_i;
        }
    };
    auto fused_more = [&]() { return hpushed < n_inj || wr_i < n_tok; };
#else
    auto hand_off = [&]() {};
    auto write_out = [&]() {};
    auto fused_more = [&]() { return false; };
#endif
    uint32_t se_addr = 0, se_n = 0, se_stage = 0, se_S = 1, se_no = 0, se_home[4] = {0, 0, 0, 0}, se_blk[4] = {0, 0, 0, 0};
    if constexpr (SE) {
        se_addr = get_arg_val<uint32_t>(a++);
        se_n = get_arg_val<uint32_t>(a++);
        se_stage = get_arg_val<uint32_t>(a++);
        se_S = get_arg_val<uint32_t>(a++);
        n_ret += get_arg_val<uint32_t>(a++);  // (credits of the row-mates' partial rows consumed here)
        se_no = get_arg_val<uint32_t>(a++);
        for (uint32_t i = 0; i < se_no; ++i) {
            se_home[i] = get_arg_val<uint32_t>(a++);
            se_blk[i] = get_arg_val<uint32_t>(a++);
        }
    }
    if constexpr (BAR) {
        const uint32_t bar_addr = get_arg_val<uint32_t>(a++), n_bt = get_arg_val<uint32_t>(a++);
        for (uint32_t i = 0; i < n_bt; ++i) {
            const uint32_t ch = get_arg_val<uint32_t>(a++), bx = get_arg_val<uint32_t>(a++), by = get_arg_val<uint32_t>(a++);
            auto* h = hdr_at(ch, 0);
            h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(bx, by, bar_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(h), sizeof(PACKET_HEADER_TYPE));
        }
        volatile tt_l1_ptr uint32_t* bw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bar_addr);
        const uint32_t tgt = (bw[1] + 1) * (N - 1);
        do {
            invalidate_l1_cache();
        } while (static_cast<int32_t>(bw[0] - tgt) < 0);
        bw[1] = bw[1] + 1;
    }
    const uint32_t t_bar = get_timestamp_32b();
    const InterleavedAddrGen<true> se_g = {.bank_base_address = se_addr, .page_size = row_bytes};
    uint32_t se_oi = 0, se_t = 0, se_rd = 0, se_ready = 0, se_sent = 0, se_free = 0;

    volatile tt_l1_ptr uint32_t* cred = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + CRED);
    volatile tt_l1_ptr uint32_t* used = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + USED);
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + RET);
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* rtot = misc + __MRTOT__;    // rows consumed per source chip (accumulators increment)
    volatile tt_l1_ptr uint32_t* rdone = misc + __MRDONE__;  // rows covered by the last snapshot sent per source chip
    const uint32_t ret0 = misc[1];
    uint32_t returned = ret0;
    // the landing pools' rings this call (layout words: tag << 24 | base << 12 | depth, tag = this call's 1..255): a
    // home's accumulators carve their pools by this call's demand and publish each source's ring to the home port that
    // returns that source's credits (LAYO), which forwards them to the source's port (LAYR). A row waits for its ring's
    // layout; its slot is base + (rows sent on the ring this call) % depth. Rings are empty at a call's start (every
    // call drains), so the cumulative credit counts still hold: room = depth + cred - used.
    volatile tt_l1_ptr uint32_t* lay_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + LAYR);
    volatile tt_l1_ptr uint32_t* lay_o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + LAYO);
    volatile tt_l1_ptr uint32_t* u0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + U0);
    const uint32_t calls_ = misc[104] + 1;
    misc[104] = calls_;
    const uint32_t tag = (calls_ - 1) % 255 + 1;
    if constexpr (__HW__) {  // (hang words: the layouts already here when this kernel starts, by home: 4 bits each)
        uint32_t m = 0;
        for (uint32_t ci = 0; ci < N * AP; ++ci) m += (lay_r[ci] != 0 ? 1u : 0u) << (4 * (ci / AP));
        misc[477] = m;
    }
    uint32_t lay_pend = 0;  // (sources whose layouts this port still forwards: bit s)
    if constexpr (!SHARED) {  // (shared: U0 holds the grant counts validated and forwarded, cumulative)
        for (uint32_t ci = 0; ci < N * AP; ++ci) u0[ci] = used[ci];
        for (uint32_t s = 0; s < N; ++s) {
            if (sx[s]) lay_pend |= 1u << s;
        }
    }
    volatile tt_l1_ptr uint32_t* gidh = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + GIDH);
    volatile tt_l1_ptr uint32_t* grec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + GREC);
    uint32_t inj_done = 0, sent = 0, pend_inj = 0, unflushed = 0;
    uint64_t ahead = 0;  // (rows sent past the prefix, bit j = pend_inj + j; LA <= 64)
    (void)ahead;
    uint32_t p_idle = 0, p_cred = 0, p_slot = 0, p_flush = 0, n_credpk = 0, n_blocked = 0, n_empty = 0, p_send = 0;
    uint32_t p_book = 0, p_blk = 0, n_iter = 0;  // (TPROF: bookkeeping per pass, passes spent blocked on a full ring)
    uint32_t t_first = 0, t_last = 0, n_bl = 0, n_bf = 0;  // (TPROF: first / last row sent; blocked: no layout yet / ring full)
    const uint32_t p0 = get_timestamp_32b();

    auto flush = [&]() {
        const uint32_t t0 = TSX();
        noc_async_writes_flushed();
        p_flush += TSX() - t0;
        for (; pend_inj; --pend_inj) cb_pop_front(cb_inj, 1);
        se_free = se_sent;
        unflushed = 0;
    };
    // credits: a snapshot of the cumulative consumed counts of source s's rings (one per accumulator) written over
    // s's credit array for my chip: one small packet returns any number of rows
    // (idle sends: at least CRED_IDLE rows, or rows waiting CRED_AGE cycles - fewer, fuller snapshots; every row's
    // credit still goes back within CRED_AGE of an idle pass)
    uint32_t tfirst[N], tbusy[N], rt0[N];
    for (uint32_t s = 0; s < N; ++s) tfirst[s] = tbusy[s] = 0, rt0[s] = rtot[s];  // (this call's baseline: no row of a
    // source lands before this port forwards its layouts)
    auto give_back = [&](bool all) {
        const uint32_t now = get_timestamp_32b();
        for (uint32_t s = 0; s < N; ++s) {
            if (sx[s] == 0) continue;
            const uint32_t pend = rtot[s] - rdone[s];
            if (pend == 0) continue;
            const bool fin = CREDFIN && rtot[s] - rt0[s] == rfin[s];  // (its last credits: the source's drain waits on them)
            if (!all && pend < CRED_THR && !fin) {
                if constexpr (CRED_BUSY == 0) continue;
                if (tbusy[s] == 0) tbusy[s] = now | 1;
                if (now - tbusy[s] < CRED_BUSY) continue;
            }
            tbusy[s] = 0;
            if constexpr (CRED_IDLE > 1) {
                if (all && pend < CRED_IDLE && !fin) {
                    if (tfirst[s] == 0) tfirst[s] = now | 1;
                    if (now - tfirst[s] < CRED_AGE) continue;
                }
                tfirst[s] = 0;
            }
            if (unflushed >= NH) flush();
            const uint32_t t0 = TSX();
            const uint32_t rot = sent % NH;
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cstage + rot * 4 * SNAPW);
            uint32_t cov = 0;
            for (uint32_t o = 0; o < A; ++o) {
                uint32_t v = ret[s * AP + o];
                if constexpr (SHARED) {  // (a grant's count goes only as far as its ids have landed here: tags)
                    const uint32_t gb = (s * AP + o) * G;
                    uint32_t w = u0[s * AP + o];
                    while (w != v && (gidh[gb + w % G] >> 24) == GTAG(w)) ++w;
                    u0[s * AP + o] = w;
                    v = w;
                    for (uint32_t j = 0; j < G; ++j) st[AP + o * G + j] = gidh[gb + j];
                }
                st[o] = v;
                cov += v;
            }
            if constexpr (SHARED) {
                if (cov == rdone[s]) continue;  // (nothing validated yet: later)
            }
            auto* h = hdr_at(s, rot);
            h->to_noc_unicast_write(
                NocUnicastCommandHeader{get_noc_addr(sx[s], sy[s], SHARED ? area + GREC + 4 * me * SNAPW : area + CRED + 4 * me * AP)},
                4 * SNAPW);
            for (uint32_t b = 0; b < __BK__; ++b) {  // (probe SPC_BK: the same snapshot bk times - idempotent)
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(cstage + rot * 4 * SNAPW, 4 * SNAPW, reinterpret_cast<uint32_t>(h));
            }
            ++sent;
            ++unflushed;
            returned += cov - rdone[s];
            rdone[s] = cov;
            ++n_credpk;
            p_cred += TSX() - t0;
        }
    };
    // forward each source's ring layouts once every accumulator that returns its credits here has published them
    auto send_layouts = [&]() {
        for (uint32_t s = 0; s < N && lay_pend; ++s) {
            if (!(lay_pend >> s & 1)) continue;
            bool ok = true;
            for (uint32_t o = 0; o < A && ok; ++o) {
                if (NL > 1 || (o + s + me) % L == lnk) ok = (lay_o[s * AP + o] >> 24) == tag;
            }
            if (!ok) continue;
            if (unflushed >= NH) flush();
            const uint32_t rot = sent % NH;
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cstage + rot * 4 * AP);
            for (uint32_t o = 0; o < A; ++o) st[o] = lay_o[s * AP + o];
            auto* h = hdr_at(s, rot);
            h->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(sx[s], sy[s], area + LAYR + 4 * me * AP)}, 4 * AP);
            for (uint32_t b = 0; b < __BK__; ++b) {  // (probe SPC_BK: the same layouts bk times - idempotent)
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(cstage + rot * 4 * AP, 4 * AP, reinterpret_cast<uint32_t>(h));
            }
            ++sent;
            ++unflushed;
            lay_pend &= ~(1u << s);
        }
    };
    auto any_ret = [&]() {
        uint32_t t = 0;
        for (uint32_t s = 0; s < N; ++s) t += rtot[s];
        return t;
    };
    volatile tt_l1_ptr uint32_t* pig = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + PIG);
    volatile tt_l1_ptr uint32_t* shw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + SHW);
    // ring ci's credits, cheap: the snapshot count and the last merged piggyback (SHW); the slots only when that is
    // not enough (send_row) or at the drain
    auto cred_fast = [&](uint32_t ci) -> uint32_t {
        if constexpr (SHARED) return grec[(ci / AP) * SNAPW + ci % AP];  // (the home's grants to me, validated)
        const uint32_t c = cred[ci];
        if constexpr (PIGGY) {
            const uint32_t s = shw[ci];
            return static_cast<int32_t>(c - s) > 0 ? c : s;
        }
        return c;
    };
    // ring ci's credits: the snapshot count and the piggybacked one merged into SHW (both monotonic)
    auto cred_of = [&](uint32_t ci) -> uint32_t {
        if constexpr (SHARED) return grec[(ci / AP) * SNAPW + ci % AP];  // (the grants: the drain waits for the last)
        const uint32_t c = cred[ci];
        if constexpr (PIGGY) {
            const uint32_t home = ci / AP, o = ci % AP, j = o / L;
            if ((o + me + home) % L != lnk) return c;  // (another link class's ring: never piggybacked here)
            uint32_t s = shw[ci];
            if (static_cast<int32_t>(c - s) > 0) s = c;
            for (uint32_t q = 0; q < (A + L - 1) / L; ++q) {  // (the receiving accumulators' slots: the furthest)
                const uint32_t w = pig[(home * 8 + q) * 4 + (j >> 1)], p16 = (j & 1) ? (w >> 16) : (w & 0xFFFF);
                const uint32_t d = (p16 - s) & 0xFFFF;
                if (d && d < 0x8000) s += d;
            }
            shw[ci] = s;
            return s;
        }
        return c;
    };
    uint32_t n_piggy = 0;
    // one row (P pieces, its trailer's header word at LASTPB) from L1 src to its home's ring; false: not yet (the
    // ring's layout has not arrived, or the ring is full)
    uint32_t fill_debt = 0;  // (probe SPC_FILLY)
    auto send_row = [&](uint32_t src) -> bool {
        const uint32_t hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + LASTPB)[0];
        const uint32_t home = hdr & 0xFF, t = (hdr >> 8) & 0xFFFF, o = t % A, ci = home * AP + o;
        const uint32_t u = used[ci], lw = lay_r[ci], dep = lw & 0xFFF;
        uint32_t gslot = 0;
        if constexpr (SHARED) {  // (my u-th row to this accumulator: grant u's slot, once its id carries u's tag)
            if (static_cast<int32_t>(cred_fast(ci) - u) < 1) {
                ++n_bf;
                return false;
            }
            const uint32_t gw = grec[home * SNAPW + AP + o * G + u % G];
            if ((gw >> 24) != GTAG(u)) {
                ++n_bl;
                return false;
            }
            gslot = gw & 0xFFFF;
        } else {
            if ((lw >> 24) != tag) {
                ++n_bl;
                return false;
            }
            if (dep + cred_fast(ci) - u < 1 && (!PIGGY || dep + cred_of(ci) - u < 1)) {
                ++n_bf;
                return false;
            }
        }
        if constexpr (PIGGY) {  // (this port returns home's credits: the row carries this link class's counts)
            // (the flag set or cleared on every row: a staged slot keeps its previous row's trailer words)
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + LASTPB)[1] &= 0xFFFFu;
            if (sx[home] && rtot[home] != rdone[home]) {
                volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + LASTPB);
                uint32_t w[4] = {0, 0, 0, 0}, cov = 0;
                for (uint32_t o2 = 0; o2 < A; ++o2) {
                    const uint32_t v = ret[home * AP + o2];
                    cov += v;
                    if ((o2 + me + home) % L == lnk) {
                        const uint32_t j = o2 / L;
                        w[j >> 1] |= (v & 0xFFFF) << (16 * (j & 1));
                    }
                }
                tr[2] = w[0];
                tr[4] = w[1];
                tr[5] = w[2];
                tr[6] = w[3];
                tr[1] = tr[1] | 0x10000u;
                returned += cov - rdone[home];
                rdone[home] = cov;
                ++n_piggy;
            }
        }
        const uint32_t dst = SHARED ? area + RINGS + gslot * SLOT                                // (my grant's slot)
                                    : area + RINGS + (((lw >> 12) & 0xFFF) + (u - u0[ci]) % dep) * SLOT;  // (my ring)
        auto& cn = conn_of(NL > 1 ? (o + me + home) % L : 0);
        for (uint32_t pc = 0; pc < P; ++pc) {
            const uint32_t pb = (pc + 1 == P ? row_bytes - pc * PD : PD) + 32;
            auto* h = hdr_at(home, sent % NH);
            h->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(ACC_X(home, o), ACC_Y(home, o), dst + pc * PS)}, pb);
            const uint32_t t0 = TSX();
            cn.wait_for_empty_write_slot();
            p_slot += TSX() - t0;
            cn.send_current_slot_non_blocking(src + pc * PS, pb, reinterpret_cast<uint32_t>(h));
            ++sent;
            ++unflushed;
        }
        used[ci] = u + 1;
        if constexpr (__TPROF__) {
            t_last = get_timestamp_32b();
            if (!t_first) t_first = t_last | 1;
        }
        if constexpr (FILLY) {  // (probe SPC_FILLY: the Galaxy's Y-link load - filler after my Y-bound rows, to the
            // Y neighbour's sink: FILLY / 256 bytes a row byte, on this row's connection)
            if (home / FC_ != me / FC_) {
                fill_debt += row_bytes * FILLY / 256;
                const uint32_t nb = ((me / FC_ + 1) % (N / FC_)) * FC_ + me % FC_;
                while (fill_debt >= PD) {
                    if (unflushed >= NH) flush();
                    auto* h = hdr_at(nb, sent % NH);
                    h->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(my_x[noc_index], my_y[noc_index], FILL_SINK)}, PD);
                    cn.wait_for_empty_write_slot();
                    cn.send_current_slot_non_blocking(src, PD, reinterpret_cast<uint32_t>(h));
                    ++sent;
                    ++unflushed;
                    fill_debt -= PD;
                }
            }
        }
        return true;
    };
    // the shared-expert rows: reads into free staging slots (a slot is free once its send is flushed), a batch ready
    // once every read of this RISC landed, then sent while their rings take them
    auto se_step = [&](bool& did, bool& blk, bool& sent_row) {
        if constexpr (SE) {
            if (se_ready < se_rd && ncrisc_noc_reads_flushed(noc_index)) se_ready = se_rd;
            while (se_rd < se_n && se_rd - se_free < NSE) {
                uint32_t row = 0, hdr = 0;
                while (se_oi < se_no) {  // (the next of my rows for this port: its owners in order, tokens on my link)
                    const uint32_t h = se_home[se_oi];
                    bool got = false;
                    while (se_t < se_S) {
                        const uint32_t t = se_t++;
                        if (NL > 1 || (t % A + me + h) % L == lnk) {
                            row = se_blk[se_oi] * se_S + t;
                            hdr = h | (t << 8) | (0xFFu << 24);
                            got = true;
                            break;
                        }
                    }
                    if (got) break;
                    ++se_oi;
                    se_t = 0;
                }
                const uint32_t sl = se_stage + (se_rd % NSE) * ISLOT;
                if constexpr (!__SENOREAD__) noc_async_read(get_noc_addr(row, se_g), sl, row_bytes);
                volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl + LASTPB);
                tr[0] = hdr;
                tr[1] = 0x3F80;  // (gate weight 1.0, bf16 bits)
                tr[2] = tr[4] = tr[5] = tr[6] = 0;
                tr[3] = 1;
                tr[7] = 1;
                ++se_rd;
                did = true;
            }
            while (se_sent < se_ready && unflushed + P <= NH) {
                if (!send_row(se_stage + (se_sent % NSE) * ISLOT)) {
                    ++n_blocked;
                    blk = true;
                    break;
                }
                ++se_sent;
                did = sent_row = true;
            }
        } else {
            (void)did, (void)blk, (void)sent_row;
        }
    };
    const uint32_t tstart = get_timestamp_32b();

    while (inj_done < n_inj || returned - ret0 < n_ret || fused_more() || lay_pend || se_sent < se_n) {
        bool did = false, blk = false, sent_row = false;
        const uint32_t it0 = TSX();
        const uint32_t c0 = p_cred;
        ++n_iter;
        if constexpr (__HW__) {  // (hang words: MISC 480.. every 512 loops - a store a loop hid the hang before)
            if ((n_iter & 511) == 1) {
                misc[480] = 0x48570000u | (calls_ & 0xFFFF), misc[481] = n_iter, misc[482] = inj_done;
                misc[483] = n_inj, misc[484] = returned - ret0, misc[485] = n_ret, misc[486] = fused_more();
                misc[487] = lay_pend, misc[488] = se_sent, misc[489] = se_n, misc[490] = any_ret();
                misc[491] = unflushed, misc[492] = n_blocked, misc[494] = me | lnk << 8;
            }
        }
        invalidate_l1_cache();
        hand_off();
        write_out();
        const uint32_t tot = any_ret();
        const bool pending = tot != returned;
        // (SPC_GBGATE: no source reaches CRED_THR unless the total pending does - skip the per-source scan)
        const bool scan = !__GBGATE__ || SHARED || CREDFIN || CRED_BUSY || tot - returned >= CRED_THR;
        if (pending && scan && unflushed < NH) give_back(false);
        if (lay_pend) send_layouts();
        p_book += TSX() - it0 - (p_cred - c0);
        se_step(did, blk, sent_row);
        if constexpr (LA > 1) {  // rows in queue order, but past a blocked row: its ring (and that ring's later rows) waits
            uint32_t bci[LA], nb = 0;
            const uint32_t ilim = get_local_cb_interface(cb_inj).fifo_limit;
            for (uint32_t sub = FAR ? 0 : 1; sub < 2; ++sub)  // (FAR: the far rows of the window first, then all)
            for (uint32_t j = 0; j < LA && unflushed + P <= NH && inj_done < n_inj; ++j) {
                if (!cb_pages_available_at_front(cb_inj, pend_inj + j + 1)) break;
                if (ahead >> j & 1) continue;
                uint32_t src = get_read_ptr(cb_inj) + (pend_inj + j) * ISLOT;
                if (src >= ilim) src -= get_local_cb_interface(cb_inj).fifo_size;
                const uint32_t hw_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + LASTPB)[0];
                if (sub == 0 && hops[hw_ & 0xFF] < FAR) continue;
                const uint32_t ci = (hw_ & 0xFF) * AP + ((hw_ >> 8) & 0xFFFF) % A;
                bool skip = false;
                for (uint32_t q = 0; q < nb && !skip; ++q) skip = bci[q] == ci;
                if (skip) continue;
                const uint32_t ts0 = TSX();
                if (!send_row(src)) {
                    bci[nb++] = ci;
                    ++n_blocked;
                    blk = true;
                    continue;
                }
                ahead |= 1ull << j;
                ++inj_done;
                did = sent_row = true;
                p_send += TSX() - ts0;
            }
            while (ahead & 1) {  // (the sent prefix: popped at the next flush)
                ++pend_inj;
                ahead >>= 1;
            }
        }
        while (LA <= 1 && unflushed + P <= NH && inj_done < n_inj && cb_pages_available_at_front(cb_inj, pend_inj + 1)) {
            if (PACE && get_timestamp_32b() - tstart < inj_done * pace) break;
            uint32_t src = get_read_ptr(cb_inj) + pend_inj * ISLOT;
            const uint32_t ilim = get_local_cb_interface(cb_inj).fifo_limit;
            if (src >= ilim) src -= get_local_cb_interface(cb_inj).fifo_size;
            const uint32_t ts0 = TSX();
            if (!send_row(src)) {  // (its ring's layout not here yet, or the ring full)
                if constexpr (__HW__) {  // (the blocked row: its header word, its ring's layout word and used / credit)
                    const uint32_t hw_h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + LASTPB)[0];
                    const uint32_t hw_ci = (hw_h & 0xFF) * AP + ((hw_h >> 8) & 0xFFFF) % A;
                    misc[493] = hw_h, misc[495] = hw_ci < N * AP ? lay_r[hw_ci] : 0xDEADu;
                    misc[479] = hw_ci < N * AP ? used[hw_ci] : 0, misc[478] = hw_ci < N * AP ? cred[hw_ci] : 0;
                }
                ++n_blocked;
                blk = true;
                break;
            }
            ++pend_inj;
            ++inj_done;
            did = sent_row = true;
            p_send += TSX() - ts0;
        }
        if (unflushed >= NH || (!did && unflushed)) {
            flush();
            did = true;
        }
        if (!did && pending) {
            give_back(true);
            did = true;
        }
        if (blk && !sent_row) p_blk += TSX() - it0;
        if (!did) {
            p_idle += TSX() - it0;
            ++n_empty;
        }
    }
    flush();
    const uint32_t t_loop = get_timestamp_32b();
    misc[1] = returned;
#if __FUSED__
    misc[101] = hbase + n_inj;
    noc_async_write_barrier();
#endif
    // drain: every row this port injected is consumed (its credit snapshot landed), so nothing of this call reaches
    // this core after the program ends (the arena may be per call: its L1 is another op's after this)
    for (bool open_ = __DRAIN__; open_;) {
        invalidate_l1_cache();
        open_ = false;
        for (uint32_t ci = 0; ci < N * AP; ++ci) open_ |= cred_of(ci) != used[ci];
    }
    if constexpr (__TPROF__) {
        const uint32_t tdbg = get_arg_val<uint32_t>(a_dbg + 2), tpage = get_arg_val<uint32_t>(a_dbg + 3);
        volatile tt_l1_ptr uint32_t* d = misc + 192;
        d[0] = 0x7E100000 | tpage; d[1] = p0; d[2] = get_timestamp_32b(); d[3] = p_slot; d[4] = p_idle; d[5] = p_cred;
        d[6] = n_blocked; d[7] = n_inj; d[8] = p_send; d[9] = n_empty; d[10] = p_flush; d[11] = n_credpk;
        d[12] = p_book; d[13] = p_blk; d[14] = n_iter; d[15] = n_piggy;
        d[16] = t_first; d[17] = t_last; d[18] = n_bl; d[19] = n_bf; d[20] = t_hdr; d[21] = t_bar; d[22] = t_k0; d[23] = t_loop;
        const InterleavedAddrGen<true> dg = {.bank_base_address = tdbg, .page_size = 512};
        noc_async_write(area + MISC + 192 * 4, get_noc_addr(tpage, dg), 96);
        noc_async_write_barrier();
    }
    if constexpr (PROF) {
        const uint32_t dbg = get_arg_val<uint32_t>(a_dbg), pidx = get_arg_val<uint32_t>(a_dbg + 1);
        volatile tt_l1_ptr uint32_t* d = misc + 192;
        d[0] = 0xDB900000 | pidx; d[1] = TSX() - p0; d[2] = p_idle; d[3] = p_cred; d[4] = p_slot;
        d[5] = p_flush; d[6] = n_credpk; d[7] = n_blocked; d[8] = n_empty; d[9] = n_inj; d[10] = n_ret; d[11] = p_send;
        d[12] = d[13] = d[14] = d[15] = 0;
        const InterleavedAddrGen<true> dg = {.bank_base_address = dbg, .page_size = 64};
        noc_async_write(area + MISC + 192 * 4, get_noc_addr(pidx, dg), 64);
        noc_async_write_barrier();
    }
    conn.close();
#if __NL__ > 1
    conn1.close();
#endif
    noc_async_atomic_barrier();
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
constexpr uint32_t PSL = get_compile_time_arg_val(5);  // landing pool slots (one ring per source, carved per call)
constexpr uint32_t A = get_compile_time_arg_val(6);
constexpr uint32_t AP = (A + 3) / 4 * 4;
constexpr uint32_t NT = get_compile_time_arg_val(7);
constexpr uint32_t N = get_compile_time_arg_val(8);
constexpr uint32_t L = get_compile_time_arg_val(9);
constexpr uint32_t RET = __RET__, MISC = __MISC__, RINGS = __RINGS__, LAYO = __LAYO__;
// landing "shared" (SHARED): no rings - the pool's slots granted to sources by id (G grants outstanding a source at most),
// each grant written into the home port's grant ring (GIDH: per (source, accumulator) G words, id | grant seq << 24)
// and counted (RET / MRTOT, as the credits are with rings): the port forwards them in its snapshots
constexpr uint32_t SHARED = __SHARED__, G = __G__, GIDH = __GIDH__;
#define GTAG(q) (0x80u | ((q) & 0x7Fu))  // (a grant's tag: never 0 - a zeroed ring entry is no grant)
constexpr uint32_t PROF = __PROF__;
constexpr uint32_t ABL = __ABL__;
constexpr uint32_t P = __P__, PD = __PD__, PS = __PS__;
constexpr uint32_t LASTPB = (P - 1) * PS + (row_bytes - (P - 1) * PD);
constexpr auto y_args = TensorAccessorArgs<10>();
constexpr auto t_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
// Accumulator reader: local rows from DRAM, then its N - 1 rings (one per source chip, each up to this call's expected
// count); a consumed slot is one credit, counted on the port that faces the row's source on this accumulator's link to
// it (that port sends the credits back).
void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tab_bytes = get_arg_val<uint32_t>(a++), off = get_arg_val<uint32_t>(a++);
    uint32_t n_local = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t cx = get_arg_val<uint32_t>(a++), cy = get_arg_val<uint32_t>(a++);
    const uint32_t counts_addr = get_arg_val<uint32_t>(a++), W = get_arg_val<uint32_t>(a++);
    const uint32_t hc = get_arg_val<uint32_t>(a++);  // this chip's index (its own ring is not there)
    uint32_t want[N], px[N], py[N];
    for (uint32_t k = 0; k < N; ++k) {
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
        // (the args' want: static rows on top of the plan's, e.g. the row-mates' shared-expert partials)
        for (uint32_t k = 0; k < N; ++k) want[k] = k == hc ? 0 : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc + 64)[k] + want[k];
    }
    uint32_t n_rows = n_local;
    for (uint32_t k = 0; k < N; ++k) n_rows += want[k];
    // shared_rows: the chip's own tokens' shared-expert partial (rows 0 .. SOWN of the partial) is added here as one
    // more row each (weight 1), between ring polls
    constexpr uint32_t SOWN = __SOWN__;
    const InterleavedAddrGen<true> own_g = {.bank_base_address = __OWNADDR__, .page_size = row_bytes};
    uint32_t oi = me;
    if constexpr (SOWN) n_rows += me < SOWN ? (SOWN - me + A - 1) / A : 0;
    constexpr uint32_t TCH = 256;
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* taken = misc + __MTAKEN__;  // per ring (source chip), cumulative
    // the landing pool, carved for this call: every source with rows here gets one slot, the rest in proportion to its
    // rows (at most its rows); each source's ring (tag << 24 | base << 12 | depth) goes to the home port that returns
    // its credits, which forwards it to the source (the source's rows wait for it)
    const uint32_t calls_ = misc[105] + 1;
    misc[105] = calls_;
    const uint32_t tag = (calls_ - 1) % 255 + 1;
    uint32_t rbase[N], rdep[N], t0[N];
    {
        uint32_t nz = 0, tw = 0;
        for (uint32_t k = 0; k < N; ++k) {
            if (k != hc && want[k]) {
                ++nz;
                tw += want[k];
            }
        }
        // half the extra slots in equal shares, half in proportion to the rows (all proportional deepened the hot
        // rings past their links' rate and starved the rest: Zipf-spread combine +14%); largest remainders hand out
        // every slot (floors alone left near-equal shares at one slot each)
        const uint32_t eq = nz ? (PSL - nz) / 2 / nz : 0, extra = PSL - nz - eq * nz;
        uint32_t rem[N], left = extra;
        for (uint32_t k = 0; k < N; ++k) {
            rdep[k] = 0;
            rem[k] = 0;
            if (k != hc && want[k]) {
                const uint64_t q = static_cast<uint64_t>(extra) * want[k];
                rdep[k] = 1 + eq + static_cast<uint32_t>(q / tw);
                rem[k] = static_cast<uint32_t>(q % tw);
                left -= rdep[k] - 1 - eq;
            }
        }
        for (; left; --left) {
            uint32_t bk = N;
            for (uint32_t k = 0; k < N; ++k) {
                if (rdep[k] && (bk == N || rem[k] > rem[bk])) bk = k;
            }
            if (bk == N) break;
            ++rdep[bk];
            rem[bk] = 0;
        }
        uint32_t b = 0;
        for (uint32_t k = 0; k < N; ++k) {
            uint32_t d = rdep[k] < want[k] ? rdep[k] : want[k];  // (no more slots than rows)
            if constexpr (SHARED) d = d < G ? d : G;  // (the first grants: the carve's shares, G at most)
            rbase[k] = b;
            rdep[k] = d;
            b += d;
            t0[k] = taken[k];
            if constexpr (!SHARED) {
                if (k != hc && px[k]) noc_inline_dw_write(get_noc_addr(px[k], py[k], area + LAYO + 4 * (k * AP + me)), tag << 24 | rbase[k] << 12 | d);
            }
        }
    }
    // shared: slot ownership (source, 0xFF free), grants outstanding / issued this call a source, the free list
    uint8_t own[SHARED ? PSL : 1];
    uint32_t gout[N], giss[N], fstack[SHARED ? PSL : 1], nfree = 0;
    auto grant = [&](uint32_t k, uint32_t slot) {  // (the id into the home port's grant ring, then its count)
        const uint32_t seq = t0[k] + giss[k];
        noc_inline_dw_write(
            get_noc_addr(px[k], py[k], area + GIDH + 4 * ((k * AP + me) * G + seq % G)), slot | GTAG(seq) << 24);
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + RET + 4 * (k * AP + me)), 1);
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + MISC + 4 * (__MRTOT__ + k)), 1);
        own[slot] = static_cast<uint8_t>(k);
        ++giss[k];
        ++gout[k];
    };
    if constexpr (SHARED) {
        for (uint32_t i = 0; i < PSL; ++i) own[i] = 0xFF;
        for (uint32_t k = 0; k < N; ++k) gout[k] = giss[k] = 0;
        for (uint32_t k = 0; k < N; ++k) {
            if (k == hc || !px[k]) continue;
            for (uint32_t j = 0; j < rdep[k]; ++j) grant(k, rbase[k] + j);
        }
        for (uint32_t i = PSL; i-- > 0;) {
            if (own[i] == 0xFF) fstack[nfree++] = i;
        }
    }
    // a freed slot to the source with rows still to grant (want - outstanding), fewest outstanding first
    auto regrant = [&]() {
        while (nfree) {
            uint32_t bk = N;
            for (uint32_t k = 0; k < N; ++k) {
                if (k == hc || gout[k] >= G || want[k] <= gout[k]) continue;
                if (bk == N || gout[k] < gout[bk] || (gout[k] == gout[bk] && want[k] - gout[k] > want[bk] - gout[bk])) bk = k;
            }
            if (bk == N) return;
            grant(bk, fstack[--nfree]);
        }
    };
    if constexpr (SHARED) regrant();  // (the slots the first grants left: to whoever has rows still to grant)
    auto ring_slot = [&](uint32_t k, uint32_t tk) { return area + RINGS + (rbase[k] + (tk - t0[k]) % rdep[k]) * SLOT; };
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
    volatile tt_l1_ptr uint32_t* landed = misc + __ALAND__;  // (128, or 160 when ports share the core)
    const uint32_t lbase = L1M ? misc[__ALBASE__] : 0;
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
    auto take_local = [&]() -> bool {  // the next local row (L1 hand-off ring), if every writer landed it
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < WL; ++w) {
            const uint32_t v = landed[w] - lbase;
            m = v < m ? v : m;
        }
        if (li >= m) return false;
        const uint32_t sl = LB + (li % QL) * SLOT;
        push_pieces(sl, (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl + LASTPB)[0] >> 8) & 0xFFFF);
        ++li;
        ++done;
        const uint32_t g = li + QL < n_local ? li + QL : n_local;
        // (grants in batches of up to 4 - at most the ring: a batch larger than QL never fills, and the writers stall)
        if (g > lgranted && (g - lgranted >= (QL < 4 ? QL : 4) || g == n_local)) lgrant_to(g);
        return true;
    };
    auto peek_local = [&]() -> uint32_t {  // the next local row's token, if every writer landed it (else ~0)
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < WL; ++w) {
            const uint32_t v = landed[w] - lbase;
            m = v < m ? v : m;
        }
        if (li >= m) return 0xFFFFFFFF;
        return (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LB + (li % QL) * SLOT + LASTPB)[0] >> 8) & 0xFFFF;
    };
    auto peek_ring = [&](uint32_t k) -> uint32_t {  // source k's next row's token, if all its pieces landed (else ~0)
        const uint32_t s = ring_slot(k, taken[k]);
        for (uint32_t pc = 0; pc < P; ++pc) {
            volatile tt_l1_ptr uint32_t* tp =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + pc * PS + (pc + 1 == P ? row_bytes - pc * PD : PD));
            if (tp[3] == 0 || tp[7] == 0) return 0xFFFFFFFF;
        }
        return (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB)[0] >> 8) & 0xFFFF;
    };
    auto take_ring = [&](uint32_t k) -> bool {  // the next row of source k's ring, if all its pieces landed
        const uint32_t tk = taken[k];
        const uint32_t s = ring_slot(k, tk);
        for (uint32_t pc = 0; pc < P; ++pc) {
            volatile tt_l1_ptr uint32_t* tp =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + pc * PS + (pc + 1 == P ? row_bytes - pc * PD : PD));
            if (tp[3] == 0 || tp[7] == 0) return false;
        }
        volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB);
        if constexpr (__PIGGY__) {  // (credits for my chip's rows to k: to my port facing k on this link class)
            if (tr[1] & 0x10000u) {
                volatile tt_l1_ptr uint32_t* sc = misc + 116;
                sc[0] = tr[2];
                sc[1] = tr[4];
                sc[2] = tr[5];
                sc[3] = tr[6];
                noc_async_write(
                    reinterpret_cast<uint32_t>(sc), get_noc_addr(px[k], py[k], area + __PIG__ + 16 * (k * 8 + me / L)), 16);
                noc_async_writes_flushed();
            }
        }
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
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + RET + 4 * (k * AP + me)), 1);
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + MISC + 4 * (__MRTOT__ + k)), 1);
        return true;
    };
    auto take_slot = [&](uint32_t i) -> bool {  // shared: slot i's row, if all its pieces landed
        const uint32_t s = area + RINGS + i * SLOT;
        for (uint32_t pc = 0; pc < P; ++pc) {
            volatile tt_l1_ptr uint32_t* tp =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + pc * PS + (pc + 1 == P ? row_bytes - pc * PD : PD));
            if (tp[3] == 0 || tp[7] == 0) return false;
        }
        volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB);
        push_pieces(s, (tr[0] >> 8) & 0xFFFF);
        for (uint32_t pc = 0; pc < P; ++pc) {
            volatile tt_l1_ptr uint32_t* tp =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + pc * PS + (pc + 1 == P ? row_bytes - pc * PD : PD));
            tp[3] = 0;
            tp[7] = 0;
        }
        const uint32_t k = own[i];
        own[i] = 0xFF;
        fstack[nfree++] = i;
        ++taken[k];
        --want[k];
        --gout[k];
        ++done;
        return true;
    };
    // Deterministic order (__DET__, with the L1 hand-off and the device plan): each home token's rows are added in
    // the planner's per-token source order (combine_plan det); a ring holds its source's rows in production order,
    // so a ring's head row is taken once it is its token's next. A token's sum no longer depends on arrival timing,
    // and a ring waits only on its head token's earlier rows (not on a global step). Deadlock-free: a token's order is
    // by (predicted finish on the source, source), nondecreasing along every source's production, so the row with
    // the lowest key not yet taken is always takeable once it lands. A table whose per-source totals disagree with
    // want / n_local is not used (the polling loop below takes everything; misc[102] flags it).
    if constexpr (__DET__ && L1M) {
        constexpr uint32_t DK = __DK__, NTOK = __DNTOK__;
        const uint32_t ol = (tl + 256 + 63) & ~63u;  // (the table CB: unused with the hand-off's device plan)
        const InterleavedAddrGen<true> og = {.bank_base_address = get_arg_val<uint32_t>(a + 9 + 2 * WL),
                                             .page_size = __DPG__};
        noc_async_read(get_noc_addr(me, og), ol, (NTOK * DK + 63) / 64 * 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint8_t* ord = reinterpret_cast<const uint8_t*>(ol);
        uint32_t tot[N];
        for (uint32_t k = 0; k < N; ++k) tot[k] = 0;
        for (uint32_t i = 0; i < NTOK * DK; ++i) {
            if (ord[i] < N) ++tot[ord[i]];
        }
        bool ok = true;
        for (uint32_t k = 0; k < N; ++k) ok = ok && tot[k] == (k == hc ? n_local : want[k]);
        misc[102] = ok ? 0 : 0xBAD0DE7;
        uint8_t pos[NTOK];
        for (uint32_t j = 0; j < NTOK; ++j) pos[j] = 0;
        auto next_is = [&](uint32_t t, uint32_t k) -> bool {
            const uint32_t j = t / A;
            return t != 0xFFFFFFFF && pos[j] < DK && ord[j * DK + pos[j]] == k;
        };
        while (ok && done < n_rows) {
            invalidate_l1_cache();
            for (uint32_t t = 0; li < n_local && next_is(t = peek_local(), hc);) {
                take_local();
                ++pos[t / A];
            }
            for (uint32_t k = 0; k < N; ++k) {
                if (k == hc) continue;
                for (uint32_t t = 0; want[k] && next_is(t = peek_ring(k), k);) {
                    take_ring(k);
                    ++pos[t / A];
                }
            }
        }
    }
    uint32_t hw_it = 0;
    while (done < n_rows) {
        invalidate_l1_cache();
        if constexpr (__HW__) {  // (hang words: MISC 496..)
            if ((++hw_it & 511) == 1) {
                misc[496] = 0x41430000u | (calls_ & 0xFFFF), misc[497] = done, misc[498] = n_rows;
                misc[499] = li, misc[500] = n_local, misc[501] = me | hc << 8, misc[502] = hw_it;
                uint32_t kw = 0;  // (the source with the most rows still wanted: its ring's head slot)
                for (uint32_t k = 0; k < N && k < 8; ++k) {
                    misc[503 + k] = (taken[k] - t0[k]) | want[k] << 16;
                    if (want[k] > want[kw]) kw = k;
                }
                const uint32_t hs = rdep[kw] ? ring_slot(kw, taken[kw]) : 0;
                uint32_t fl = 0;
                for (uint32_t pc = 0; hs && pc < P && pc < 8; ++pc) {
                    volatile tt_l1_ptr uint32_t* tp =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hs + pc * PS + (pc + 1 == P ? row_bytes - pc * PD : PD));
                    fl |= (tp[3] != 0 ? 1u : 0u) << (2 * pc) | (tp[7] != 0 ? 2u : 0u) << (2 * pc);
                }
                misc[476] = kw | rdep[kw] << 8 | rbase[kw] << 16, misc[477] = fl;
            }
        }
        if (li == n_local && t_local == 0) t_local = get_timestamp_32b();
        while (L1M && li < n_local) {
            if (!take_local()) break;
        }
        while (!L1M && li < n_local) {
            const uint32_t* en = lentry(li);
            if (!expert_done(en[1] >> 24)) break;
            push(__NOYREAD__ ? get_noc_addr(tl) : y.get_noc_addr(en[0]), (en[1] >> 8) & 0xFFFF);  // (probe: no y reads)
            ++li;
            ++done;
        }
        if constexpr (SHARED) {  // (every granted slot, any order; freed slots regranted at once)
            bool any = false;
            for (uint32_t i = 0; i < PSL; ++i) {
                if (own[i] != 0xFF) any = take_slot(i) || any;
            }
            if (any) regrant();
        } else {
            for (uint32_t k = 0; k < N; ++k) {
                if (want[k]) {
                    take_ring(k);
                    for (uint32_t b = 1; b < __BK__; ++b) (void)peek_ring(k);  // (probe SPC_BK: bk rings polled a source)
                }
            }
        }
        if (SOWN && oi < SOWN) {
            push(get_noc_addr(oi, own_g), oi);
            oi += A;
            ++done;
        }
    }
    if constexpr (__DEVPLAN__) {  // (the compute's end marker: it does not know the row count)
        cb_reserve_back(cb_meta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = 0xFFFFFFFF;
        cb_push_back(cb_meta, 1);
    }
    const uint32_t p_total = get_timestamp_32b() - p0;
    if (L1M) misc[__ALBASE__] = lbase + n_local;
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


def first_hop(src, dst, R, C, wrap_x, wrap_y, ytie=False):
    h = route(src, dst, R, C, wrap_x, wrap_y, 0, ytie)
    return h[0][1] if h else None


A_MAX = 16  # accumulators at most (the state layout, the device planner)
WRITER_WORDS_ROWS = 192  # the L1 hand-off writers' words (grants, grant bases, table page, trailer stages): 64 B rows


def fused_arena_reserve(flat_static_bytes, *, l1=True):
    """L1 a core leaves below the fused flat + combine arena: the flat program's static CBs (its plan's
    static_l1_bytes, checked by its program factory; the combine's CBs all sit in the arena) and, with the L1 hand-off,
    the writers' words tensor allocated after the arena (WRITER_WORDS_ROWS x 64 B), plus 2 KB for the allocator's
    alignment. Programs that run while the arena is allocated must fit their static CBs in it."""
    return (flat_static_bytes + 63) // 64 * 64 + (WRITER_WORDS_ROWS * 64 if l1 else 0) + 2048


def combine_l1_layout(
    N,
    S,
    W,
    A,
    *,
    arena_bytes,
    l1=True,
    acc_on_ports=False,
    inj_slots=64,
    depth=8,
    piece_max=None,
    ring_depth=None,
    pool_slots=None,
    det_k=0,
    det_pg=0,
    shared_rows=False,
    local_keep=8,
    landing="rings",
):
    """What the combine puts in each core's arena (bytes), as sparse_combine_rf_prepare lays it out: N chips, S home
    tokens a chip, rows W wide (bf16), A accumulators. Pure (no device): the planner's sizing and its reports use it.
    Returns dict(slot, PSL (landing pool slots per accumulator), D (= PSL / (N - 1)), QP (a port's injection ring),
    QL (an accumulator's local ring), acc_used / port_used (bytes of the arena), fits, why)."""
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    row_bytes = W * 2
    NT = (row_bytes + TILE - 1) // TILE
    PIECE_MAX_ = piece_max or PIECE_MAX
    NP = (row_bytes + PIECE_MAX_ - 1) // PIECE_MAX_
    PD = ((row_bytes + NP - 1) // NP + 63) // 64 * 64
    SLOT = ISLOT = NP * ((PD + 32 + 63) // 64) * 64
    FUSED = bool(acc_on_ports)
    RINGS = state_layout(N, A, GRANTS if landing == "shared" else 0)[6]
    MAXT = (S + A - 1) // A
    tab_cb = max(256 * 8 + 128, 256 + 64 + (MAXT * det_k + 63) // 64 * 64) if det_k else 256 * 8 + 128
    acc_cb_list = [tab_cb, 2 * NT * TILE, 2 * 32, MAXT * NT * 4096, 2 * NT * TILE] + ([2 * TILE] if l1 else [])
    acc_cbs = sum(al(v) for v in acc_cb_list)
    out = dict(slot=SLOT, acc_fp32=MAXT * NT * 4096, acc_cbs=acc_cbs, rings=RINGS, fits=True, why="")
    if l1 and NP != 1:  # (sparse_combine_rf_prepare asserts it: the hand-off grants one slot a row, one packet)
        return {
            **out,
            "fits": False,
            "why": f"the L1 hand-off takes one piece a row: piece {PIECE_MAX_} < {row_bytes} B",
        }
    QP = inj_slots
    keep = (local_keep + min(32, QP)) * ISLOT if FUSED else (local_keep * ISLOT if l1 else 0)
    fit = (arena_bytes - RINGS - acc_cbs - keep) // SLOT
    PSL = min(depth * (N - 1), fit)
    PSL = PSL if ring_depth is None else int(ring_depth) * (N - 1)
    PSL = PSL if pool_slots is None else int(pool_slots)
    NSE = 8 if shared_rows else 0
    out.update(PSL=PSL, D=PSL // (N - 1) if N > 1 else 0)
    if PSL < N - 1:
        return {**out, "fits": False, "why": f"landing pool {PSL} < {N - 1} slots (one per source chip)"}
    if PSL >= 4096:
        return {**out, "fits": False, "why": "landing pool >= 4096 slots (12-bit ring fields)"}
    acc_area = RINGS + PSL * SLOT
    a_end = al(acc_area) + acc_cbs
    if FUSED:
        QL = min(16, (arena_bytes - a_end - min(32, QP) * ISLOT) // ISLOT)
        QP = min(QP, (arena_bytes - a_end - QL * ISLOT) // ISLOT - NSE)
        if QL < 2 or QP < 8:
            return {**out, "fits": False, "why": f"acc_on_ports: local ring {QL} / injection ring {QP} slots"}
        acc_used = a_end + QL * ISLOT + (QP + NSE) * ISLOT
        port_used = acc_used
    else:
        QL = min(16, (arena_bytes - a_end) // ISLOT) if l1 else 0
        if l1 and QL < 2:
            return {**out, "fits": False, "why": f"no room for the accumulators' local ring ({QL} slots)"}
        acc_used = a_end + QL * ISLOT
        port_used = al(RINGS + ((QP + NSE) * ISLOT if l1 else 0)) + tab_cb
        if port_used > arena_bytes:
            return {**out, "fits": False, "why": f"port: {port_used >> 10} KB > the arena's {arena_bytes >> 10} KB"}
    return {**out, "QP": QP, "QL": QL, "acc_used": acc_used, "port_used": port_used}


def combine_auto(
    N,
    S,
    W,
    n_cells,
    port_cells,
    *,
    arena_bytes,
    l1=True,
    depth=8,
    piece_max=None,
    a_max=A_MAX,
    mode=None,
    landing="rings",
):
    """The combine's configuration from what the flat plan leaves (n_cells free cells per chip, port_cells of them
    the ports' - the union over the chips) and the arena: ports on their own cells with A = n_cells - port_cells
    accumulators (at most a_max), or the accumulators on the ports' cells (acc_on_ports: A = n_cells, at most a_max) -
    whichever leaves the deeper landing pool (ties: separate ports - accumulators on the ports cost GLM-5.3 1.5% at the
    same A, and more of them at the same depth 2.4%: FLAT_COMBINE_OVERLAP_WORKLOG.md 2026-10-22). Every injection ring
    (64 -> 8 slots) and kept local ring (8 or 2 slots) that fits is tried: the deepest landing pool wins (ring depth D,
    the pool's slots per source chip), ties to the deeper injection ring, then the larger kept local ring. Returns dict(acc_on_ports, n_acc,
    inj_slots, depth, local_keep, layout = combine_l1_layout's, modes = each mode's best) or, when nothing fits,
    dict(fits=False, why=...). mode "ports apart" / "acc_on_ports": that mode only (A/B runs)."""
    best, why, modes = None, [], {}
    score = lambda c: (c["layout"]["D"], c["inj_slots"], c["local_keep"], c["layout"]["PSL"])  # noqa: E731
    for fused in (False, True):
        A = min(a_max, n_cells if fused else n_cells - port_cells)
        name = "acc_on_ports" if fused else "ports apart"
        if mode is not None and mode != name:
            continue
        if A < 1 or (fused and port_cells > A):
            why.append(f"{name}: {A} accumulator cells")
            continue
        mode_best, last = None, None
        for keep in (8, 2):
            for qp in (64, 32, 16, 8):
                lay = combine_l1_layout(
                    N, S, W, A, arena_bytes=arena_bytes, l1=l1, acc_on_ports=fused, inj_slots=qp, depth=depth,
                    piece_max=piece_max, local_keep=keep, landing=landing,
                )  # fmt: skip
                if not lay["fits"]:
                    last = lay
                    continue
                cand = dict(acc_on_ports=fused, n_acc=A, inj_slots=qp, depth=depth, local_keep=keep, layout=lay)
                if mode_best is None or score(cand) > score(mode_best):
                    mode_best = cand
        if mode_best is None:
            why.append(f"{name} A {A}: {last['why']} at the shallowest rings")
            continue
        modes[name] = mode_best
        if best is None or mode_best["layout"]["D"] > best["layout"]["D"]:
            best = mode_best
    return {**best, "fits": True, "modes": modes} if best else dict(fits=False, why="; ".join(why), modes=modes)


def _snap_bits(q):
    """SNAP: the fp32 bits of C = 1.5 x 2^23 q ((v + C) - C rounds v to a multiple of q for |v| < 2^22 q), 0 off."""
    if not q:
        return 0
    e = math.log2(q)
    assert e == int(e), f"snap_q {q}: a power of two"
    return struct.unpack("<I", struct.pack("<f", 1.5 * 2.0 ** (e + 23)))[0]


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
    deterministic=False,
    ports_per_direction=False,
    acc_on_ports=False,
    piece_max=None,
    inj_slots=64,
    ring_depth=None,
    probes=None,
    shared_rows=None,
    barrier=None,
    dispatch=None,
    piggyback_credits=True,
    pool_slots=None,
    local_keep=8,
    fill_sink=None,
    landing="rings",
    snap_q=None,
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
    the plan, rows_by_chip only places the cores. deterministic (with l1 and a devplan holding the planner's "dord"
    order table, combine_plan det=): every accumulator takes its rows in a fixed (source schedule entry, source chip)
    order, so the sums are bit-identical from call to call. snap_q (a power of two): every term rounded to a multiple of
    q before its add (_ACC_COMPUTE_W SNAP) - bit-identical sums in any order while every partial sum is below 2^24 q,
    no fixed order needed. ports_per_direction (with l1 and a devplan from
    combine_plan port_luts(per_direction=True)): one port core per direction, its sender driving every link of that
    direction (a row goes out on its link (accumulator + source + home) % L; credit snapshots on link 0, so they stay in
    order): half the port cells with two links. acc_on_ports (with l1): the ports sit on accumulator cells (the A
    accumulator cells are the same on every chip; each chip's ports take the ones nearest its Ethernet cores), each
    port core also an accumulator - its sender RISC also runs the hand-off grants and the output writes - so the
    combine takes max(ports, A) cells instead of ports + A. piece_max: row bytes per fabric packet at most (one piece a
    row with the L1 hand-off: >= the row, and a fabric payload >= row + 32 B). inj_slots: a port's L1 hand-off
    injection ring (slots; less leaves the arena room for deeper rings with wide rows). ring_depth: the rings' depth
    exactly (default: min(depth, what the arena leaves)). probes: CombineProbes (default: CombineProbes.from_env()).
    shared_rows (with l1): dict(tensor=, axis=1) - the shared expert's partials of the chip's group (its mesh row, axis
    1, or column, axis 0; G chips), [G S, W] bf16 row-major in DRAM, block j = the owner j places along the group
    ((r, (c + j) % C) / ((r + j) % R, c); block 0: mine, not sent): every chip sends its group-mates their blocks before
    the flat's rows, into the same accumulators with weight 1 (the split shared expert's reduce-scatter, on the
    combine's rings and credits; static counts on top of the plan's; the accumulators add block 0, the own tokens'
    partial, from DRAM). dispatch (with l1, a port per link): dict(sems, relays, words, ready_off) - the sparse 2D dispatch
    (sparse_all_gather_2d.dispatch_phase) runs first in this program's port kernels, on these ports and their fabric
    connections: build(.., disp=dict(output, plan, manifest)) - its readers then count in on the flat op's x relays'
    ready words (the flat layout's x_ready). barrier (with l1): dict(tensor=) - a persistent L1
    tensor of >= 8 B per core on every core (zeros at creation, never rewritten by the caller): the program starts with
    a cross-chip barrier, so the caller may zero the arena right before it (no other sync in between).
    piggyback_credits: rows carry their link class's consumed counts back to the chip they go to (standalone credit
    packets only when no row goes that way); on by default where it applies (a port per link, two links, A <= 16):
    GLM-Flash S640 combine alone -3.5% uniform / -4.3% spread. pool_slots: the landing pool's slots
    exactly (e.g. another build's run.pool, when two builds share the arena: equal pools, so neither's CBs sit on the
    other's ring slots - whose trailer flags mark landed rows). local_keep: the accumulators' local ring slots kept
    free before the landing pool takes the rest of the arena (>= 2; fewer leave a deeper pool)."""
    R, C = tuple(mesh_device.shape)
    N, L, A0 = R * C, num_links, n_acc
    row_bytes = W * 2
    NT = (row_bytes + TILE - 1) // TILE
    PIECE_MAX = piece_max or globals()["PIECE_MAX"]  # (piece_max: e.g. one piece a row for the L1 hand-off)
    NP = (row_bytes + PIECE_MAX - 1) // PIECE_MAX
    PD = ((row_bytes + NP - 1) // NP + 63) // 64 * 64
    PS = ((PD + 32 + 63) // 64) * 64
    SLOT = ISLOT = NP * PS
    # the L1 hand-off's y writers (se_yrm.hpp SE_Y_L1) write a row's columns at slot + column offset and its trailer
    # at slot + row bytes: one piece a row. Wider rows (DeepSeek V4.1 H 5120: 10240 B) need piece_max >= the row and a
    # fabric payload >= row + 32 B (e.g. piece_max 10240, fabric_payload_size 10880), else the combine waits for piece
    # trailers that never come (a hang)
    assert l1 is None or NP == 1, (
        f"combine L1 hand-off: {row_bytes} B rows need one piece (piece_max {PIECE_MAX} < row: pass piece_max >= "
        f"{row_bytes} and a fabric payload >= {row_bytes + 32})"
    )
    pr = probes or CombineProbes.from_env()
    # idle credit snapshots batched (at least 8 rows, or 6 us old) where there are many sources: a snapshot costs an EDM
    # slot on the row-injecting sender, so per-source packets scale with the source count - on the Galaxy's 31 (the
    # LoudBox with CombineProbes.bk 4) batching took 9-10 % off the fused call (GLM-5.3 836 -> 762 us); with the LoudBox's
    # 7 it was a wash (+-5 us), so off there
    assert (
        not pr.fill_y or fill_sink
    ), "CombineProbes.fill_y: an L1 sink (fill_sink: a packet's bytes on every port cell)"
    sources = R * C * max(int(pr.bk), 1)
    # (final credits at once: host-planned rings only - the per-source counts are the host's)
    CREDFIN = bool(pr.credfin) and devplan is None and landing != "shared" and shared_rows is None and not ports_per_direction
    cred_idle, cred_age = (8, 8100) if sources >= 16 else (1, 0)
    if pr.credidle:
        cred_idle, cred_age = int(pr.credidle), int(pr.credage)
    chips, nb, (wrap_x, wrap_y) = mesh_plan(mesh_device, num_links, torus)
    FUSED = bool(acc_on_ports)
    assert not FUSED or l1 is not None, "acc_on_ports: the L1 hand-off"
    grid = mesh_device.compute_with_storage_grid_size()
    pool = cells if cells is not None else [(x, y) for y in range(grid.y) for x in range(grid.x)]
    if FUSED:  # the accumulator cells: every chip's nearest-to-Ethernet port cells, then the pool's first others
        free_ports = place_ports(mesh_device, chips, pool)
        aset_xy = sorted({(c.x, c.y) for c in free_ports.values()}, key=pool.index)
        assert len(aset_xy) <= n_acc, f"acc_on_ports: {len(aset_xy)} port cells over the chips > {n_acc} accumulators"
        aset_xy += [c for c in pool if c not in aset_xy][: n_acc - len(aset_xy)]
        port_core = place_ports(mesh_device, chips, aset_xy)
    else:
        port_core = place_ports(mesh_device, chips, cells)
    PER_DIR = bool(ports_per_direction)
    assert not PER_DIR or all(len(p["links"]) == num_links for ports in chips.values() for p in ports.values())
    if PER_DIR:  # (one core per (chip, direction): link 0's; every link of the direction maps to it)
        port_core = {k: port_core[k[:2] + (0,)] for k in port_core}
    used_cores = {(c.x, c.y) for c in port_core.values()}
    # (per-chip roles, ports apart: each chip's accumulators are the cells less ITS OWN ports - a cell may be a port on
    # one chip and an accumulator on another, the senders address a home chip's accumulator through a table; the
    # collector (the flat op's one counts core) is a cell no chip has a port on, accumulator 0 everywhere. Else (or
    # SPC_ACC_PERCHIP=0) the same cells on every chip: the cells less every chip's ports - the union varies per Galaxy)
    PERCHIP = not FUSED and os.environ.get("SPC_ACC_PERCHIP", "1") == "1"
    acc_of = None
    if FUSED:
        acc_cores = [ttnn.CoreCoord(x, y) for x, y in aset_xy]
        used_cores = set()  # (the ports are accumulator cells)
    elif PERCHIP:
        never = [c for c in pool if c not in used_cores]
        assert never, "combine: no cell free of every chip's ports (the collector's)"
        own = {ch: {(c.x, c.y) for k, c in port_core.items() if k[0] == ch} for ch in chips}
        acc_of = {ch: ([never[0]] + [c for c in pool if c not in own[ch] and c != never[0]])[:A0] for ch in chips}
        A_ = min(len(v) for v in acc_of.values())
        acc_of = {ch: [ttnn.CoreCoord(x, y) for x, y in v[:A_]] for ch, v in acc_of.items()}
        acc_cores = acc_of[next(iter(chips))]  # (a representative: the count, the collector at index 0)
    else:
        acc_cores = [ttnn.CoreCoord(x, y) for x, y in pool if (x, y) not in used_cores][:A0]
    accs_of = (lambda ch: acc_of[ch]) if PERCHIP else (lambda ch: acc_cores)  # noqa: E731
    all_acc_xy = {(c.x, c.y) for ch in chips for c in accs_of(ch)}
    A = len(acc_cores)
    assert N <= 32 and A <= 16, (N, A)
    AP = (A + 3) // 4 * 4
    SHARED = landing == "shared"
    assert landing in ("rings", "shared"), landing
    assert not SHARED or not (deterministic or FUSED), "landing shared: the accumulators apart, nondeterministic"
    G_ = GRANTS if SHARED else 0
    CRED, USED, RET, CNT, MISC, HDRS, RINGS = state_layout(N, A, G_)
    LAYR, LAYO, U0 = state_layout_ext(N, A, G_)
    SHW, PIG = state_layout_pig(N, A)
    GIDH, GREC = state_layout_grants(N, A, G_) if SHARED else (0, 0)
    run_cells = dict(
        ports={k: (c.x, c.y) for k, c in port_core.items()},
        accs=[(c.x, c.y) for c in acc_cores],
        accs_by_chip={ch: [(c.x, c.y) for c in accs_of(ch)] for ch in chips},
    )
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    idx = lambda ch: ch[0] * C + ch[1]
    fh = lambda s, d: first_hop(s, d, R, C, wrap_x, wrap_y, pr.ytie)

    inj = {k: [] for k in port_core}
    n_ret = {k: 0 for k in port_core}
    ret_src = {k: [0] * N for k in port_core}  # (per source chip: the rows whose credits a port returns)
    loc = {(ch, i): [] for ch in chips for i in range(A)}
    want = {(ch, i): [0] * N for ch in chips for i in range(A)}
    for ch, rows in rows_by_chip.items():
        for row in rows:
            yrow, home, t = row[:3]
            assert t < (1 << 16)
            hdr = idx(home) | (t << 8) | ((row[4] if len(row) > 4 else 0) << 24)
            o = t % A
            if home == ch:
                loc[(home, o)].append((yrow, hdr))
                continue
            l = link_of(o, idx(ch), idx(home), L)  # (the row's link is fixed by (accumulator, source, home))
            inj[(ch, fh(ch, home), l)].append((yrow, hdr))
            want[(home, o)][idx(ch)] += 1
            n_ret[(home, fh(home, ch), l)] += 1
            ret_src[(home, fh(home, ch), l)][idx(ch)] += 1
    # the shared expert's partials (shared_rows): row-mate h's block from chip ch, token t -> accumulator t % A on h
    PIGGY = (pr.piggy == "1") if pr.piggy in ("0", "1") else bool(piggyback_credits)
    if pr.piggy == "fused":  # (probe: piggyback only in the L1 hand-off programs)
        PIGGY = l1 is not None
    PIGGY = PIGGY and not PER_DIR and L == 2 and A0 <= 16  # (where it applies: a port per link, two links, A <= 16)
    # (not with the deterministic order: a piggybacked row's credits count as returned when it is sent but reach the
    # port only when the home's accumulator takes the row, and that order leaves landed rows in their ring until their
    # token's turn - the row next in order can be the one waiting on those credits: a deadlock)
    PIGGY = PIGGY and not deterministic and landing != "shared"  # (shared: grants ride the snapshots only)
    # (off on the Galaxy unless forced: with 31 sources a soak hangs at ~1640 calls - a source's credits run ahead of
    # what its home consumed (its drain ends, the home still waits for its rows); SPC_PIGGY=0 soaks clean. Root cause
    # in the piggyback path not found yet - GALAXY_SCALING worklog 2026-10-26)
    PIGGY = PIGGY and (pr.piggy == "1" or R * C < 16)
    SE_ON = shared_rows is not None
    assert not SE_ON or l1 is not None, "shared_rows: the L1 hand-off"
    assert not SE_ON or arena is not None, "shared_rows: the staging slots live in the arena"
    se_n = {k: 0 for k in port_core}
    se_ret = {k: 0 for k in port_core}
    se_own = {k: [] for k in port_core}
    se_want = {(ch, i): [0] * N for ch in chips for i in range(A)}
    se_axis = int(shared_rows.get("axis", 1)) if SE_ON else 1
    se_g = C if se_axis == 1 else R
    se_owner = lambda ch, j: (ch[0], (ch[1] + j) % C) if se_axis == 1 else ((ch[0] + j) % R, ch[1])  # noqa: E731
    if SE_ON and pr.se != "off":
        for ch in chips:
            for j in range(1, se_g):
                h = se_owner(ch, j)
                for t in range(S):
                    o = t % A
                    l = link_of(o, idx(ch), idx(h), L)
                    k = (ch, fh(ch, h), l)
                    se_n[k] += 1
                    if (idx(h), j) not in se_own[k]:
                        se_own[k].append((idx(h), j))
                    se_want[(h, o)][idx(ch)] += 1
                    se_ret[(h, fh(h, ch), l)] += 1
    if PER_DIR:  # (a direction's port injects every link's rows: its lists merged, under link 0's key)
        for k in [k for k in port_core if k[2] != 0]:
            k0 = k[:2] + (0,)
            inj[k0] = inj[k0] + inj.pop(k)
            n_ret[k0] += n_ret.pop(k)
            se_n[k0] += se_n.pop(k)
            se_ret[k0] += se_ret.pop(k)
            se_own[k0] = se_own[k0] + [o_ for o_ in se_own.pop(k) if o_ not in se_own[k0]]
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    MAXT = (S + A - 1) // A
    det = bool(deterministic)
    assert not det or (
        l1 is not None and devplan is not None and "dord" in devplan
    ), "deterministic: the L1 hand-off and a device plan with the planner's order table (combine_plan det=)"
    det_k = int(devplan.get("top_k", 0)) if det else 0
    assert not det or det_k, "deterministic: devplan top_k (the routing's top-k: the order table's row stride)"
    det_pg = int(devplan["dord"].shape[-1]) * 4 if det else 0
    assert not det or MAXT * det_k <= det_pg, (MAXT, det_k, det_pg)
    tab_cb = max(256 * 8 + 128, 256 + 64 + (MAXT * det_k + 63) // 64 * 64) if det else 256 * 8 + 128
    acc_cbs = sum(
        al(v) for v in [tab_cb, 2 * NT * TILE, 2 * 32, MAXT * NT * 4096, 2 * NT * TILE] + ([2 * TILE] if l1 else [])
    )
    if arena is not None:  # (the rings' depth from what the arena leaves: the accumulators' CBs and 2 local slots)
        sh_ = arena.memory_config().shard_spec
        arena_bytes = int(sh_.shape[0]) * int(sh_.shape[1]) * (4 if arena.dtype == ttnn.uint32 else 2)
        QP0 = inj_slots
        # (fused: room kept for an 8-slot local ring and a 32-slot injection ring; what is left after the rings
        # grows them, below)
        keep = (local_keep + min(32, QP0)) * ISLOT if FUSED else (local_keep * ISLOT if l1 else 0)
        fit = (arena_bytes - RINGS - acc_cbs - keep) // SLOT
        PSL = min(depth * (N - 1), fit)
    else:
        PSL = depth * (N - 1)
    # the landing pool: PSL slots per accumulator, carved into one ring per source per call by that call's demand
    # (every source with rows gets at least one slot - the per-ring deadlock argument holds - the rest in proportion
    # to its rows); depth = the pool's average per source
    PSL = PSL if ring_depth is None else int(ring_depth) * (N - 1)
    PSL = PSL if pool_slots is None else int(pool_slots)
    assert not SHARED or PSL < 255, f"landing shared: slot owners are bytes ({PSL} slots)"
    assert PSL >= N - 1, f"combine: no room for one landing slot per source chip ({N - 1} x {SLOT} B)"
    assert PSL < 4096, "combine: a ring's base / depth are 12-bit fields"
    D = PSL // (N - 1)
    # credit policy (SPC_CREDAUTO): when every accumulator's pool holds all the remote rows it takes this call, no source
    # waits for a credit mid-call - credits matter only at the end (the drain): no piggybacking (a piggybacked credit
    # rides a row that may land late), mid-call snapshots rare (64 rows, or idle 8 rows / 3 us), a source's final
    # credits at once. Otherwise sources need credits mid-call: the default policy
    # (decided from a bound every build of the op shares - they share the arena's cumulative credit state, so a fused
    # build with the device plan and its alone build must agree on piggybacking: an accumulator takes at most
    # ceil(S / A) tokens of at most K rows each; the final-credit shortcut needs the host's per-source counts)
    cred_thr = pr.credthr
    cred_policy = "default"
    K_ = max(collections.Counter((row[1], row[2]) for rows_ in rows_by_chip.values() for row in rows_).values(), default=0)
    if pr.credauto and not SHARED and shared_rows is None and not PER_DIR and not det and MAXT * K_ <= PSL:
        PIGGY, cred_thr, (cred_idle, cred_age) = False, "64", (8, 4050)
        CREDFIN = devplan is None
        cred_policy = "end"
    QP = inj_slots  # a port's L1 hand-off injection ring (its own region, no rings there)
    NSE = 8 if SE_ON else 0  # (the shared-expert rows' staging slots at a port, after its injection ring)
    acc_area = RINGS + PSL * SLOT
    if FUSED and arena is not None:  # (local ring up to 16 slots, then the injection ring up to QP, from what is left)
        a_end = al(acc_area) + acc_cbs
        QL_F = min(16, (arena_bytes - a_end - min(32, QP) * ISLOT) // ISLOT)
        QP = min(QP, (arena_bytes - a_end - QL_F * ISLOT) // ISLOT - NSE)
        assert QL_F >= 2 and QP >= 8, f"acc_on_ports: local ring {QL_F} / injection ring {QP} slots"
    port_area = RINGS + ((QP + NSE) * ISLOT if l1 is not None else 0)  # (fused: after the accumulator's, see build)
    # (per-chip roles: every home chip's accumulator cells, x | y << 8 a 16-bit entry, at the port's area end - the
    # sender copies it there from DRAM at its start)
    LUT_BYTES = al(2 * N * A) if PERCHIP else 0
    lut_off = al(port_area)
    port_area = lut_off + LUT_BYTES if PERCHIP else port_area
    area_bytes = max(acc_area, port_area)
    all_cores = sorted(used_cores | all_acc_xy)
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
    area = int(area_t.buffer_address())
    if pr.hw:  # (hang words: where a triage add-on finds them - every combine core's MISC 480..511)
        import json

        os.makedirs("generated/moe_block", exist_ok=True)
        cores = sorted({(v.x, v.y) for v in (virt(ttnn.CoreCoord(*c)) for c in all_cores)})
        json.dump(
            dict(misc=area + MISC, layr=area + LAYR, layo=area + LAYO, nap=N * AP, cores=cores),
            open("generated/moe_block/combine_hw.json", "w"),
        )
    cnt_off = CNT if counts_off is None else counts_off
    assert cnt_off % 16 == 0 and RET + 4 * N * AP <= cnt_off and cnt_off + 512 <= MISC
    coll = acc_cores[0]
    coll_v = virt(coll)

    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    tabs, offs = {}, {}
    for ch in chips:
        lst = []
        for k in sorted(k for k in port_core if k[0] == ch and k in inj):
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
    pkey = (lambda k: k[:2]) if PER_DIR else (lambda k: k)  # noqa: E731  (a physical port's key)
    if l1 is not None:  # destination ids per chip: its ports (sorted; per direction with PER_DIR), then accumulators
        port_ids = {}
        for ch in chips:
            uk = sorted({pkey(k) for k in port_core if k[0] == ch})
            port_ids[ch] = {k: uk.index(pkey(k)) for k in port_core if k[0] == ch}
        n_y = max((max((r[0] for r in rows), default=-1) for rows in rows_by_chip.values()), default=-1) + 1
        n_tiles = max((n_y + 31) // 32, 1)
        lt = torch.full((R, C, n_tiles, 64), 0, dtype=torch.int64)
        for ch, rows in rows_by_chip.items():
            for row in rows:
                yrow, home, t = row[:3]
                l = link_of(t % A, idx(ch), idx(home), L)
                hdr = idx(home) | (t << 8) | ((row[4] if len(row) > 4 else 0) << 24)
                n_pid = len(set(port_ids[ch].values()))
                dd = n_pid + t % A if home == ch else port_ids[ch][(ch, fh(ch, home), l)]
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
    l1_args = []
    prof = pr.prof
    pace_us = pr.pace_us  # pace every port's injections to finish at this time
    inj_b = pr.inj_b
    subs = {
        "__CRED__": str(CRED),
        "__USED__": str(USED),
        "__RET__": str(RET),
        "__MISC__": str(MISC),
        "__HDRS__": str(HDRS),
        "__RINGS__": str(RINGS),
        "__LAYR__": str(LAYR),
        "__LAYO__": str(LAYO),
        "__U0__": str(U0),
        "__PROF__": str(int(prof)),
        "__ABL__": pr.abl,
        "__SNAP__": str(_snap_bits(snap_q)),
        "__INJB__": str(inj_b),
        "__CREDTHR__": cred_thr,
        "__CREDIDLE__": str(cred_idle),
        "__FILLY__": str(int(round(pr.fill_y * 256))),
        "__FILLSINK__": str(int(fill_sink or 0)),
        "__FILLC__": str(C),
        "__CREDAGE__": str(cred_age),
        "__CREDBUSY__": str(int(pr.credbusy)),
        "__CREDFIN__": str(int(CREDFIN)),
        "__HCACHE__": str(int(pr.hcache)),
        "__GBGATE__": str(int(pr.gbgate)),
        "__FAR__": str(int(pr.far) if not det and int(pr.la) > 1 else 0),
        "__PACE__": str(int(pace_us > 0)),
        "__P__": str(NP),
        "__TPROF__": pr.tprof,
        "__L1__": str(int(l1 is not None)),
        "__CBTAB__": str(CB_TAB),
        "__CBW__": str(CB_W),
        "__DEVPLAN__": str(int(devplan is not None)),
        "__DET__": str(int(det)),
        "__DK__": str(max(det_k, 1)),
        "__DNTOK__": str(MAXT if det else 1),
        "__DPG__": str(max(det_pg, 64)),
        "__NOYREAD__": pr.noyread,  # (probe: rows not read from y: garbage)
        "__PD__": str(PD),
        "__PS__": str(PS),
        "__DRAIN__": "false" if pr.nodrain else "true",  # (probe: no end-of-call credit drain)
        "__HW__": str(int(pr.hw)),
        "__BK__": str(int(pr.bk)),
        "__LA__": str(0 if det else min(int(pr.la), 64)),  # (strict order with the deterministic combine)
        "__SHARED__": str(int(SHARED)),
        "__G__": str(max(G_, 1)),
        "__GIDH__": str(GIDH),
        "__GREC__": str(GREC),
        "__MRTOT__": str(M_RTOT),
        "__MRDONE__": str(M_RDONE),
        "__MTAKEN__": str(M_TAKEN),
        "__NH__": str(NH_),
        "__NL__": str(L if PER_DIR else 1),
        "__FUSED__": str(int(FUSED)),
        "__CBOUT__": str(CB_OUT),
        "__NT__": str(NT),
        "__ALAND__": "160" if FUSED else "128",
        "__ALBASE__": "103" if FUSED else "101",
        "__HSZ__": str(HSZ_),
        "__SE__": str(int(SE_ON)),
        "__NSE__": str(max(NSE, 1)),
        "__BAR__": str(int(barrier is not None)),
        "__PIGGY__": str(int(PIGGY)),
        "__PIG__": str(PIG),
        "__SHW__": str(SHW),
        "__SOWN__": str(S if SE_ON and pr.se not in ("off", "noown") else 0),
        "__SENOREAD__": str(int(pr.se == "noread")),
        "__OWNADDR__": str(int(shared_rows["tensor"].buffer_address()) if SE_ON else 0),
    }
    dbg_t = ttnn.from_torch(
        torch.zeros(R, C, 128, 16, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1)),
    )
    hcache_t = ttnn.from_torch(  # (the senders' routed headers: a page per port, N x NH headers + a 64 B key; zero: unset)
        torch.zeros(R, C, 8, (N * NH_ * HSZ_ + 64) // 4, dtype=torch.int32),
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
    # dedicated NoCs, fixed per RISC: RISCV_0 on NoC1, RISCV_1 on NoC0. A kernel changes NoC by changing RISC: SPC_SNOC=0
    # runs the sender (and the accumulator writer) on RISCV_1 / NoC0, the port and accumulator readers on RISCV_0 / NoC1
    R_S, R_R = ("RISCV_1", "RISCV_0") if pr.snoc == "0" else ("RISCV_0", "RISCV_1")
    noc_of = {"RISCV_0": NOC1, "RISCV_1": NOC0}
    chip_ids = [int(node((i // C, i % C)).chip_id) for i in range(N)]
    mesh_id = int(node((0, 0)).mesh_id)
    acc_xy = []
    for ac in acc_cores:
        v = virt(ac)
        acc_xy += [v.x, v.y]
    lut_t = None
    if PERCHIP:  # (a home chip's accumulator o: entry idx(home) * A + o; replicated, one page)
        lut = torch.zeros(LUT_BYTES // 2, dtype=torch.int32)
        for ch in chips:
            for o, ac in enumerate(accs_of(ch)):
                v = virt(ac)
                assert v.x < 256 and v.y < 256
                lut[idx(ch) * A + o] = v.x | v.y << 8
        lut_t = ttnn.from_torch(
            (lut[0::2] | lut[1::2] << 16).reshape(1, -1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    state = {}

    def arena_cbs(placed, arena_t):
        cbs = []
        for cb, crs, off, total, fmt, page in placed:
            d_ = ttnn.cb_descriptor_from_sharded_tensor(
                cb, arena_t, address_offset=off, total_size=total, core_ranges=crs
            )
            d_.format_descriptors = [ttnn.CBFormatDescriptor(cb, fmt, page)]
            cbs.append(d_)
        return cbs

    def rebind(arena_t):
        """The last build's descriptor with its CBs on arena_t (a per-call arena at the same address): what a
        program-cache miss needs (the CBs bind the buffer, a freed one is no use)."""
        assert int(arena_t.buffer_address()) == area, "the arena moved: the kernels bake its address"
        mesh_desc = ttnn.MeshProgramDescriptor()
        for ch, program, placed in state["placed"]:
            program.cbs = arena_cbs(placed, arena_t) + state.get("static", {}).get(ch, [])
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*ch), ttnn.MeshCoordinate(*ch))] = program
        return mesh_desc

    def build(y, out, arena_t=None, disp=None):
        nonlocal area_t
        if arena_t is not None:  # (a per-call arena: the same address, its buffer bound by the CBs)
            assert int(arena_t.buffer_address()) == area, "the arena moved: the kernels bake its address"
            area_t = arena_t
        dpieces, ksrc = None, dict(srcs)
        state["static"] = {}
        if dispatch is not None:  # (the dispatch as the port kernels' first phase)
            from .sparse_all_gather_2d import dispatch_phase

            assert disp is not None and l1 is not None and not PER_DIR and not FUSED, "dispatch: L1 hand-off, per link"
            ps_ct0 = 8
            dpieces, (rinc, rph, sinc, sph, rd_ct, sd_ct) = dispatch_phase(
                mesh_device,
                disp["output"],
                disp["plan"],
                disp["manifest"],
                port_core,
                dispatch["sems"],
                rd_ct0=1,
                sd_ct0=ps_ct0,
                cbs=DISP_CBS,
                relays=dispatch["relays"],
                words_addr=dispatch["words"],
                ready_off=dispatch["ready_off"],
                num_links=L,
                torus=torus,
                group=dispatch.get("group", 8),
            )
            rd_n = len(next(iter(next(iter(dpieces.values()))["rd"].values())))
            sd_n = len(next(iter(next(iter(dpieces.values()))["sd"].values())))
            pre_r, pre_s = "void kernel_main() {\n    size_t a = 0;\n", "void kernel_main() {\n    size_t a = 0;\n"
            assert (
                ksrc["pr1"].count(pre_r) == 1
                and ksrc["ps"].count(pre_s) == 1
                and ksrc["ps"].count("    conn.open();\n") == 1
            )
            ksrc["pr1"] = (
                rinc
                + rph
                + ksrc["pr1"].replace(pre_r, f"void kernel_main() {{\n    sa::read_phase(0);\n    size_t a = {rd_n};\n")
            )
            ksrc["ps"] = (
                sinc
                + sph
                + ksrc["ps"]
                .replace(pre_s, f"void kernel_main() {{\n    size_t a = {sd_n};\n")
                .replace(
                    "    conn.open();\n", "    conn.open();\n    sa::send_phase(0, conn);  // (the dispatch first)\n"
                )
            )
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        t_ct = list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
        o_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        mesh_desc = ttnn.MeshProgramDescriptor()
        state["placed"] = []
        for ch, ports in chips.items():
            program = ttnn.ProgramDescriptor()
            placed = []  # (the arena-backed CBs: rebind(arena) rebuilds them on another arena tensor)
            state["placed"].append((ch, program, placed))
            my = [
                (d, li, port_core[(ch, d, li)])
                for d, p in ports.items()
                for li in range(len(p["links"]))
                if not PER_DIR or li == 0
            ]
            p_xy = {(c.x, c.y) for _, _, c in my}  # (this chip's port cores; fused: accumulator cells too)
            pset = ttnn.CoreRangeSet([ttnn.CoreRange(k, k) for _, _, k in my])
            accs = accs_of(ch)  # (this chip's accumulator cells)
            aset = ttnn.CoreRangeSet([ttnn.CoreRange(k, k) for k in accs])
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
                ends = {"p": al(port_area), "a": al(acc_area)}
                for cb, crs, total, fmt, page in specs:
                    if l1 is not None and cb == CB_INJ:  # (the hand-off ring: the writers address it directly)
                        if not FUSED:
                            placed.append((cb, crs, RINGS, QP * ISLOT, fmt, page))
                        continue  # (fused: after the accumulator's local ring, below)
                    if cb == CB_TAB:  # (both kinds, each in its own region; fused: the accumulators' only)
                        for k, crs_k in (("p", pset), ("a", aset)) if not FUSED else (("a", aset),):
                            placed.append((cb, crs_k, ends[k], total, fmt, page))
                            ends[k] = al(ends[k] + total)
                        continue
                    k = "p" if cb == CB_INJ else "a"
                    placed.append((cb, crs, ends[k], total, fmt, page))
                    ends[k] = al(ends[k] + total)
                cbs = arena_cbs(placed, area_t)
                assert (
                    max(ends.values()) <= arena_bytes
                ), f"combine needs {max(ends.values()) >> 10} KB per core, the arena has {arena_bytes >> 10} KB"
                if (
                    l1 is not None
                ):  # the accumulators' local ring: what is left of the arena (fused: and the port's ring)
                    loc_off = ends["a"]
                    QL = QL_F if FUSED else min(16, (arena_bytes - loc_off) // ISLOT)
                    assert QL >= 2, f"no room for the accumulators' local ring ({(arena_bytes - loc_off) >> 10} KB)"
                    inj_off = loc_off + QL * ISLOT if FUSED else RINGS
                    if FUSED:
                        assert (
                            inj_off + (QP + NSE) * ISLOT <= arena_bytes
                        ), "acc_on_ports: no room for the injection ring"
                        cbs.append(arena_cbs([(CB_INJ, pset, inj_off, QP * ISLOT, ttnn.uint32, ISLOT)], area_t)[0])
                        placed.append((CB_INJ, pset, inj_off, QP * ISLOT, ttnn.uint32, ISLOT))
            if dpieces is not None:  # (the dispatch's CBs: static, on this chip's port cores)
                state["static"][ch] = list(dpieces[ch]["cbs"])
                cbs = cbs + state["static"][ch]
            program.cbs = cbs
            if l1 is not None:
                pids = port_ids[ch]
                tab_dev = devplan["table"] if devplan is not None else l1_tab
                n_pid = len(set(pids.values()))
                la = [n_pid + A, ISLOT, int(tab_dev.buffer_address()), int(l1["words"]), 0, 0, 0, 0]
                for j in range(n_pid):
                    v = virt(port_core[min(k for k in pids if pids[k] == j)])
                    la += [v.x, v.y, area + inj_off, QP, area + MISC + 512]
                for i_, ac in enumerate(accs):
                    v = virt(ac)
                    la += [v.x, v.y, area + loc_off, QL, area + MISC + (640 if FUSED else 512)]
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
                acc_args = [int(lut_t.buffer_address()), area + lut_off] if PERCHIP else acc_xy
                args = [area, mesh_id, idx(ch), li] + cnt_args + chip_ids + acc_args
                n_conn = len(ports[d]["links"]) if PER_DIR else 1
                for i in range(N):
                    s = (i // C, i % C)
                    if s != ch and fh(ch, s) == d:
                        v = virt(port_core[(s, fh(s, ch), li)])
                        args += [v.x, v.y]
                    else:
                        args += [0, 0]
                if CREDFIN:
                    args += ret_src[k]
                if int(pr.far) and not det and int(pr.la) > 1:  # (hops to each home chip: route()'s length)
                    args += [len(route(ch, (i // C, i % C), R, C, wrap_x, wrap_y, 0, pr.ytie)) for i in range(N)]
                args += [int(pace_us * 1350 / len(inj[k])) if pace_us > 0 and inj[k] else 0]
                args += [int(dbg_t.buffer_address()), DIRS.index(d) * 2 + li]
                args += [int(tdbg_t.buffer_address()), 8 + DIRS.index(d) * 2 + li]
                args += [int(hcache_t.buffer_address()), DIRS.index(d) * 2 + li]
                for lj in range(n_conn) if PER_DIR else [li]:
                    args += list(
                        ttnn.setup_fabric_connection(node(ch), node(peer), ports[d]["links"][lj], program, core)
                    )
                if FUSED:  # (the hand-off receiver's args, then the accumulator writer's)
                    ai = [(c.x, c.y) for c in accs].index((core.x, core.y))
                    args += [len(wv), port_ids[ch][k], QP, int(l1["words"]), int(out.buffer_address())]
                    args += [len(range(ai, S, A)), ai] + wxy
                if SE_ON:  # (the shared-expert rows: partials, count, staging, block rows, credits, owners)
                    stage = (inj_off if FUSED else RINGS) + QP * ISLOT
                    args += [int(shared_rows["tensor"].buffer_address()), se_n[k], area + stage, S, se_ret[k]]
                    args += [len(se_own[k])] + [v_ for o_ in se_own[k] for v_ in o_]
                if (
                    barrier is not None
                ):  # (after the SE args, as the kernel reads them; the port facing each other chip increments every port core there)
                    bt = []
                    if li == 0:
                        for s_ in chips:
                            if s_ != ch and fh(ch, s_) == d:
                                for xy in sorted({(c_.x, c_.y) for kk_, c_ in port_core.items() if kk_[0] == s_}):
                                    v_ = virt(ttnn.CoreCoord(*xy))
                                    bt += [idx(s_), v_.x, v_.y]
                    args += [int(barrier["tensor"].buffer_address()), len(bt) // 3] + bt
                if dpieces is not None:  # (the dispatch phase's args first)
                    args = list(dpieces[ch]["sd"][(core.x, core.y)]) + args
                    pr_rt[core.x][core.y] = list(dpieces[ch]["rd"][(core.x, core.y)]) + list(pr_rt[core.x][core.y])
                ps_rt[core.x][core.y] = args
            ar_rt, ac_rt, aw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, core in enumerate(accs):
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
                    idx(ch),
                ]
                for kk in range(N):  # per source chip: its ring's count and the port returning its credits
                    s = (kk // C, kk % C)
                    w = want[(ch, i)][kk]
                    if s != ch:  # (every source: its ring's layout goes out through that port even with no rows)
                        v = virt(port_core[(ch, fh(ch, s), link_of(i, kk, idx(ch), L))])
                        args += [(0 if devplan is not None else w) + se_want[(ch, i)][kk], v.x, v.y]
                    else:
                        args += [0, 0, 0]
                args += [int(dbg_t.buffer_address()), 8 + i, int(tdbg_t.buffer_address()), 16 + i]
                if l1 is not None:
                    args += [QL, area + loc_off, len(wv), len(set(port_ids[ch].values())) + i, int(l1["words"])] + wxy
                    if det:  # (the planner's order table: page i)
                        args += [int(devplan["dord"].buffer_address())]
                else:
                    args += [0, 0, 0, 0, 0]
                ar_rt[core.x][core.y] = args
                n_rows = len(loc[(ch, i)]) + sum(want[(ch, i)]) + sum(se_want[(ch, i)])
                ac_rt[core.x][core.y] = [0 if devplan is not None else n_rows, n_tok]
                if not FUSED or (core.x, core.y) not in p_xy:  # (fused port cores: the sender writes)
                    aw_rt[core.x][core.y] = [int(out.buffer_address()), n_tok, i]
            aw_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in accs if not FUSED or (c.x, c.y) not in p_xy])
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=ksrc["pr1"] if l1 is not None else srcs["pr"],
                    source_type=st,
                    core_ranges=pset,
                    compile_time_args=(
                        [CB_INJ] + (rd_ct if dpieces is not None else [])
                        if l1 is not None
                        else [CB_INJ, CB_TAB, row_bytes, ISLOT] + y_ct + t_ct
                    ),
                    runtime_args=pr_rt,
                    config=dm(R_R, noc_of[R_R]),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=(f"#define ACC_LUT\n#define LUT_BYTES {LUT_BYTES}u\n" if PERCHIP else "") + ksrc["ps"],
                    source_type=st,
                    core_ranges=pset,
                    compile_time_args=[CB_INJ, row_bytes, SLOT, PSL, A, N, L, ISLOT]
                    + (o_ct if FUSED else [])
                    + (sd_ct if dpieces is not None else []),
                    runtime_args=ps_rt,
                    config=dm(R_S, noc_of[R_S]),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["ar"],
                    source_type=st,
                    core_ranges=aset,
                    compile_time_args=[CB_ROW, CB_META, CB_TAB, row_bytes, SLOT, PSL, A, NT, N, L] + y_ct + t_ct,
                    runtime_args=ar_rt,
                    config=dm(R_R, noc_of[R_R]),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["ac"],
                    source_type=st,
                    core_ranges=aset,
                    compile_time_args=[CB_ROW, CB_META, CB_ACC, CB_OUT, NT, MAXT] + ([CB_W] if l1 is not None else []),
                    runtime_args=ac_rt,
                    config=acc_compute_config(ttnn.MathFidelity.HiFi4),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=srcs["aw"],
                    source_type=st,
                    core_ranges=aw_set,
                    compile_time_args=[CB_OUT, row_bytes, NT, A] + o_ct,
                    runtime_args=aw_rt,
                    config=dm(R_S, noc_of[R_S]),
                ),
            ]
            if FUSED:  # (no separate hand-off receiver: the sender runs it; no writer kernel on the port cores)
                program.kernels = [k_ for k_ in program.kernels if k_.kernel_source != srcs["pr1"]]
                if not p_xy.isdisjoint({(c.x, c.y) for c in accs}) and len(p_xy) == len(accs):
                    program.kernels = [k_ for k_ in program.kernels if k_.kernel_source != srcs["aw"]]
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*ch), ttnn.MeshCoordinate(*ch))] = program
        return mesh_desc

    def run(y, out):
        k = (int(y.buffer_address()), int(out.buffer_address()))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(y, out)
        return ttnn.generic_op([y, tab_t, out], state["desc"])

    def build_l1(
        y, out, arena_t=None, disp=None
    ):  # (the descriptor and, per chip in its order, the y writers' hand-off args)
        l1_args.clear()
        desc = build(y, out, arena_t, disp)
        return desc, [list(v) for v in l1_args]

    # what a fresh (uninitialized) arena needs zeroed for a call to start from scratch: per core (logical x, y), runs of
    # (offset in the arena, bytes, stride, count): the state block, and at the accumulators every ring slot's piece
    # trailers (a slot is taken when its trailer flags are set)
    trl = [pc * PS + (row_bytes - pc * PD if pc + 1 == NP else PD) for pc in range(NP)]
    zero_runs = {c: [(0, RINGS, 0, 1)] for c in used_cores}
    for c in sorted(all_acc_xy):  # (per-chip roles: a cell an accumulator on any chip - zeroed the same on every chip)
        zero_runs[c] = [(0, RINGS, 0, 1)] + [(RINGS + t, 32, SLOT, PSL) for t in trl]

    run.build = build
    run.build_l1 = build_l1
    run.rebind = rebind
    run.tdbg = tdbg_t
    run.hcache = hcache_t
    run.cred_policy = cred_policy
    run.tab = tab_t
    run.collector = dict(core=coll, counts_addr=area + cnt_off, writers=writers)
    run.counts = dict(n_inj={k: len(v) for k, v in inj.items()}, n_ret=n_ret)
    run.cells = run_cells
    run.depth = D
    run.pool = PSL
    run.l1 = dict(acc_area=acc_area, port_area=port_area, rings=PSL * SLOT, state=RINGS, slot=SLOT, inj=QP)
    run.zero_runs = zero_runs
    run.dbg = dbg_t
    return run
