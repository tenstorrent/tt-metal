# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""routing_a2a — the head's routing exchange as a direct all-to-all of routing entries, and the combine planner from
local data only (no chip ever holds the whole mesh's routing).

What crosses chips: only "every routing entry that points at my experts". Everything else is local: a home chip owns
its tokens, so the combine's n_local / want / n_ret come from its own top-k (and so do the dispatch's route words).

Input per chip: its own tokens' joint routing rows (router_topk's [S / TPP, TPP x 2 K] uint16: per token the K global
expert ids, then the K gate weights' bf16 bits). Output: combine_plan.py's per-chip tables - the same tensors and the
same words (counts, regions, token_index, table, plan), so the flat expert and the combine read them unchanged.

One program per chip, M planner cores (BRISC) and the chip's fabric port cores (BRISC; one per direction and link):

  planner core q owns accumulators a = q, q + M, ... (their home tokens t, t % A = a) and local experts [e0, e1).
  (1) Over its home tokens: their rows' n_local / want / n_ret (local), and a count per global expert g.
  (2) The per-destination totals go to the other planner cores (one exchange), giving each core its offset in every
      destination's packet.
  (3) Its entries, sorted by global expert (= destination chip, then its expert), written straight into the packet
      buffer of the port core that serves each destination, at its offset; with its slice's per-expert prefix.
  (4) Core 0 waits for every packet of the call (N, its own chip's included), then wakes the others.
  (5) Every packet header (global counts -> regions; each core's experts' row range is contiguous), every packet's
      entries (one read each); a stable counting sort by token of its experts' entries (sources in order) gives each
      expert's rows in (token, home, k) order - combine_plan's token_major order; the rows placed (token index, table,
      n_inj); its row range written. Core 0 adds the n_inj / n_ret partials and writes counts / regions / plan.

  port core: destination d goes out on the port of its first hop (Y first, then X: the fabric's route tables), link
  d % L; one port core also takes my own chip's packet. Once the M planner cores' slices are in, one packet per
  destination - EXACTLY one per (source, destination) pair, empty ones included - as a direct 2D unicast the fabric
  routers forward (multi-hop), in max-payload chunks (plain writes, the last a fused write + increment of the
  destination's arrival counter); my own: a local write and increment.

A packet: the header (HBP bytes: words 0..M-1 each slice's entry count; at 64 the per-expert totals, uint16; at
64 + BB slice q's per-expert prefix, uint16, EPC + 1 of them), then the slices' entries in order, each sorted by
expert, 4 bytes an entry: token (11 bits) | local expert (6 bits) << 11 | the gate weight's bf16 bits but the sign
<< 17 (gate weights are >= 0; the home chip is implied by the slot the packet lands in).

Receivers: a slot tensor per chip in DRAM, [2 N pages] (page = parity N + source chip: double-buffered by the call's
parity, so a fast chip's next call never lands in a slot a slow chip still reads), sized for the worst case (every
entry of a source to one chip: S x min(K, EPC)). Two arrival counters (one per parity, global semaphores on planner
core 0): wait for N, then subtract N (counts stay right with the next call's early arrivals).
"""

import os

import ttnn

from .fabric_all_gather_py import allowed_cores
from .sparse_all_gather_2d import DIRS, mesh_plan, place_ports

PLAN_WORDS = 1024


def _al16(v):
    return (v + 15) // 16 * 16


def _al64(v):
    return (v + 63) // 64 * 64


def header_bytes(EPC, M):
    """A packet's header: M slice counts (64 B), the per-expert totals, then M per-expert prefixes (uint16)."""
    return _al64(64 + _al16(2 * EPC) + M * _al16(2 * (EPC + 1)))


def slot_bytes(S, K, EPC, M):
    """A source's packet (and its slot on a destination chip): the header and every entry it could send there."""
    return _al64(header_bytes(EPC, M) + 4 * S * min(K, EPC))


_FIRST_HOP = r"""
constexpr uint32_t WX = __WX__, WY = __WY__;  // torus axes (rings: the shorter way, ties Y- / X+ as the fabric)
constexpr uint32_t YT = __YT__;  // (used only for the combine plan's ports: this op's own packets route as the fabric)
FORCE_INLINE uint32_t first_hop(uint32_t src, uint32_t dst) {  // DIRS index: 0 X+, 1 X-, 2 Y+, 3 Y-; Y first, then X
    const uint32_t sr = src / C, sc = src % C, dr = dst / C, dc = dst % C;
    if (sr != dr) {
        if constexpr (WY) {  // (YT, the combine's SPC_YTIE: its plan's ports - the opposite row splits by column parity)
            const uint32_t f2 = 2 * ((dr + R - sr) % R);
            return f2 < R || (YT && f2 == R && ((sc + dc) & 1)) ? 2 : 3;
        }
        return dr > sr ? 2 : 3;
    }
    if constexpr (WX) return 2 * ((dc + C - sc) % C) <= C ? 0 : 1;
    return dc > sc ? 0 : 1;
}
"""

_SEND = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#if __NR__ > 0
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#endif
constexpr uint32_t N = __N__, ME = __ME__, M = __M__, EPC = __EPC__;
constexpr uint32_t ND = __ND__;      // packets I send: the remote destinations first, then my own chip's (SELF)
constexpr uint32_t NR = __NR__;      // remote ones (over the fabric)
constexpr uint32_t SELF = __SELF__;  // the last one is my own chip's (a local write)
constexpr uint32_t HBP = __HBP__, BB = __BB__, BBP = __BBP__, PCAP = __PCAP__, MAXP = __MAXP__, NH = 4;
constexpr auto sl_args = TensorAccessorArgs<0>();
constexpr uint32_t O_DBG = ND * PCAP;

void kernel_main() {
    size_t a = 0;
    const uint32_t sl_addr = get_arg_val<uint32_t>(a++);
    const uint32_t c0 = get_arg_val<uint32_t>(a++), c1 = get_arg_val<uint32_t>(a++);
    const uint32_t pcx = get_arg_val<uint32_t>(a++), pcy = get_arg_val<uint32_t>(a++);
    const uint32_t calls_addr = get_arg_val<uint32_t>(a++), ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t mesh_arg = get_arg_val<uint32_t>(a++);
    const uint32_t dbg_addr = get_arg_val<uint32_t>(a++), dbg_page = get_arg_val<uint32_t>(a++);
    uint32_t chip[ND];
    for (uint32_t j = 0; j < ND; ++j) chip[j] = get_arg_val<uint32_t>(a++);
    uint32_t ts[4];
    ts[0] = get_timestamp_32b();
    set_l1_data_cache<true>();  // (off again before returning; NoC arrivals invalidated)
    const uint32_t B = get_write_ptr(0);
#if __NR__ > 0
    const uint16_t mesh = static_cast<uint16_t>(mesh_arg);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[NH];
    for (uint32_t h = 0; h < NH; ++h) hdrs[h] = PacketHeaderPool::allocate_header();
    conn.open();
#else
    (void)mesh_arg;
#endif
    volatile tt_l1_ptr uint32_t* calls = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(calls_addr);
    const uint32_t par = calls[0] & 1;  // (every port core and planner core counts its calls: they stay in step)
    calls[0] = calls[0] + 1;
    const auto sl = TensorAccessor(sl_args, sl_addr, PCAP);
    const uint64_t cnt_noc = get_noc_addr(pcx, pcy, par ? c1 : c0);
    const uint64_t my_slot = sl.get_noc_addr(par * N + ME);  // (the same page on every chip: my slot there)
    // the M planner cores' slices (a persistent counter: wait for M, take them)
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr), M);
    noc_semaphore_inc(get_noc_addr(ready_addr), 0u - M);
    noc_async_atomic_barrier();
    invalidate_l1_cache();
    ts[1] = get_timestamp_32b();
    uint32_t bytes[ND];
    for (uint32_t j = 0; j < ND; ++j) {  // (the packet's bytes: its slices' entries; the per-expert totals are the
                                         // planners' - the owner instance of the destination wrote them)
        const uint32_t pb = B + j * PCAP;
        const uint32_t* nq = reinterpret_cast<const uint32_t*>(pb);
        uint32_t n = 0;
        for (uint32_t q = 0; q < M; ++q) n += nq[q];
        bytes[j] = (HBP + 4 * n + 63) / 64 * 64;
    }
    ts[2] = get_timestamp_32b();
#if __NR__ > 0
    uint32_t sent = 0;
    for (uint32_t j = 0; j < NR; ++j) {
        for (uint32_t off = 0; off < bytes[j]; off += MAXP) {
            const uint32_t n = bytes[j] - off < MAXP ? bytes[j] - off : MAXP;
            const bool last = off + n >= bytes[j];
            if (sent >= NH && sent % NH == 0) noc_async_writes_flushed();  // (a header is reused: its last send out)
            volatile tt_l1_ptr PACKET_HEADER_TYPE* h = hdrs[sent % NH];
            (void)fabric_set_unicast_route(h, static_cast<uint16_t>(chip[j]), mesh);
            if (last) {
                h->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{my_slot + off, cnt_noc, 1, true}, n);
            } else {
                h->to_noc_unicast_write(NocUnicastCommandHeader{my_slot + off}, n);
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(B + j * PCAP + off, n, reinterpret_cast<uint32_t>(h));
            ++sent;
        }
    }
#endif
    if constexpr (SELF) {  // my own chip's packet: a local write, then the increment
        noc_async_write(B + (ND - 1) * PCAP, my_slot, bytes[ND - 1]);
        noc_async_write_barrier();
        noc_semaphore_inc(cnt_noc, 1);
    }
#if __NR__ > 0
    noc_async_writes_flushed();
    conn.close();
#endif
    ts[3] = get_timestamp_32b();
    if (dbg_addr) {
        uint32_t* d = reinterpret_cast<uint32_t*>(B + O_DBG);
        for (uint32_t i = 0; i < 4; ++i) d[i] = ts[i];
        d[4] = ND;
        d[5] = B;
        const InterleavedAddrGen<true> dg = {.bank_base_address = dbg_addr, .page_size = 64};
        noc_async_write(B + O_DBG, get_noc_addr(dbg_page, dg), 64);
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
    set_l1_data_cache<false>();
}
"""

_PLAN = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t S = __S__, K = __K__, EPC = __EPC__, R = __R__, C = __C__, N = R * C, ME = __ME__;
constexpr uint32_t A = __A__, L = __L__, P = __P__, TPP = __TPP__, M = __M__, NG = N * EPC, E0 = ME * EPC;
constexpr uint32_t HBP = __HBP__, BB = __BB__, BBP = __BBP__, PCAP = __PCAP__, NPORT = __NPORT__;
constexpr uint32_t ROWS_CAP = __ROWS_CAP__, ENT_CAP = __ENT_CAP__;  // (ENT_CAP: an instance's entries, worst case)
constexpr uint32_t INST_ROWS = ENT_CAP + 32 * EPC < ROWS_CAP ? ENT_CAP + 32 * EPC : ROWS_CAP;  // (an instance's rows)
constexpr uint32_t ROWB = 4 * K, PB = TPP * ROWB, PSZ = (PB + 63) / 64 * 64, NPG = S / TPP;
constexpr uint32_t NOWN = (A + M - 1) / M;  // accumulators a core owns at most
constexpr uint32_t al16(uint32_t v) { return (v + 15) / 16 * 16; }
constexpr uint32_t al64(uint32_t v) { return (v + 63) / 64 * 64; }
constexpr uint32_t TSM = NOWN * ((S + A - 1) / A);  // my home tokens at most
constexpr uint32_t O_RT = 0, O_EX = al64(NPG * PSZ), O_NQ = O_EX + al64(4 * M * N), O_PRE = O_NQ + al64(4 * N);
constexpr uint32_t O_STG = O_PRE + N * BBP, O_HDR = O_STG + al64(4 * TSM * K + 16 * N);
constexpr uint32_t O_ENT = O_HDR + N * HBP, O_RUN = O_ENT + al64(4 * ENT_CAP + 128 * N * M);
constexpr uint32_t O_ET = O_RUN + al64(4 * N * M);
constexpr uint32_t O_TIDX = O_ET + al64(2 * EPC * S), O_TAB = O_TIDX + al64(4 * INST_ROWS);
constexpr uint32_t O_WANT = O_TAB + al64(8 * INST_ROWS), O_PIN = O_WANT + al64(128 * NOWN);
constexpr uint32_t O_PLAN = O_PIN + M * 64, O_CNT = O_PLAN + 128, O_RGN = O_CNT + al64(4 * (EPC + 32));
constexpr uint32_t O_ROWS = O_RGN + al64(4 * (EPC + 32));
constexpr uint32_t NOWND = (N + M - 1) / M;  // (destinations an instance owns the header totals of: d % M == q)
constexpr uint32_t O_CX = O_ROWS + 64, O_PSUM = O_CX + N * BB, O_TOT = O_PSUM + NOWND * M * BB;
constexpr uint32_t O_DBG = O_TOT + al64(NOWND * BB);
constexpr uint32_t LMW = 1024;  // local-memory counters: per global expert, later per (my expert, token) if they fit
static_assert(NG <= LMW, "local counters: NG up to 1024");
static_assert(S <= 2048 && EPC <= 64, "an entry: token (11 bits) | expert (6 bits) | weight (15 bits)");
constexpr auto rt_args = TensorAccessorArgs<0>();
constexpr auto sl_args = TensorAccessorArgs<rt_args.next_compile_time_args_offset()>();
"""
    + _FIRST_HOP
    + r"""
void kernel_main() {
    size_t a = 0;
    const uint32_t rt_addr = get_arg_val<uint32_t>(a++), sl_addr = get_arg_val<uint32_t>(a++);
    const uint32_t c0 = get_arg_val<uint32_t>(a++), c1 = get_arg_val<uint32_t>(a++);
    const uint32_t calls_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++), reg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tidx_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    uint32_t lut[8];
    for (uint32_t i = 0; i < 8; ++i) lut[i] = get_arg_val<uint32_t>(a++);
    const uint32_t q = get_arg_val<uint32_t>(a++);
    a += 2;  // (static expert ranges: unused, the ranges follow the call's rows)
    uint32_t px[M], py[M];
    for (uint32_t i = 0; i < M; ++i) {
        px[i] = get_arg_val<uint32_t>(a++);
        py[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t dst_arg = a;  // per destination chip: its port core (x, y) and its packet buffer's offset there
    a += 3 * N;
    const uint32_t port_arg = a;  // the port cores (x, y)
    a += 2 * NPORT;
    const uint32_t dbg_addr = get_arg_val<uint32_t>(a++), ready_sem = get_arg_val<uint32_t>(a++);  // (port cores')
    const uint32_t go_sem = get_semaphore(0), parts_sem = get_semaphore(1), ex_sem = get_semaphore(2);
    uint32_t ts[10];
    ts[0] = get_timestamp_32b();
#if __DCACHE__
    // (the RISC's L1 data cache - off by default: every L1 load is then ~10+ cycles; write-through, so only NoC
    // arrivals need the invalidations this kernel already does; off again before returning, for the firmware)
    set_l1_data_cache<true>();
#endif
    const uint32_t B = get_write_ptr(0);
    const auto rt = TensorAccessor(rt_args, rt_addr, PB);
    for (uint32_t pg = 0; pg < NPG; ++pg) noc_async_read(rt.get_noc_addr(pg), B + O_RT + pg * PSZ, PB);
    volatile tt_l1_ptr uint32_t* calls = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(calls_addr);
    const uint32_t par = calls[0] & 1;
    calls[0] = calls[0] + 1;
    alignas(4) uint16_t lm[LMW];  // (local memory: entries per global expert, then cursors; later per (my expert, token))
    for (uint32_t i = 0; i < NG; ++i) lm[i] = 0;
    noc_async_read_barrier();
    invalidate_l1_cache();
    // (1) my home tokens (t % A owned by me): my entries per global expert
    auto row = [&](uint32_t t) __attribute__((always_inline)) {
        return reinterpret_cast<const uint32_t*>(B + O_RT + (t / TPP) * PSZ + (t % TPP) * ROWB);
    };
    uint32_t n_my = 0;
    for (uint32_t aa = q; aa < A; aa += M) {
        for (uint32_t t = aa; t < S; t += A) {
            const uint32_t* r2 = row(t);
            for (uint32_t kk = 0; kk < K / 2; ++kk) {
                const uint32_t v = r2[kk];
                ++lm[v & 0xFFFF];
                ++lm[v >> 16];
            }
            n_my += K;
        }
    }
    ts[1] = get_timestamp_32b();
    // (2) my per-destination totals to every planner core
    uint32_t* nq = reinterpret_cast<uint32_t*>(B + O_NQ);
    for (uint32_t d = 0; d < N; ++d) {
        uint32_t s = 0;
        for (uint32_t e = 0; e < EPC; ++e) s += lm[d * EPC + e];
        nq[d] = s;
    }
    for (uint32_t r = 0; r < M; ++r) noc_async_write(B + O_NQ, get_noc_addr(px[r], py[r], B + O_EX + q * N * 4), N * 4);
    {  // my per-expert counts of destination d to its owner instance d % M (slot q): the owner sums the M slices'
       // into the packet header's totals - the sender then only sends
        uint16_t* cx = reinterpret_cast<uint16_t*>(B + O_CX);
        for (uint32_t d = 0; d < N; ++d)
            for (uint32_t e = 0; e < BB / 2; ++e) cx[d * (BB / 2) + e] = e < EPC ? lm[d * EPC + e] : 0;
        for (uint32_t d = 0, r = 0; d < N; ++d, r = r + 1 == M ? 0 : r + 1)
            noc_async_write(B + O_CX + d * BB, get_noc_addr(px[r], py[r], B + O_PSUM + ((d / M) * M + q) * BB), BB);
    }
    noc_async_write_barrier();
    for (uint32_t r = 0; r < M; ++r) {
        if (r != q) noc_semaphore_inc(get_noc_addr(px[r], py[r], ex_sem), 1);
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem), M - 1);
    invalidate_l1_cache();
    ts[2] = get_timestamp_32b();
    // (3) my entries sorted by global expert into staging (each destination's run aligned as its place in the packet,
    // mod 16), straight into the port cores' packet buffers
    const uint32_t* ex = reinterpret_cast<const uint32_t*>(B + O_EX);
    {
        uint32_t doff[N], stg[N];
        uint32_t ptr = B + O_STG;
        for (uint32_t d = 0; d < N; ++d) {
            uint32_t o = 0;
            for (uint32_t r = 0; r < q; ++r) o += ex[r * N + d];
            doff[d] = HBP + 4 * o;
            ptr += (doff[d] - ptr) & 15u;
            stg[d] = ptr;
            uint16_t* pre = reinterpret_cast<uint16_t*>(B + O_PRE + d * BBP);
            uint32_t c = 0, w = (ptr - (B + O_STG)) / 4;
            for (uint32_t e = 0; e < EPC; ++e) {
                pre[e] = static_cast<uint16_t>(c);
                const uint32_t n = lm[d * EPC + e];
                lm[d * EPC + e] = static_cast<uint16_t>(w);  // (cursor: word index in the staging)
                w += n;
                c += n;
            }
            pre[EPC] = static_cast<uint16_t>(c);
            ptr += 4 * c;
        }
        uint32_t* sw = reinterpret_cast<uint32_t*>(B + O_STG);
        for (uint32_t aa = q; aa < A; aa += M) {
            for (uint32_t t = aa; t < S; t += A) {
                const uint32_t* r2 = row(t);
                for (uint32_t kk = 0; kk < K / 2; ++kk) {  // (token | expert << 11 | weight's 15 low bits << 17)
                    const uint32_t v = r2[kk], w = r2[K / 2 + kk], g0 = v & 0xFFFF, g1 = v >> 16;
                    sw[lm[g0]++] = t | ((g0 % EPC) << 11) | ((w & 0x7FFF) << 17);
                    sw[lm[g1]++] = t | ((g1 % EPC) << 11) | ((w >> 16) << 17);
                }
            }
        }
        for (uint32_t d = q; d < N; d += M) {  // (my destinations' header totals: the M slices' counts, two 16-bit
                                                // lanes a word - a lane's sum <= the destination's entries <= S K)
            const uint32_t* cs = reinterpret_cast<const uint32_t*>(B + O_PSUM + (d / M) * M * BB);
            uint32_t* tw = reinterpret_cast<uint32_t*>(B + O_TOT + (d / M) * BB);
            for (uint32_t i = 0; i < BB / 4; ++i) {
                uint32_t acc = 0;
                for (uint32_t r = 0; r < M; ++r) acc += cs[r * (BB / 4) + i];
                tw[i] = acc;
            }
        }
        for (uint32_t d = 0; d < N; ++d) {
            const uint32_t x = get_arg_val<uint32_t>(dst_arg + 3 * d), y = get_arg_val<uint32_t>(dst_arg + 3 * d + 1);
            const uint32_t pb = B + get_arg_val<uint32_t>(dst_arg + 3 * d + 2);  // (the port's CB 0: the same base)
            if (nq[d]) noc_async_write(stg[d], get_noc_addr(x, y, pb + doff[d]), 4 * nq[d]);
            noc_async_write(B + O_PRE + d * BBP, get_noc_addr(x, y, pb + 64 + BB + q * BBP), BBP);
            noc_inline_dw_write(get_noc_addr(x, y, pb + 4 * q), nq[d]);
            if (d % M == q) noc_async_write(B + O_TOT + (d / M) * BB, get_noc_addr(x, y, pb + 64), BB);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < NPORT; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(port_arg + 2 * i), y = get_arg_val<uint32_t>(port_arg + 2 * i + 1);
            noc_semaphore_inc(get_noc_addr(x, y, ready_sem), 1);
        }
    }
    ts[3] = get_timestamp_32b();
    // (4) while the packets fly: my home tokens' rows (local: I am their home) - n_local / want / n_ret; want rows
    // straight to the plan, n_local to core 0
    uint32_t nr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    {
        uint8_t hop[N * L];  // n_ret's port: the first hop toward a source, its link
        for (uint32_t s = 0; s < N; ++s) {
            for (uint32_t l = 0; l < L; ++l) hop[s * L + l] = s == ME ? 0 : static_cast<uint8_t>(lut[first_hop(ME, s) * 2 + l]);
        }
        uint32_t* wr = reinterpret_cast<uint32_t*>(B + O_WANT);
        const InterleavedAddrGen<true> pgg = {.bank_base_address = plan_addr, .page_size = 4096};
        for (uint32_t i = 0, aa = q; aa < A; ++i, aa += M) {
            uint32_t want[N], nloc = 0;
            for (uint32_t s = 0; s < N; ++s) want[s] = 0;
            for (uint32_t t = aa; t < S; t += A) {
                const uint32_t* r2 = row(t);
                for (uint32_t kk = 0; kk < K / 2; ++kk) {
                    const uint32_t v = r2[kk], s0 = (v & 0xFFFF) / EPC, s1 = (v >> 16) / EPC;
                    if (s0 == ME) {
                        ++nloc;
                    } else {
                        ++want[s0];
                        ++nr[hop[s0 * L + (aa + s0 + ME) % L]];
                    }
                    if (s1 == ME) {
                        ++nloc;
                    } else {
                        ++want[s1];
                        ++nr[hop[s1 * L + (aa + s1 + ME) % L]];
                    }
                }
            }
            for (uint32_t s = 0; s < 32; ++s) wr[i * 32 + s] = s < N ? want[s] : 0;
            noc_async_write(B + O_WANT + i * 128, get_noc_addr(0, pgg) + (32 + aa * 32) * 4, 128);
            noc_inline_dw_write(get_noc_addr(px[0], py[0], B + O_PLAN + (16 + aa) * 4), nloc);
        }
    }
    // (5) every packet of the call: core 0 waits, then wakes the others
    if (q == 0) {
        const uint32_t ca = par ? c1 : c0;
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ca), N);
        noc_semaphore_inc(get_noc_addr(ca), 0u - N);
        for (uint32_t r = 1; r < M; ++r) noc_semaphore_inc(get_noc_addr(px[r], py[r], go_sem), 1);
    } else {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), 1);
    }
    ts[4] = get_timestamp_32b();
    // (6) the headers (every expert's total -> regions; my experts: a contiguous range balanced by rows), then every
    // packet's entries
    const auto sl = TensorAccessor(sl_args, sl_addr, PCAP);
    for (uint32_t o = 0; o < N; ++o) noc_async_read(sl.get_noc_addr(par * N + o), B + O_HDR + o * HBP, HBP);
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t tot[EPC], reg[EPC], ent[N];
    // (every expert's total over the headers: two 16-bit lanes a word - a lane's sum <= N S, one entry a token)
    static_assert(N * S < 65536, "packed expert totals");
    constexpr uint32_t EW = (EPC + 1) / 2;
    uint32_t tw[EW];
    for (uint32_t i = 0; i < EW; ++i) tw[i] = 0;
    for (uint32_t o = 0; o < N; ++o) {
        const uint32_t* ht = reinterpret_cast<const uint32_t*>(B + O_HDR + o * HBP + 64);  // (EPC uint16: BB >= 2 EW words)
        for (uint32_t i = 0; i < EW; ++i) tw[i] += ht[i];
    }
    for (uint32_t e = 0; e < EPC; ++e) tot[e] = (tw[e / 2] >> (16 * (e & 1))) & 0xFFFF;
    uint32_t off = 0, T = 0;
    for (uint32_t e = 0; e < EPC; ++e) {
        T += tot[e];
        reg[e] = off;
        off += (tot[e] + 31) / 32 * 32;
    }
    // my experts: from boundary q to q + 1 - boundary j the expert boundary nearest j T / M entries (whole experts: the
    // nearest, not the next - an instance's work is its entries)
    auto bound = [&](uint32_t j) __attribute__((always_inline)) -> uint32_t {
        if (j == 0) return 0;
        if (j >= M) return EPC;
        for (uint32_t e = 0, c = 0; e < EPC; ++e) {
            const uint32_t c1 = c + tot[e];
            if (c1 * M >= j * T) return 2 * j * T <= (c + c1) * M ? e : e + 1;
            c = c1;
        }
        return EPC;
    };
    const uint32_t e0 = bound(q), e1 = bound(q + 1);
    const uint32_t R0 = e0 < EPC ? reg[e0] : off, R1 = e1 < EPC ? reg[e1] : off;
    // my experts' entries only: per source, per slice, its run of my experts (a slice is sorted by expert) - each read
    // keeps its source's 64 B alignment; run (o, r)'s entry x at run[o M + r] + 4 x. An instance's entries are at most
    // T / M + one expert's (the boundaries are the nearest expert edges): ENT_CAP, the build's worst case - packed
    // routing included (the rest of the chip's entries are other instances')
    uint32_t* run = reinterpret_cast<uint32_t*>(B + O_RUN);
    {
        uint32_t nall = 0, pos = B + O_ENT;
        for (uint32_t o = 0; o < N; ++o) {
            const uint32_t hb = B + O_HDR + o * HBP;
            const uint32_t* nqo = reinterpret_cast<const uint32_t*>(hb);
            ent[o] = 0;
            for (uint32_t r = 0, base = 0; r < M; base += nqo[r], ++r) {
                const uint16_t* pre = reinterpret_cast<const uint16_t*>(hb + 64 + BB + r * BBP);
                const uint32_t x0 = base + pre[e0], n = pre[e1] - pre[e0];
                run[o * M + r] = 0;
                if (!n || nall + n > ENT_CAP) continue;  // (past the worst case: dropped - off > ROWS_CAP or P_OVER)
                const uint32_t src = HBP + 4 * x0, a0 = src & ~63u, len = (src - a0 + 4 * n + 63) & ~63u;
                noc_async_read(sl.get_noc_addr(par * N + o) + a0, pos, len);
                run[o * M + r] = pos + (src - a0) - 4 * x0;
                ent[o] = 1;
                pos += len;
                nall += n;
            }
        }
    }
    // my experts' counts / regions (core 0: the 64-byte span of the page that holds them, the rest of the span other
    // chips' experts: zero) - now, under the entries' reads and the fill
    const uint32_t b0 = (E0 * 4) & ~63u, b1 = ((E0 + EPC) * 4 + 63) & ~63u;
    if (q == 0) {
        uint32_t* cw = reinterpret_cast<uint32_t*>(B + O_CNT);
        uint32_t* rw = reinterpret_cast<uint32_t*>(B + O_RGN);
        for (uint32_t i = 0; i < (b1 - b0) / 4; ++i) cw[i] = rw[i] = 0;
        for (uint32_t e = 0; e < EPC; ++e) {
            cw[E0 + e - b0 / 4] = tot[e];
            rw[E0 + e - b0 / 4] = reg[e];
        }
        const InterleavedAddrGen<true> cgg = {.bank_base_address = cnt_addr, .page_size = NG * 4};
        const InterleavedAddrGen<true> rgg = {.bank_base_address = reg_addr, .page_size = NG * 4};
        noc_async_write(B + O_CNT, get_noc_addr(0, cgg) + b0, b1 - b0);
        noc_async_write(B + O_RGN, get_noc_addr(0, rgg) + b0, b1 - b0);
    }
    const uint32_t W1c = R1 < ROWS_CAP ? R1 : ROWS_CAP, W1 = W1c < R0 + INST_ROWS ? W1c : R0 + INST_ROWS;
    const uint32_t ER = e1 - e0;
    uint16_t* et = ER * S <= LMW ? lm : reinterpret_cast<uint16_t*>(B + O_ET);  // per (my expert, token)
    {  // (32-bit stores: et is 4-byte aligned, S even)
        uint32_t* e32 = reinterpret_cast<uint32_t*>(et);
        for (uint32_t i = 0; i < ER * S / 2; ++i) e32[i] = 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    ts[5] = get_timestamp_32b();
    // my experts' entries: sources in order, each source's slices in order, each slice's run of my experts (one
    // contiguous range: a slice is sorted by expert); an entry carries its token and expert
    // rows: per expert, ordered by (token, home, k): count per (expert, token), prefix, then each entry its row
    for (uint32_t o = 0; o < N; ++o) {
        if (!ent[o]) continue;
        const uint32_t hb = B + O_HDR + o * HBP;
        const uint32_t* nqo = reinterpret_cast<const uint32_t*>(hb);
        for (uint32_t r = 0, base = 0; r < M; base += nqo[r], ++r) {
            if (!run[o * M + r]) continue;
            const uint32_t* eo = reinterpret_cast<const uint32_t*>(run[o * M + r]);
            const uint16_t* pre = reinterpret_cast<const uint16_t*>(hb + 64 + BB + r * BBP);
            const uint32_t x1 = base + pre[e1];
            for (uint32_t x = base + pre[e0]; x < x1; ++x) {
                const uint32_t w = eo[x];
                ++et[(((w >> 11) & 63) - e0) * S + (w & 0x7FF)];
            }
        }
    }
    for (uint32_t e = e0; e < e1; ++e) {
        uint16_t* ete = et + (e - e0) * S;
        uint32_t c = reg[e] - R0;
        for (uint32_t t = 0; t < S; ++t) {
            const uint32_t n = ete[t];
            ete[t] = static_cast<uint16_t>(c);
            c += n;
        }
    }
    ts[6] = get_timestamp_32b();
    uint32_t* tidx = reinterpret_cast<uint32_t*>(B + O_TIDX);
    uint32_t* tab = reinterpret_cast<uint32_t*>(B + O_TAB);
    uint32_t ninj[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t o = 0; o < N; ++o) {
        if (!ent[o]) continue;
        const uint32_t hb = B + O_HDR + o * HBP;
        const uint32_t* nqo = reinterpret_cast<const uint32_t*>(hb);
        const uint32_t os = o * S, lk = ME + o;
        uint32_t hp[L];  // (the port toward home o, per link)
        for (uint32_t l = 0; l < L; ++l) hp[l] = o == ME ? 0 : lut[first_hop(ME, o) * 2 + l];
        for (uint32_t r = 0, base = 0; r < M; base += nqo[r], ++r) {
            if (!run[o * M + r]) continue;
            const uint32_t* eo = reinterpret_cast<const uint32_t*>(run[o * M + r]);
            const uint16_t* pre = reinterpret_cast<const uint16_t*>(hb + 64 + BB + r * BBP);
            const uint32_t x1 = base + pre[e1];
            for (uint32_t x = base + pre[e0]; x < x1; ++x) {
                const uint32_t w = eo[x], t = w & 0x7FF, e = (w >> 11) & 63;
                const uint32_t rr = et[(e - e0) * S + t]++;
                if (rr >= INST_ROWS || R0 + rr >= ROWS_CAP) continue;  // (past the worst case only: never)
                const uint32_t aa = t % A;
                uint32_t dest;
                if (o == ME) {
                    dest = P + aa;
                } else {
                    dest = hp[(aa + lk) % L];
                    ++ninj[dest];
                }
#if !__PROBE_NOSTORE__
                tidx[rr] = os + t;
                tab[2 * rr] = dest | ((w >> 17) << 16);
                tab[2 * rr + 1] = o | (t << 8) | (e << 24);
#else
                (void)os;
#endif
            }
        }
    }
    for (uint32_t e = e0; e < e1; ++e) {  // (padding rows: zero)
        const uint32_t nx = e + 1 < EPC ? reg[e + 1] : off, p1 = nx < W1 ? nx : W1;
        for (uint32_t r = reg[e] + tot[e]; r < p1; ++r) {
            tidx[r - R0] = 0;
            tab[2 * (r - R0)] = 0;
            tab[2 * (r - R0) + 1] = 0;
        }
    }
    if (W1 > R0) {
        const InterleavedAddrGen<true> ig = {.bank_base_address = tidx_addr, .page_size = ROWS_CAP * 4};
        const InterleavedAddrGen<true> tg = {.bank_base_address = tab_addr, .page_size = 256};
        noc_async_write(B + O_TIDX, get_noc_addr(0, ig) + R0 * 4, (W1 - R0) * 4);
        for (uint32_t pg = R0 / 32; pg < W1 / 32; ++pg) {
            noc_async_write(B + O_TAB + (pg - R0 / 32) * 256, get_noc_addr(pg, tg), 256);
        }
    }
    ts[7] = get_timestamp_32b();
    // (7) n_inj / n_ret partials to core 0; core 0 writes counts / regions and the plan's words 0..31 and 600, 601
    {
        uint32_t* pin = reinterpret_cast<uint32_t*>(B + O_PIN + q * 64);
        for (uint32_t i = 0; i < 8; ++i) {
            pin[i] = ninj[i];
            pin[8 + i] = nr[i];
        }
        if (q != 0) {
            noc_async_write(B + O_PIN + q * 64, get_noc_addr(px[0], py[0], B + O_PIN + q * 64), 64);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(px[0], py[0], parts_sem), 1);
        }
    }
    if (q == 0) {
        noc_async_write_barrier();  // (my own n_local inline writes, into my plan words)
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem), M - 1);
        invalidate_l1_cache();
        uint32_t* plan = reinterpret_cast<uint32_t*>(B + O_PLAN);  // (words 16..31: n_local, from the owners)
        for (uint32_t i = 0; i < 16; ++i) plan[i] = 0;
        for (uint32_t aa = A; aa < 16; ++aa) plan[16 + aa] = 0;  // (no accumulator: no owner writes it)
        for (uint32_t r = 0; r < M; ++r) {
            const uint32_t* pr = reinterpret_cast<const uint32_t*>(B + O_PIN + r * 64);
            for (uint32_t i = 0; i < 16; ++i) plan[i] += pr[i];
        }
        uint32_t* rows = reinterpret_cast<uint32_t*>(B + O_ROWS);  // (plan words 592..607)
        for (uint32_t i = 0; i < 16; ++i) rows[i] = 0;
        rows[8] = off;
        rows[9] = off > ROWS_CAP ? 1 : 0;
        const InterleavedAddrGen<true> pgg = {.bank_base_address = plan_addr, .page_size = 4096};
        noc_async_write(B + O_PLAN, get_noc_addr(0, pgg), 128);
        noc_async_write(B + O_ROWS, get_noc_addr(0, pgg) + 592 * 4, 64);
    }
    noc_async_write_barrier();
    ts[8] = get_timestamp_32b();
    if (dbg_addr) {
        uint32_t* d = reinterpret_cast<uint32_t*>(B + O_DBG);
        for (uint32_t i = 0; i < 9; ++i) d[i] = ts[i];
        d[9] = 0;
        for (uint32_t e = e0; e < e1; ++e) d[9] += tot[e];
        d[10] = R0;
        d[11] = R1;
        d[12] = n_my;
        d[13] = B;
        const InterleavedAddrGen<true> dg = {.bank_base_address = dbg_addr, .page_size = 64};
        noc_async_write(B + O_DBG, get_noc_addr(q, dg), 64);
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
#if __DCACHE__
    set_l1_data_cache<false>();
#endif
}
"""
)


def _sub(src, **kw):
    for k, v in kw.items():
        src = src.replace(f"__{k}__", str(int(v)))
    left = src.replace("__attribute__", "")
    assert "__" not in left, f"unfilled kernel placeholder near {left[left.find('__') - 40:left.find('__') + 40]}"
    return src


def inst_cap(N, S, K, EPC, M):
    """A planner instance's entries at most, whatever the routing: the chip's T <= N S min(K, EPC) split at the expert
    edges nearest j T / M - an instance's share plus one expert's (<= N S: a token picks an expert once)."""
    return (-(-N * S * min(K, EPC) // M) + N * S + 31) // 32 * 32


def worst_rows(N, S, K, EPC):
    """A chip's plan rows at most (every source's tokens on its experts), each expert padded to 32."""
    return (N * S * min(K, EPC) + 31) // 32 * 32 + 32 * EPC


def routing_a2a_sizes(*, N, S, K, EPC, A, rows_cap, M=None, TPP=32, port_dests=None):
    """Pure (no device): what one call needs. M planner cores (default min(8, EPC, A)); port_dests: the most
    destinations one port core serves (default N, an upper bound). Returns dict(M, plan_l1 (a planner core's CB),
    port_l1 (a port core's CB, port_dests packets), packet (one packet's bytes at most), slots_dram (a chip's slot
    tensor), local (a planner RISC's counters, bytes), fits, why)."""
    M = M or min(8, EPC, A)
    NG = N * EPC
    HBP = header_bytes(EPC, M)
    PCAP = slot_bytes(S, K, EPC, M)
    NOWN = (A + M - 1) // M
    TSM = NOWN * ((S + A - 1) // A)
    BBP = _al16(2 * (EPC + 1))
    # (an instance's entries and rows: the worst case - or rows_cap when a caller knows its routing's rows (a harness))
    IC = min(inst_cap(N, S, K, EPC, M), rows_cap)
    IR = min(IC + 32 * EPC, rows_cap)
    plan_l1 = (
        _al64((S // TPP) * _al64(TPP * 4 * K))
        + _al64(4 * M * N)
        + _al64(4 * N)
        + N * BBP
        + _al64(4 * TSM * K + 16 * N)
        + N * HBP
        + _al64(4 * IC + 128 * N * M)
        + _al64(4 * N * M)
        + _al64(2 * EPC * S)
        + _al64(4 * IR)
        + _al64(8 * IR)
        + _al64(128 * NOWN)
        + M * 64
        + 128
        + 2 * _al64(4 * (EPC + 32))
        + 64
        + N * _al16(2 * EPC)
        + ((N + M - 1) // M) * M * _al16(2 * EPC)
        + _al64(((N + M - 1) // M) * _al16(2 * EPC))
        + 128
    )
    port_l1 = (port_dests or N) * PCAP + 64
    why = []
    if N > 32 or A > 16 or K % 2 or S % TPP or not 1 <= M <= min(EPC, A, 16):
        why.append(f"shape: N {N} <= 32, A {A} <= 16, K {K} even, S {S} % {TPP}, M {M} <= min(EPC, A, 16)")
    if NG > 1024:
        why.append(f"local counters: N EPC {NG} > 1024")
    if S > 2048 or EPC > 64:
        why.append(f"an entry: S {S} <= 2048 and EPC {EPC} <= 64")
    l1 = 1_400_000  # (a core's usable L1, about)
    if plan_l1 > l1 or port_l1 > l1:
        why.append(f"L1: planner {plan_l1 >> 10} KB, port {port_l1 >> 10} KB")
    return dict(
        M=M,
        inst_cap=IC,
        plan_l1=plan_l1,
        port_l1=port_l1,
        packet=PCAP,
        slots_dram=2 * N * PCAP,
        local=2 * 1024 + 4 * (2 * EPC + 3 * N) + 64,
        fits=not why,
        why="; ".join(why),
    )


def routing_a2a_reserve(mesh_device, *, L=2, torus=(False, False), max_cores=16):
    """The op's persistent L1 (global semaphores: arrival counters, call counts), on up to max_cores planner cores
    and the port cores. Create it before sizing or allocating anything per call (an L1 arena): pass it to
    routing_a2a_planner_prepare(reserved=...)."""
    chips, _, wraps = mesh_plan(mesh_device, L, torus)
    port_core = place_ports(mesh_device, chips)  # {(coord, dir, link index): logical core}
    port_cells = {(c.x, c.y) for c in port_core.values()}
    free = sorted(set(allowed_cores(mesh_device)) - port_cells, key=lambda c: (c[1], c[0]))
    pcells = free[: min(max_cores, len(free))]
    crs = lambda cells: ttnn.CoreRangeSet(  # noqa: E731
        [ttnn.CoreRange(ttnn.CoreCoord(*x), ttnn.CoreCoord(*x)) for x in sorted(cells)]
    )
    c0_ = crs(pcells[:1])
    sems = [ttnn.create_global_semaphore(mesh_device, c0_, 0) for _ in range(2)]  # (arrivals, per call parity)
    sems.append(ttnn.create_global_semaphore(mesh_device, crs(pcells), 0))  # (every planner core's call count)
    sems.append(ttnn.create_global_semaphore(mesh_device, crs(port_cells), 0))  # (every port core's call count)
    sems.append(ttnn.create_global_semaphore(mesh_device, crs(port_cells), 0))  # (port cores: "slices in")
    return dict(
        chips=chips,
        wraps=wraps,
        port_core=port_core,
        pcells=pcells,
        sems=sems,
        addrs=[int(ttnn.get_global_semaphore_address(x)) for x in sems],
        torus=torus,
        L=L,
    )


def routing_a2a_planner_prepare(
    mesh_device,
    *,
    S,
    K,
    EPC,
    A,
    L,
    port_luts,
    n_ports,
    rows_cap,
    tokens_per_page=32,
    torus=(False, False),
    plan_cores=None,
    debug=True,
    dcache=True,
    probe_nostore=False,
    reserved=None,
):
    """The routing all-to-all + local planner. Returns run(own_routing) -> dict of the per-chip tables (the same
    tensors as combine_planner_prepare's: counts, regions, token_index, table, plan), allocated once. own_routing:
    each chip's own tokens' joint routing rows, [S / tokens_per_page, tokens_per_page x 2 K] uint16 row-major (one
    page per tokens_per_page tokens, router_topk's layout), DRAM interleaved. port_luts / n_ports: combine_plan's.
    plan_cores: M (default min(8, EPC, A): core q owns accumulators q, q + M, ... - their home tokens - and a range of
    experts balanced by the call's rows)."""
    import torch

    R, C = tuple(mesh_device.shape)
    N = R * C
    NG = N * EPC
    TPP = tokens_per_page
    if plan_cores is None:
        plan_cores = min(8, EPC, A)
    M = plan_cores
    assert N <= 32 and A <= 16 and L <= 2 and K % 2 == 0 and EPC <= 255 and rows_cap % 32 == 0
    assert NG <= 1024, "local-memory counters"
    assert S <= 2048 and EPC <= 64, "an entry: token (11 bits) | local expert (6 bits) | weight (15 bits, >= 0)"
    assert S % TPP == 0 and 1 <= M <= min(EPC, A, 16), (S, TPP, M, EPC, A)
    rs = reserved if reserved is not None else routing_a2a_reserve(mesh_device, L=L, torus=torus)
    assert rs["L"] == L and tuple(rs["torus"]) == tuple(torus) and len(rs["pcells"]) >= M, "reserved for another setup"
    chips, (wrap_x, wrap_y), port_core = rs["chips"], rs["wraps"], rs["port_core"]
    pcells = rs["pcells"][:M]
    pcs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*x), ttnn.CoreCoord(*x)) for x in pcells])
    c0a, c1a, calls_p, calls_s, ready_a = rs["addrs"]

    HBP = header_bytes(EPC, M)
    BB, BBP = _al16(2 * EPC), _al16(2 * (EPC + 1))
    PCAP = slot_bytes(S, K, EPC, M)
    MAXP = int(ttnn.get_tt_fabric_max_payload_size_bytes()) // 64 * 64
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))

    def dev(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )

    z = lambda *s: torch.zeros(R, C, *s, dtype=torch.int32)  # noqa: E731
    outs = dict(
        counts=dev(z(1, NG)),
        regions=dev(z(1, NG)),
        token_index=dev(z(1, rows_cap)),
        table=dev(z(rows_cap // 32, 64)),
        plan=dev(z(1, PLAN_WORDS)),
    )
    slots = dev(z(2 * N, PCAP // 4))  # page parity N + source: one source's packet
    dbg_t = dev(z(M + 8, 16)) if debug else None
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))  # noqa: E731
    mesh_id = int(node((0, 0)).mesh_id)
    chip_ids = [int(node((i // C, i % C)).chip_id) for i in range(N)]
    virt = lambda xy: mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))  # noqa: E731
    pv = [virt(x) for x in pcells]

    def first_hop(s, d):
        sr, sc, dr, dc = s // C, s % C, d // C, d % C
        if sr != dr:
            if wrap_y:
                return 2 if 2 * ((dr - sr) % R) < R else 3
            return 2 if dr > sr else 3
        if wrap_x:
            return 0 if 2 * ((dc - sc) % C) <= C else 1
        return 0 if dc > sc else 1

    sz = routing_a2a_sizes(N=N, S=S, K=K, EPC=EPC, A=A, rows_cap=rows_cap, M=M, TPP=TPP)
    assert sz["fits"], sz["why"]
    ENT_CAP = sz["inst_cap"]  # (an instance's entries: the worst case, or rows_cap when smaller - a caller's routing)
    pl_bytes = sz["plan_l1"]

    def build(rt_t):
        rt_ct = list(ttnn.TensorAccessorArgs(rt_t).get_compile_time_args())
        sl_ct = list(ttnn.TensorAccessorArgs(slots).get_compile_time_args())
        rt_addr, sl_addr = int(rt_t.buffer_address()), int(slots.buffer_address())
        dbg_addr = int(dbg_t.buffer_address()) if dbg_t is not None else 0
        mesh_desc = ttnn.MeshProgramDescriptor()
        for r in range(R):
            for cc in range(C):
                me = r * C + cc
                ch = (r, cc)
                pd = ttnn.ProgramDescriptor()
                # destination d on the port of its first hop, link d % L; my own packet on the port with the fewest
                serve = {}
                for d in range(N):
                    if d != me:
                        k_ = (ch, DIRS[first_hop(me, d)], d % L)
                        assert k_ in port_core, f"no port {k_}"
                        serve.setdefault(k_, []).append(d)
                mine = sorted(k for k in port_core if k[0] == ch)
                own_k = min(mine, key=lambda k: (len(serve.get(k, [])), DIRS.index(k[1]), k[2]))
                serve.setdefault(own_k, []).append(me)  # (last: SELF)
                senders = [k for k in mine if serve.get(k)]
                where = {}  # destination -> (port core, its packet slot)
                for k_ in senders:
                    for j, d in enumerate(serve[k_]):
                        where[d] = (port_core[k_], j)
                cbs = [
                    ttnn.CBDescriptor(
                        total_size=pl_bytes,
                        core_ranges=pcs,
                        format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, pl_bytes)],
                    )
                ]
                kernels = []
                scs = ttnn.CoreRangeSet([ttnn.CoreRange(port_core[k], port_core[k]) for k in senders])
                for si, k_ in enumerate(senders):
                    dests = serve[k_]
                    self_ = k_ == own_k
                    nr_ = len(dests) - int(self_)
                    core = port_core[k_]
                    ND = len(dests)
                    sd_bytes = ND * PCAP + 64
                    cs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
                    cbs.append(
                        ttnn.CBDescriptor(
                            total_size=sd_bytes,
                            core_ranges=cs,
                            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, sd_bytes)],
                        )
                    )
                    args = [sl_addr, c0a, c1a, pv[0].x, pv[0].y, calls_s, ready_a, mesh_id, dbg_addr, M + si]
                    args += [chip_ids[d] for d in dests]
                    if nr_:
                        peer = chips[ch][k_[1]]["peer"]
                        link = chips[ch][k_[1]]["links"][k_[2]]
                        args += list(ttnn.setup_fabric_connection(node(ch), node(peer), link, pd, core))
                    rta = ttnn.RuntimeArgs()
                    rta[core.x][core.y] = args
                    src = _sub(
                        _SEND,
                        N=N,
                        ME=me,
                        M=M,
                        EPC=EPC,
                        ND=ND,
                        NR=nr_,
                        SELF=int(self_),
                        HBP=HBP,
                        BB=BB,
                        BBP=BBP,
                        PCAP=PCAP,
                        MAXP=MAXP,
                    )
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=src,
                            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                            core_ranges=cs,
                            compile_time_args=sl_ct,
                            runtime_args=rta,
                            config=ttnn.DataMovementConfigDescriptor(
                                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                            ),
                        )
                    )
                assert len(senders) <= 8
                # planner cores (the port cores' packet buffers: CB 0, the same L1 base as the planner's - checked
                # through the debug words)
                rtp = ttnn.RuntimeArgs()
                head = [
                    rt_addr,
                    sl_addr,
                    c0a,
                    c1a,
                    calls_p,
                    int(outs["counts"].buffer_address()),
                    int(outs["regions"].buffer_address()),
                    int(outs["token_index"].buffer_address()),
                    int(outs["table"].buffer_address()),
                    int(outs["plan"].buffer_address()),
                ] + list(port_luts[ch])
                pxy = [w for v in pv for w in (v.x, v.y)]
                dst_args = []
                for d in range(N):
                    core, j = where[d]
                    v = virt((core.x, core.y))
                    dst_args += [v.x, v.y, j * PCAP]  # (the buffer offset: + the CB base, the same on both)
                port_args = [w for k in senders for v in [virt((port_core[k].x, port_core[k].y))] for w in (v.x, v.y)]
                for pi, x in enumerate(pcells):
                    e0, e1 = pi * EPC // M, (pi + 1) * EPC // M
                    rtp[x[0]][x[1]] = head + [pi, e0, e1] + pxy + dst_args + port_args + [dbg_addr, ready_a]
                psrc = _sub(
                    _PLAN,
                    S=S,
                    K=K,
                    EPC=EPC,
                    R=R,
                    C=C,
                    ME=me,
                    A=A,
                    L=L,
                    P=n_ports[ch],
                    TPP=TPP,
                    M=M,
                    HBP=HBP,
                    BB=BB,
                    BBP=BBP,
                    PCAP=PCAP,
                    NPORT=len(senders),
                    ROWS_CAP=rows_cap,
                    ENT_CAP=ENT_CAP,
                    WX=int(wrap_x),
                    WY=int(wrap_y),
                    YT=int(os.environ.get("SPC_YTIE") == "1"),
                    DCACHE=int(dcache),
                    PROBE_NOSTORE=int(probe_nostore),
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=psrc,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=pcs,
                        compile_time_args=rt_ct + sl_ct,
                        runtime_args=rtp,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                        ),
                    )
                )
                pd.cbs = cbs
                pd.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=pcs, initial_value=0) for i in (0, 1, 2)]
                pd.kernels = kernels
                mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, cc), ttnn.MeshCoordinate(r, cc))] = pd
        return mesh_desc

    state = {}

    def run(own_routing):
        k = int(own_routing.buffer_address())
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(own_routing)
            ttnn.freeze_program_hashes(state["desc"])
        ttnn.generic_op(
            [own_routing, slots, outs["counts"], outs["regions"], outs["token_index"], outs["table"], outs["plan"]]
            + ([dbg_t] if dbg_t is not None else []),
            state["desc"],
        )
        return outs

    run.outs = outs
    run.dbg = dbg_t
    run.M = M
    run.slots = slots
    run.cores = [ttnn.CoreCoord(*x) for x in pcells]
    run.token_major = True
    return run
