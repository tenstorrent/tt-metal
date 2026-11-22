# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""kv_push_plan — the planning side of a sparse KV push (GLM-5.3 shared layers) in the sparse 2D dispatch's style:
every row of the TP-deduped block-cyclic KVPE cache goes once per tree link from its owner to the chips whose top-k
reference it (dimension-order drop-off trees, sparse_all_gather_2d), instead of one unicast per requester.

Cache layout (as Iva's sparse push): natural row n lives in block b = n / SC on owner b % N at local row
(b / N) SC + n % SC. Prefix capacity P rows (a multiple of N SC); an owner holds P / N rows, as WOP words of bits
(padded to whole 4 KB pages: PPO pages of 1024 words).

Three steps per chip, all on the device:
  1 want     (want_program)  top-k indices [Q, K] -> this chip's want bitmap, owner-major: owner o's WOP words land in
                             the exchange buffer xbuf [1, 1, N S, 1024] (S = N PPO 4 KB pages per chip) at pages
                             ME S + o PPO + j. NP producer cores bucket their index ranges by bitmap slice into the NC
                             consumer cores' L1 inboxes (NoC writes, counts inline, one semaphore); each consumer
                             sets its 512-word slice in local memory and writes it out. Indices >= P are ignored
                             (SENTINEL 0xFFFFFFFF included).
  2 exchange (exchange_route_words + sparse_all_gather_2d) page ME S + o PPO + j goes to owner o only (a static route
                             word): every owner then holds every requester's slice of its rows at pages
                             q S + o PPO + j.
  3 route    (route_program) owner ME: the N requester slices of each 32-row word -> per-row destination masks (own bit
                             dropped), and for each row with a destination the 2D dispatch's order (X- or Y-first by
                             fewer tree links, ties by a hash) and link (hash bit): a compacted list per core, (mask,
                             row | order << 24 | link << 25) pairs after a 64-byte header (word 0: count).
"""

import ttnn

from .sparse_all_gather_2d import _COMMON, route_words, span_subs


def want_split(n_prod):
    """(WC words per consumer slice, NC consumer cores): slices of whole 64 B over all n_prod cores (local memory)."""
    wc = -(-WORDS // n_prod // 16) * 16
    return wc, -(-WORDS // wc)


WORDS = 32768  # bitmap words per chip (1M rows)
PAGE_WORDS = 1024  # exchange rows: 4 KB


def geometry(N, P, SC):
    """(owned rows per owner, WOP words per owner, PPO pages per owner, S exchange rows per chip). S = N PPO + 1: one
    unused page per chip, so that one owner's pages from the N requesters (q S + o PPO + j) spread over the DRAM banks
    instead of all landing on one (S a multiple of the bank count)."""
    assert P % (N * SC) == 0, f"prefix capacity {P} must be whole stripes of {N} x {SC}"
    owned = P // N
    wop = -(-owned // 32 // PAGE_WORDS) * PAGE_WORDS
    assert N * wop == WORDS, f"N {N} x {wop} words per owner: the want kernel assumes {WORDS} words per chip"
    ppo = wop // PAGE_WORDS
    return owned, wop, ppo, N * ppo + 1


_WANT = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
// both data-movement RISCs of a core are producers (pid = 2 p + RISC, its own index range, outboxes and inbox
// slots); a consumer core's two RISCs each OR half of the inboxes into a local-memory slice, RISC 1 hands its half
// over in L1 and RISC 0 writes the slice out
constexpr uint32_t RISC = __RISC__;
constexpr uint32_t N = __N__, ME = __ME__, P = __P__, SC = __SC__, WOP = __WOP__, PPO = __PPO__, S = __SX__;
constexpr uint32_t NP = __NP__, NPR = 2 * NP, NC = __NC__, WC = __WC__, WORDS = __WORDS__, IPH = __IPH__, KQ = __KQ__;
constexpr uint32_t OB = __OB__, INCAP = __INCAP__;
static_assert(WC % 16 == 0 && OB % 4 == 0 && INCAP % 4 == 0 && IPH % 16 == 0 && NC * WC >= WORDS);
constexpr uint32_t al64(uint32_t x) { return (x + 63) & ~63u; }
// L1 (the same offsets on every core: CB 0 sits at one address on all of them)
constexpr uint32_t O_IDX = 0, O_OUTB = al64(O_IDX + 2 * IPH * 4), O_INB = al64(O_OUTB + 2 * NC * OB * 4);
constexpr uint32_t O_CNT = al64(O_INB + NPR * INCAP * 4), O_SLC = al64(O_CNT + NPR * 4), O_SLC1 = al64(O_SLC + WC * 4);
constexpr uint32_t O_ST = al64(O_SLC1 + WC * 4);
constexpr auto idx_args = TensorAccessorArgs<0>();
constexpr auto x_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
constexpr auto st_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
uint32_t slice[WC];  // (consumer: local memory)

void kernel_main() {
    size_t a = 0;
    const uint32_t idx_addr = get_arg_val<uint32_t>(a++), x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t st_addr = get_arg_val<uint32_t>(a++), p = get_arg_val<uint32_t>(a++);
    const uint32_t g0 = get_arg_val<uint32_t>(a++), g1 = get_arg_val<uint32_t>(a++);  // (my index range)
    uint32_t px[NC], py[NC];
    for (uint32_t i = 0; i < NC; ++i) {
        px[i] = get_arg_val<uint32_t>(a++);
        py[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t pid = 2 * p + RISC;
    const uint32_t B = (get_write_ptr(0) + 63) & ~63u, IX = B + O_IDX + RISC * IPH * 4, OBX = B + O_OUTB + RISC * NC * OB * 4;
    const bool cons = p < NC;
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(B + O_ST + RISC * 64);
    st[1] = get_timestamp_32b();
    if (cons) {
        for (uint32_t i = 0; i < WC; ++i) slice[i] = 0;
    }
    const auto idx = TensorAccessor(idx_args, idx_addr, KQ * 4);
    for (uint32_t g = g0; g < g1;) {
        const uint32_t off = g % KQ, len = KQ - off < g1 - g ? KQ - off : g1 - g;
        noc_async_read(idx.get_noc_addr(g / KQ) + off * 4, IX + (g - g0) * 4, len * 4);
        g += len;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    st[2] = get_timestamp_32b();
    // bucket by consumer slice: entry = bitmap bit address (word << 5 | bit)
    uint32_t obc[NC], sent[NC], ovf = 0;
    for (uint32_t d = 0; d < NC; ++d) obc[d] = sent[d] = 0;
    const uint32_t* ix = reinterpret_cast<const uint32_t*>(IX);
    volatile tt_l1_ptr uint32_t* ob = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(OBX);
    auto flush = [&](uint32_t d, uint32_t n) {
        if (sent[d] + ((n + 3) & ~3u) > INCAP) {
            ovf += n;
            return;
        }
        noc_async_write(OBX + d * OB * 4, get_noc_addr(px[d], py[d], B + O_INB + (pid * INCAP + sent[d]) * 4), ((n * 4) + 15) & ~15u);
        sent[d] += n;
    };
    const uint32_t n_ix = g1 - g0;
#pragma GCC unroll __UNROLL__
    for (uint32_t i = 0; i < n_ix; ++i) {
        const uint32_t v = ix[i];
        if (v >= P) continue;
        const uint32_t blk = v / SC, lr = (blk / N) * SC + (v - blk * SC);
        const uint32_t w = (blk % N) * WOP + (lr >> 5), d = w / WC;
        ob[d * OB + obc[d]] = (w << 5) | (lr & 31);
        if (++obc[d] == OB) {
            flush(d, OB);
            noc_async_writes_flushed();  // (before the outbox is refilled)
            obc[d] = 0;
        }
    }
    st[3] = get_timestamp_32b();
    for (uint32_t d = 0; d < NC; ++d) {
        if (obc[d]) flush(d, obc[d]);
    }
    noc_async_write_barrier();
    for (uint32_t d = 0; d < NC; ++d) noc_inline_dw_write(get_noc_addr(px[d], py[d], B + O_CNT + pid * 4), sent[d]);
    noc_async_write_barrier();
    for (uint32_t d = 0; d < NC; ++d) noc_semaphore_inc(get_noc_addr(px[d], py[d], get_semaphore(0)), 1);
    st[0] = ovf;
    st[4] = get_timestamp_32b();
    if (cons) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), NPR);
        invalidate_l1_cache();
        st[5] = get_timestamp_32b();
        const uint32_t* cnt = reinterpret_cast<const uint32_t*>(B + O_CNT);
        const uint32_t w0 = p * WC, wn = WORDS - w0 < WC ? WORDS - w0 : WC;
        for (uint32_t q = RISC; q < NPR; q += 2) {
            const uint32_t* in = reinterpret_cast<const uint32_t*>(B + O_INB) + q * INCAP;
            const uint32_t n = cnt[q];
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t e = in[i];
                slice[(e >> 5) - w0] |= 1u << (e & 31);
            }
        }
        volatile tt_l1_ptr uint32_t* half = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(B + O_SLC1);
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        if constexpr (RISC == 1) {
            for (uint32_t i = 0; i < wn; ++i) half[i] = slice[i];
            asm volatile("fence" ::: "memory");
            *ready = 1;
        } else {
            while (*ready == 0) invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(B + O_SLC);
            for (uint32_t i = 0; i < wn; ++i) sl[i] = slice[i] | half[i];
            const auto xb = TensorAccessor(x_args, x_addr, 4096);
            for (uint32_t w = w0; w < w0 + wn;) {  // (pieces within one 4 KB page)
                const uint32_t ow = w % WOP, ip = ow % 1024, len = 1024 - ip < w0 + wn - w ? 1024 - ip : w0 + wn - w;
                noc_async_write(B + O_SLC + (w - w0) * 4, xb.get_noc_addr(ME * S + (w / WOP) * PPO + ow / 1024) + ip * 4, len * 4);
                w += len;
            }
        }
        st[6] = get_timestamp_32b();
    }
    const auto stv = TensorAccessor(st_args, st_addr, 64);
    noc_async_write(B + O_ST + RISC * 64, stv.get_noc_addr(pid), 64);
    noc_async_write_barrier();
    noc_async_atomic_barrier();  // (no atomic may land after the next program's launch)
}
"""


_ROUTE = (
    _COMMON
    + r"""
constexpr uint32_t N = __N__, ME = __ME__, WOP = __WOP__, PPO = __PPO__, SX = __SX__;
constexpr uint32_t PROBE = __PROBE__;  // 0 full, 1 reads only, 2 masks + compaction (no order rule)
constexpr auto x_args = TensorAccessorArgs<0>();
constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr uint32_t BW = 16;
static_assert(C == 4 && R <= 8);  // (nibble arithmetic: a row of the mesh is a nibble of the mask)
FORCE_INLINE uint32_t links(uint32_t s) { return (s & 15) + (s >> 4); }
// bits 4 r -> bit r (r < 8)
FORCE_INLINE uint32_t compress8(uint32_t x) {
    x &= 0x11111111u;
    x = (x | (x >> 3)) & 0x03030303u;
    x = (x | (x >> 6)) & 0x000F000Fu;
    return (x | (x >> 12)) & 0xFFu;
}
uint32_t mt[BW * 32];  // (local memory: per-row masks of a block)

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), out_addr = get_arg_val<uint32_t>(1);
    const uint32_t w0 = get_arg_val<uint32_t>(2), w1 = get_arg_val<uint32_t>(3), idx = get_arg_val<uint32_t>(4);
    const uint32_t nw = w1 - w0;
    const uint32_t bm_l1 = (get_write_ptr(0) + 63) & ~63u, out_l1 = bm_l1 + ((N * nw * 4 + 63) & ~63u);
    const auto xb = TensorAccessor(x_args, x_addr, 4096);
    const auto out = TensorAccessor(out_args, out_addr, 64 + nw * 32 * 8);
    for (uint32_t q = 0; q < N; ++q) {  // (nw divides 1024: a core's words never straddle a page)
        noc_async_read(xb.get_noc_addr(q * SX + ME * PPO + w0 / 1024) + (w0 % 1024) * 4, bm_l1 + q * nw * 4, nw * 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    uint32_t n = 0;
    if constexpr (PROBE != 1) {
        const uint32_t* bm = reinterpret_cast<const uint32_t*>(bm_l1);
        const uint32_t r0 = ME / C, c0 = ME % C;
        for (uint32_t k = 0; k < BW * 32; ++k) mt[k] = 0;
        for (uint32_t i0 = 0; i0 < nw; i0 += BW) {
            uint32_t u[BW];
            for (uint32_t j = 0; j < BW; ++j) u[j] = 0;
            for (uint32_t q = 0; q < N; ++q) {
                const uint32_t* p = bm + q * nw + i0;
                const uint32_t bit = 1u << q;
                for (uint32_t j = 0; j < BW; ++j) {
                    uint32_t x = p[j];
                    u[j] |= x;
                    for (; x; x &= x - 1) mt[j * 32 + __builtin_ctz(x)] |= bit;
                }
            }
            for (uint32_t j = 0; j < BW; ++j) {
                for (uint32_t uu = u[j]; uu; uu &= uu - 1) {
                    const uint32_t t = __builtin_ctz(uu);
                    const uint32_t m = mt[j * 32 + t] & ~(1u << ME);
                    mt[j * 32 + t] = 0;
                    if (m == 0) continue;
                    const uint32_t row = (w0 + i0 + j) * 32 + t;
                    uint32_t flags = 0;
                    if constexpr (PROBE == 0) {
                        uint32_t x = m | (m >> 16);
                        x |= x >> 8;
                        x |= x >> 4;
                        const uint32_t cm = x & 15u, rm = compress8(m | (m >> 1) | (m >> 2) | (m >> 3));
                        uint32_t hx = links(xspan(c0, cm & ~(1u << c0), row)), hy = links(yspan(r0, rm & ~(1u << r0), row));
                        for (uint32_t cc = cm | (1u << c0); cc; cc &= cc - 1) {
                            hx += links(yspan(r0, compress8(m >> __builtin_ctz(cc)) & ~(1u << r0), row));
                        }
                        for (uint32_t rr = rm | (1u << r0); rr; rr &= rr - 1) {
                            hy += links(xspan(c0, (m >> (4 * __builtin_ctz(rr))) & 15u & ~(1u << c0), row));
                        }
                        const uint32_t hsh = row * 2654435761u;
                        const uint32_t order = hx < hy ? 0 : (hy < hx ? 1 : (hsh >> 31) & 1);
                        flags = order | (((hsh >> 29) & 1) << 1);
                    }
                    o[16 + 2 * n] = m;
                    o[16 + 2 * n + 1] = row | (flags << 24);
                    ++n;
                }
            }
        }
    }
    o[0] = n;
    noc_async_write(out_l1, out.get_noc_addr(idx), 64 + n * 8);
    noc_async_write_barrier();
}
"""
)

_DM = dict(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default)


def _cores(device, n):
    grid = device.compute_with_storage_grid_size()
    assert n <= grid.x * grid.y
    return [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]


def want_program(device, *, R, C, me, P, SC, indices, xbuf, status, n_prod=64, outbox=128, inbox_cap=512, unroll=1):
    """Step 1 for chip `me` of an R x C mesh. indices [1, 1, Q, K] uint32 row-major (this chip's top-k); xbuf the
    exchange buffer; status [1, 1, 2 n_prod, 16] uint32 (per producer RISC: word 0 entries dropped - must be 0 -,
    words 1..6 phase timestamps)."""
    N = R * C
    _, wop, ppo, sx = geometry(N, P, SC)
    Q, K = int(indices.shape[-2]), int(indices.shape[-1])
    T = Q * K
    WC, NC = want_split(n_prod)
    ip = -(-T // n_prod // 32) * 32  # (per core, both RISCs; 16-index halves keep the DRAM reads 64 B aligned)
    iph = ip // 2
    cores = _cores(device, n_prod)
    phys = [device.worker_core_from_logical_core(c) for c in cores[:NC]]
    l1 = ip * 4 + 2 * NC * outbox * 4 + 2 * n_prod * inbox_cap * 4 + 2 * n_prod * 4 + 2 * WC * 4 + 2 * 64 + 8 * 64
    coords = [v for pc in phys for v in (pc.x, pc.y)]
    cset = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    pd = ttnn.ProgramDescriptor()
    pd.cbs = [
        ttnn.CBDescriptor(
            total_size=l1, core_ranges=cset, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, l1)]
        )
    ]
    pd.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cset, initial_value=0) for i in (0, 1)]
    kernels = []
    for risc in (0, 1):
        subs = dict(
            __RISC__=risc, __N__=N, __ME__=me, __P__=P, __SC__=SC, __WOP__=wop, __PPO__=ppo, __NP__=n_prod, __NC__=NC, __WC__=WC,
            __WORDS__=WORDS, __IPH__=iph, __KQ__=K, __OB__=outbox, __INCAP__=inbox_cap, __SX__=sx, __UNROLL__=unroll,
        )  # fmt: skip
        src = _WANT
        for k_, v_ in subs.items():
            src = src.replace(k_, str(v_))
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            lo, hi = min(T, i * ip + risc * iph), min(T, i * ip + (risc + 1) * iph)
            rt[c.x][c.y] = [
                int(indices.buffer_address()),
                int(xbuf.buffer_address()),
                int(status.buffer_address()),
                i,
                lo,
                hi,
            ] + coords
        dm = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
            noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=src,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cset,
                compile_time_args=[
                    v for t in (indices, xbuf, status) for v in ttnn.TensorAccessorArgs(t).get_compile_time_args()
                ],
                runtime_args=rt,
                config=dm,
            )
        )
    pd.kernels = kernels
    return pd


def route_out_shape(N, P, SC, n_cores):
    """The route step's output: [1, 1, n_cores, 16 + 64 nw] uint32 (nw words of the owner's bitmap per core)."""
    _, wop, _, _ = geometry(N, P, SC)
    nw = wop // n_cores
    return [1, 1, n_cores, 16 + 64 * nw]


def route_program(device, *, R, C, me, torus, P, SC, xbuf, out, n_cores=32, probe=0):
    """Step 3 for owner `me` of an R x C mesh (torus = (wrap X, wrap Y)): xbuf after the exchange -> out
    (route_out_shape)."""
    N = R * C
    _, wop, ppo, sx = geometry(N, P, SC)
    nw = wop // n_cores
    assert nw * n_cores == wop and 1024 % nw == 0 and nw % 16 == 0
    src = _ROUTE
    for k_, v_ in span_subs(R, C, bool(torus[0]) and C > 2, bool(torus[1]) and R > 2).items():
        src = src.replace(k_, v_)
    subs = dict(__R__=R, __C__=C, __S__=sx, __SX__=sx, __GMAX__=max(1, R + C - 3), __RW__=route_words(N), __N__=N, __ME__=me,
                __WOP__=wop, __PPO__=ppo, __PROBE__=probe)  # fmt: skip
    for k_, v_ in subs.items():
        src = src.replace(k_, str(v_))
    cores = _cores(device, n_cores)
    cset = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    l1 = N * nw * 4 + 64 + 64 + nw * 32 * 8 + 64
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [int(xbuf.buffer_address()), int(out.buffer_address()), i * nw, (i + 1) * nw, i]
    pd = ttnn.ProgramDescriptor()
    pd.cbs = [
        ttnn.CBDescriptor(
            total_size=l1, core_ranges=cset, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, l1)]
        )
    ]
    pd.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cset,
            compile_time_args=[v for t in (xbuf, out) for v in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(**_DM),
        )
    ]
    return pd


def exchange_route_words(R, C, P, SC):
    """Step 2's route words, [N, 1, 1, RW S] int64 (host; shard dim 0 over the mesh): page t = o PPO + j of chip me
    goes to owner o alone (its own owner's pages nowhere)."""
    import torch

    N = R * C
    _, _, ppo, S = geometry(N, P, SC)
    RW = route_words(N)
    w = torch.zeros(N, 1, 1, RW * S, dtype=torch.int64)
    for me in range(N):
        for t in range(S):
            o = t // ppo
            m = 0 if o == me or o >= N else 1 << o
            h = (t * 2654435761) & 0xFFFFFFFF
            f = ((h >> 31) & 1) | (((h >> 29) & 1) << 1)
            if RW == 2:
                w[me, 0, 0, 2 * t], w[me, 0, 0, 2 * t + 1] = m - (1 << 32 if m >= 1 << 31 else 0), f
            else:
                w[me, 0, 0, t] = m | (f << 16)
    return w
