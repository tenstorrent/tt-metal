// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailEngine engine core, RISCV_1 on NoC1: my share's and my E tile's way to my row, the partner exchange, the
// credits, the CB pushes, the output writes. The landing, ring, E and exchange slots are CBs at the same address on
// every engine core of every chip; their bases are read before this kernel pushes any of them.

#include "tail_engine_common.hpp"
#include "api/debug/kernel_profiler.hpp"

constexpr auto o_args = TensorAccessorArgs<11>();

void kernel_main() {
    const uint32_t land = get_write_ptr(CB_L0), ring = get_write_ptr(CB_A);
    const uint32_t part = get_write_ptr(CB_PART), xrecv = get_write_ptr(CB_XRECV);
    size_t a = 0;
    const uint32_t p = get_arg_val<uint32_t>(a++), pn = get_arg_val<uint32_t>(a++), f0 = get_arg_val<uint32_t>(a++);
    const uint32_t fn = get_arg_val<uint32_t>(a++), ak0 = get_arg_val<uint32_t>(a++), ak1 = get_arg_val<uint32_t>(a++);
    const uint32_t lx = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), cx = get_arg_val<uint32_t>(a++);
    const uint32_t o_addr = get_arg_val<uint32_t>(a++), NTO = get_arg_val<uint32_t>(a++);
    const uint32_t SHMAX = get_arg_val<uint32_t>(a++);
    const uint32_t mx = get_arg_val<uint32_t>(a++), my = get_arg_val<uint32_t>(a++), py = get_arg_val<uint32_t>(a++);
    const uint32_t words = get_write_ptr(CB_WD), nst = get_arg_val<uint32_t>(a++), nown = get_arg_val<uint32_t>(a++);
    const uint32_t ord = a, xs = ord + NCK,
                   wkb = xs + GX;  // (chunk order; my row's NoC x by column; W' K range bounds)
    // (the steps I watch: a landed row tile's last share is mine - its port's one fused increment, behind the row's
    // other shares on that link - so I count it landed for the other cores of its part: s | c0 << 8 | c1 << 16)
    const uint32_t nrl = get_arg_val<uint32_t>(wkb + PAIRS + 1), rls = wkb + PAIRS + 2;
    uint32_t rdone = 0;
    auto word = [&](uint32_t kind, uint32_t s) { return words + 16 * (kind * NSTMAX + s); };  // (0 landed, 1 SA, 2 SP)
    auto wv = [&](uint32_t kind, uint32_t s) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word(kind, s)); };
    // (the row's shares of step s in: a word a column, set to the call count by its share's multicast right behind
    // the data - same NoC, same VC, so in order: no write barrier between, the chain's next core goes on seeing it)
    auto sa = [&](uint32_t s, uint32_t c) { return words + 16 * (3 * NSTMAX + 1 + s * GX + c); };
    auto sav = [&](uint32_t s, uint32_t c) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sa(s, c)); };
    {  // (this call's words zeroed on every engine core: the zero gate's go - TL-ZGATE)
        volatile tt_l1_ptr uint32_t* go = sem(SZGO);
        while (*go == 0) {
            invalidate_l1_cache();
        }
        *go = 0;  // (re-arm: one set a call)
    }
    constexpr uint32_t ep = 1;  // (this call's words: a step's word reaches ep x its count)
    *wv(1, 0) = ep;             // (the shares' words' source)
    const auto og = TensorAccessor(o_args, o_addr, 2048);
    auto row = [&](uint32_t addr) { return get_noc_multicast_addr(rx, my, lx, my, addr); };  // (NOC1: right end first)
    const uint32_t n = ak1 - ak0;
    uint32_t cs = 0, wr = 0, owned = 0;
    auto pump = [&]() {
        invalidate_l1_cache();
        for (uint32_t i = 0; i < nrl; ++i) {  // (any order: the directions land their row tiles independently)
            const uint32_t e = get_arg_val<uint32_t>(rls + i), rs = e & 0xFF;
            if ((rdone >> i & 1) || *wv(0, rs) < ep) {
                continue;
            }
            for (uint32_t c = e >> 8 & 0xFF; c < (e >> 16); ++c) {
                if (c != cx) {
                    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(xs + c), my, word(0, rs)), 1);
                }
            }
            rdone |= 1u << i;
        }
        while (owned < nown && cb_pages_available_at_front(CB_L0, SHMAX * (owned + 1))) {  // (my own share tilized)
            noc_semaphore_inc(get_noc_addr(word(0, owned)), 1);
            ++owned;
        }
        while (wr < PAIRS && (*sem(SW) >> wr & 1)) {  // (W' K range wr landed: to the compute, in K order)
            const uint32_t m = (get_arg_val<uint32_t>(wkb + wr + 1) - get_arg_val<uint32_t>(wkb + wr)) * pn;
            cb_reserve_back(CB_W, m);
            cb_push_back(CB_W, m);
            ++wr;
        }
        // my share of step cs: landed, its ring slot free on the whole row, my turn in the row (column order)
        if (cs >= nst || *wv(0, cs) < ep || (cs >= NSLOT && *sem(SFREE) < GX * (cs - NSLOT + 1))) {
            return;
        }
        if (cx > 0 && *sav(cs, cx - 1) < ep) {
            return;
        }
        DeviceZoneScopedN("TL-SHARE");
        const uint32_t src = land + cs * SHMAX * 2048, dst = ring + ((cs % NSLOT) * KH + ak0) * 2048;
        noc_async_write_multicast_loopback_src(src, row(dst), n * 2048, GX);
        noc_semaphore_set_multicast_loopback_src(word(1, 0), row(sa(cs, cx)), GX);
        ++cs;
    };
    auto wait = [&](volatile tt_l1_ptr uint32_t* w, uint32_t m) {
        while (*w < m) {
            pump();
        }
    };
    auto barrier = [&]() {  // (my writes acked - the chain's next share of mine meanwhile: a hop waits on my pump)
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            pump();
        }
    };
    auto wait_front = [&](uint32_t cb, uint32_t m) {
        while (!cb_pages_available_at_front(cb, m)) {
            pump();
        }
        cb_wait_front(cb, m);
    };
    auto write_out = [&](uint32_t s) {  // my finished columns of step s; then its ring / E slots are free here (FREE)
        const uint32_t t = step_t(s, p), R = get_arg_val<uint32_t>(ord + t / RT) * RT + t % RT;
        wait_front(CB_OUT, fn);
        const uint32_t src = get_read_ptr(CB_OUT);
        for (uint32_t j = 0; j < fn; ++j) {
            noc_async_write(src + j * 2048, og.get_noc_addr(R * NTO + f0 + j), 2048);
        }
        barrier();
        cb_pop_front(CB_OUT, fn);
        if (GX > 1) {
            noc_semaphore_inc_multicast(row(get_semaphore(SFREE)), 1, GX - 1);
        }
        noc_semaphore_inc(get_noc_addr(get_semaphore(SFREE)), 1);
    };
    for (uint32_t s = 0; s < nst; ++s) {
        {
            DeviceZoneScopedN("TL-AWAIT");
            for (uint32_t c = 0; c < GX; ++c) {
                wait(sav(s, c), ep);
            }
        }
        cb_reserve_back(CB_A, KH);
        cb_push_back(CB_A, KH);
        {  // my E tile to every core of my row (unicasts: 4 KB - no multicast path to reserve), once its E slot is free
            wait_front(CB_PMY, 1);
            if (s >= NSLOT) {
                wait(sem(SFREE), GX * (s - NSLOT + 1));
            }
            const uint32_t src = get_read_ptr(CB_PMY), dst = part + ((s % NSLOT) * GX + cx) * 4096;
            for (uint32_t i = 0; i < GX; ++i) {
                noc_async_write(src, get_noc_addr(get_arg_val<uint32_t>(xs + i), my, dst), 4096);
            }
            barrier();
            for (uint32_t i = 0; i < GX; ++i) {
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(xs + i), my, word(2, s)), 1);
            }
            cb_pop_front(CB_PMY, 1);
        }
        {
            DeviceZoneScopedN("TL-PWAIT");
            wait(wv(2, s), ep * GX);
        }
        cb_reserve_back(CB_PART, GX);
        cb_push_back(CB_PART, GX);
        {  // the partner's columns' partials + my half's sum of squares to its XRECV slot of step s (a column unicast)
            wait_front(CB_XSEND, XT);
            noc_async_write(get_read_ptr(CB_XSEND), get_noc_addr(mx, py, xrecv + s * XT * 4096), XT * 4096);
            barrier();
            noc_semaphore_inc(get_noc_addr(mx, py, get_semaphore(SX)), 1);
            cb_pop_front(CB_XSEND, XT);
        }
        {
            DeviceZoneScopedN("TL-XWAIT");
            wait(sem(SX), s + 1);
        }
        cb_reserve_back(CB_XRECV, XT);
        cb_push_back(CB_XRECV, XT);
        if (s > 0) {
            write_out(s - 1);
        }
    }
    if (nst > 0) {
        write_out(nst - 1);
    }
    while (wr < PAIRS || cs < nst || owned < nown || rdone != (nrl ? (2u << (nrl - 1)) - 1 : 0u)) {
        pump();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
