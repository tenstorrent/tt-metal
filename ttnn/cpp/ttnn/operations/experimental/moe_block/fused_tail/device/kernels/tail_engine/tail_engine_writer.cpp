// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailEngine engine core, RISCV_1 on NoC1: my share's and my E tile's way to my row, the partner exchange, the
// credits, the CB pushes, the output writes. The landing, ring, E and exchange slots are CBs at the same address on
// every engine core of every chip; their bases are read before this kernel pushes any of them. Token loops: the
// kernel's own loop (tail_engine_common.hpp: the global step, the gate a loop, the loop's thresholds).

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_args.hpp"
#include "tail_engine_common.hpp"
#include "api/debug/kernel_profiler.hpp"

constexpr auto o_args = TensorAccessorArgs<11>();
// token loops: their count and, a loop, the output's rows that much further in every bank (bytes)
constexpr uint32_t LOOPS = get_compile_time_arg_val(o_args.next_compile_time_args_offset());
constexpr uint32_t SOB = get_compile_time_arg_val(o_args.next_compile_time_args_offset() + 1);

void kernel_main() {
    mb_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    const uint32_t land = get_write_ptr(CB_L0), ring = get_write_ptr(CB_A);
    const uint32_t part = get_write_ptr(CB_PART), xrecv = get_write_ptr(CB_XRECV);
    size_t a = 0;
    const uint32_t p = mb_args::arg(a++), pn = mb_args::arg(a++), f0 = mb_args::arg(a++);
    const uint32_t fn = mb_args::arg(a++), ak0 = mb_args::arg(a++), ak1 = mb_args::arg(a++);
    const uint32_t lx = mb_args::arg(a++), rx = mb_args::arg(a++), cx = mb_args::arg(a++);
    const uint32_t o_addr = mb_args::arg(a++), NTO = mb_args::arg(a++);
    const uint32_t SHMAX = mb_args::arg(a++);
    const uint32_t mx = mb_args::arg(a++), my = mb_args::arg(a++), py = mb_args::arg(a++);
    const uint32_t words = get_write_ptr(CB_WD), nst = mb_args::arg(a++), nown = mb_args::arg(a++);
    const uint32_t ord = a, xs = ord + NCK,
                   wkb = xs + GX;  // (chunk order; my row's NoC x by column; W' K range bounds)
    // (the steps I watch: a landed row tile's last share is mine - its port's one fused increment, behind the row's
    // other shares on that link - so I count it landed for the other cores of its part: s | c0 << 8 | c1 << 16)
    const uint32_t nrl = mb_args::arg(wkb + PAIRS + 1), rls = wkb + PAIRS + 2;
    const uint32_t gw = mb_args::arg(rls + NSTMAX);  // (the gate's lead core: x | y << 16)
    auto word = [&](uint32_t kind, uint32_t s) { return words + 16 * (kind * NSTMAX + s); };  // (0 landed, 1 SA, 2 SP)
    auto wv = [&](uint32_t kind, uint32_t s) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word(kind, s)); };
    // (the row's shares of step s in: a word a column, set to the loop's count by its share's multicast right behind
    // the data - same NoC, same VC, so in order: no write barrier between, the chain's next core goes on seeing it)
    auto sa = [&](uint32_t s, uint32_t c) { return words + 16 * (3 * NSTMAX + 1 + s * GX + c); };
    auto sav = [&](uint32_t s, uint32_t c) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sa(s, c)); };
    auto row = [&](uint32_t addr) { return get_noc_multicast_addr(rx, my, lx, my, addr); };  // (NOC1: right end first)
    const uint32_t n = ak1 - ak0;
    uint32_t wr = 0;  // (W' K ranges pushed to the compute: once a call - resident for every loop)
    for (uint32_t l = 0; l < LOOPS; ++l) {
        const uint32_t g0 = l * nst;  // (loop l's first step, global: the slots and the cumulative semaphores)
        // (loop 0: this call's words zeroed on every engine core; loop l > 0: every engine writer and port reader done
        // with loop l - 1 - the gate's go, TL-ZGATE / TL-LGATE)
        {
            volatile tt_l1_ptr uint32_t* go = sem(SZGO);
            while (*go == 0) {
                invalidate_l1_cache();
            }
            *go = 0;  // (re-arm: one set a gate)
        }
        // (loop l - 1's landing slots read by everyone: freed whole - the compute's pointer back at the base)
        if (l > 0) {
            cb_wait_front(CB_L0, NSTMAX * SHMAX);
            cb_pop_front(CB_L0, NSTMAX * SHMAX);
        }
        const uint32_t ep = l + 1;  // (loop l's words: a step's word reaches ep x its count a loop)
        *wv(1, 0) = ep;             // (the shares' words' source)
        const auto og = TensorAccessor(o_args, o_addr + l * SOB, 2048);
        uint32_t cs = 0, owned = 0, rdone = 0;
        auto pump = [&]() {
            invalidate_l1_cache();
            for (uint32_t i = 0; i < nrl; ++i) {  // (any order: the directions land their row tiles independently)
                const uint32_t e = mb_args::arg(rls + i), rs = e & 0xFF;
                if ((rdone >> i & 1) || *wv(0, rs) < ep) {
                    continue;
                }
                for (uint32_t c = e >> 8 & 0xFF; c < (e >> 16); ++c) {
                    if (c != cx) {
                        noc_semaphore_inc(get_noc_addr(mb_args::arg(xs + c), my, word(0, rs)), 1);
                    }
                }
                rdone |= 1u << i;
            }
            while (owned < nown && cb_pages_available_at_front(CB_L0, SHMAX * (owned + 1))) {  // (my own share tilized)
                noc_semaphore_inc(get_noc_addr(word(0, owned)), 1);
                ++owned;
            }
            while (wr < PAIRS && (*sem(SW) >> wr & 1)) {  // (W' K range wr landed: to the compute, in K order)
                const uint32_t m = (mb_args::arg(wkb + wr + 1) - mb_args::arg(wkb + wr)) * pn;
                cb_reserve_back(CB_W, m);
                cb_push_back(CB_W, m);
                ++wr;
            }
            // my share of step cs: landed, its ring slot free on the whole row, my turn in the row (column order)
            const uint32_t gs = g0 + cs;
            if (cs >= nst || *wv(0, cs) < ep || (gs >= NSLOT && *sem(SFREE) < GX * (gs - NSLOT + 1))) {
                return;
            }
            if (cx > 0 && *sav(cs, cx - 1) < ep) {
                return;
            }
            DeviceZoneScopedN("TL-SHARE");
            const uint32_t src = land + cs * SHMAX * 2048, dst = ring + ((gs % NSLOT) * KH + ak0) * 2048;
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
        // my finished columns of step s; then its ring / E slots are free here (FREE)
        auto write_out = [&](uint32_t s) {
            const uint32_t t = step_t(s, p), R = mb_args::arg(ord + t / RT) * RT + t % RT;
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
            const uint32_t gs = g0 + s;
            {
                DeviceZoneScopedN("TL-AWAIT");
                for (uint32_t c = 0; c < GX; ++c) {
                    wait(sav(s, c), ep);
                }
            }
            cb_reserve_back(CB_A, KH);
            cb_push_back(CB_A, KH);
            {  // my E tile to every core of my row (unicasts: 4 KB - no multicast path to reserve), once its E slot is
               // free
                wait_front(CB_PMY, 1);
                if (gs >= NSLOT) {
                    wait(sem(SFREE), GX * (gs - NSLOT + 1));
                }
                const uint32_t src = get_read_ptr(CB_PMY), dst = part + ((gs % NSLOT) * GX + cx) * 4096;
                for (uint32_t i = 0; i < GX; ++i) {
                    noc_async_write(src, get_noc_addr(mb_args::arg(xs + i), my, dst), 4096);
                }
                barrier();
                for (uint32_t i = 0; i < GX; ++i) {
                    noc_semaphore_inc(get_noc_addr(mb_args::arg(xs + i), my, word(2, s)), 1);
                }
                cb_pop_front(CB_PMY, 1);
            }
            {
                DeviceZoneScopedN("TL-PWAIT");
                wait(wv(2, s), ep * GX);
            }
            cb_reserve_back(CB_PART, GX);
            cb_push_back(CB_PART, GX);
            {  // the partner's columns' partials + my half's sum of squares to its XRECV slot of step gs (a column
               // unicast; the slot g % NSTMAX: the partner's loop l - 1 slots all consumed - its writer counted in
               // at the gate only after its compute's last out)
                wait_front(CB_XSEND, XT);
                noc_async_write(
                    get_read_ptr(CB_XSEND), get_noc_addr(mx, py, xrecv + (gs % NSTMAX) * XT * 4096), XT * 4096);
                barrier();
                noc_semaphore_inc(get_noc_addr(mx, py, get_semaphore(SX)), 1);
                cb_pop_front(CB_XSEND, XT);
            }
            {
                DeviceZoneScopedN("TL-XWAIT");
                wait(sem(SX), gs + 1);
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
        if (l + 1 < LOOPS) {  // (done with loop l - its landing slots, words and E / ring slots: count in on the gate)
            noc_semaphore_inc(get_noc_addr(gw & 0xFFFF, gw >> 16, get_semaphore(SZ)), 1);
            noc_async_atomic_barrier();
        }
    }
}

