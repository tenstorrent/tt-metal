// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The shared expert's matmul cores' NCRISC kernel.
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/se_defs.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t GY = get_compile_time_arg_val(0), MT = get_compile_time_arg_val(1), KB = get_compile_time_arg_val(2);
constexpr uint32_t NSG = get_compile_time_arg_val(3), NPG = get_compile_time_arg_val(4),
                   NCH = get_compile_time_arg_val(5);
constexpr uint32_t KBD = get_compile_time_arg_val(6), NSD = get_compile_time_arg_val(7),
                   NPD = get_compile_time_arg_val(8);
constexpr uint32_t NWS = get_compile_time_arg_val(9), WSLOT = get_compile_time_arg_val(10),
                   WTB = get_compile_time_arg_val(11);
constexpr uint32_t WV = get_compile_time_arg_val(12), WFREE = get_compile_time_arg_val(13);
constexpr auto w_args = TensorAccessorArgs<14>();
constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto p_args = TensorAccessorArgs<SE_PREP_CT0>();  // (the dispatch buffer [N S, W] row-major)
#ifdef SE_RTR
FORCE_INLINE volatile tt_l1_ptr uint32_t* ew(uint32_t o) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_EW + o);
}
constexpr auto rx_args = TensorAccessorArgs<SE_RTR_CT0>();  // (the chip's x shard [rows, H / TP], DRAM)
constexpr auto rw_args = TensorAccessorArgs<rx_args.next_compile_time_args_offset()>();  // (W_r^T [E, H / TP])
#endif
constexpr uint32_t CB_WIN = 1, NB = NPG * NSG + NPD * NSD;  // (W blocks a chunk)
constexpr uint32_t NSD1 = NSD ? NSD : 1;                    // (never divided by 0)
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
// the column's W stream: block b of a chunk (gate / up pass b / NSG, step b % NSG; then down) held by row b % GY,
// multicast down the column into W slot (round % NWS); every core pushes what lands and credits the slot's next holder
// (the args equal on every core of the chip go once, as common args [S, NC, those NC values]: arg i is the core's
// own arg i below S, common below S + NC, the core's arg i - NC past them - every index stays the full list's)
FORCE_INLINE uint32_t se_ra(uint32_t i) {
    const uint32_t s = get_common_arg_val<uint32_t>(0), nc = get_common_arg_val<uint32_t>(1);
    return i < s        ? get_arg_val<uint32_t>(i)
           : i < s + nc ? get_common_arg_val<uint32_t>(2 + i - s)
                        : get_arg_val<uint32_t>(i - nc);
}
void kernel_main() {
    size_t a = 0;
    // (mine first; from ty to the tail the chip's: se_ra's common span)
    const uint32_t cy = se_ra(a++), cxv = se_ra(a++);
    const uint32_t wbase = se_ra(a++), n0d = se_ra(a++);
    const uint32_t wbank = se_ra(a++);  // (the DRAM bank holding my W-own region)
#ifdef SE_RTR
    const uint32_t R0 = a;  // (the router logits' args)
    a += 10 + SE_RD * SE_MTR;
#endif
    const uint32_t wg0 = a, wd0 = wg0 + NPG;  // pass widths (gate / up, down)
    a = wd0 + NPD;
    const uint32_t ty = se_ra(a++), by = se_ra(a++);
    const uint32_t w_addr = se_ra(a++), wown = se_ra(a++), win = se_ra(a++);
    const uint32_t words = se_ra(a++), y_addr = se_ra(a++);
    const uint32_t ys0 = a;                                     // (the column's NoC y)
    const uint32_t sps = se_ra(ys0 + GY), ord0 = ys0 + GY + 2;  // (steps a shard; the shard order)
    // round w's block (natural index): a gate / up pass's steps in this chip's processing order, then down's
    auto bsize = [&](uint32_t b) {  // (tiles)
        return b < NPG * NSG ? KB * se_ra(wg0 + b / NSG) : KBD * se_ra(wd0 + (b - NPG * NSG) / NSD1);
    };
    // (a chunk's round -> block, and my blocks' offsets in my W-own, tabled once: computed in place, the round's block
    // (4 runtime divisions - sps is a runtime arg) and the offset's walk over my blocks ran on every pass of the
    // stream's loop below (the prefetch scan, the send's skip, the credit) - ~340 cycles a pass, a W round ~2.7
    // passes; tabled ~285 (K3 head handshake-only gate / up 48.8 -> 41.1 us, traced head ~-4 us; GLM-5.3 -5.6 us))
    static_assert(NB <= 1024, "W rounds a chunk");
    uint16_t rblk[NB], woffs[NB / GY + 1];
    for (uint32_t c = 0; c < NB; ++c) {
        uint32_t b = c;
        if (c < NPG * NSG) {
            const uint32_t k = c / (NPG * sps), rem = c % (NPG * sps), pg = rem / sps, st = rem % sps;
            b = pg * NSG + se_ra(ord0 + k) * sps + st;  // (shard-major: shard, pass, step)
        }
        rblk[c] = b;
    }
    for (uint32_t kb = 0, o = 0; cy + kb * GY < NB; ++kb) {  // (my blocks: cy, cy + GY, ..)
        woffs[kb] = o;
        o += bsize(cy + kb * GY);
    }
    auto blk = [&](uint32_t w) -> uint32_t { return rblk[w % NB]; };
    const auto yg = TensorAccessor(y_args, y_addr, 2048);
    // my blocks from DRAM, once: my W-own is one contiguous region in one bank (per-bank layout, wbase its offset),
    // read block by block in large bursts, every read issued now - the k-th block's under transaction id
    // min(k, 14) + 1; a block is sent once its reads landed (the column streams while the weights are still arriving)
#ifdef SE_RTR
    // the router's logits on the engine: the row's first core reads the row's x (MT row tiles, the chip's K slice) and
    // multicasts it along the row; the column's first core reads the column's W_r^T expert tiles and multicasts them
    // down the column; then the compute's partial logits go to the router cores' HOLD (the main loop below)
    // (args at R0: x, W, my column's expert tiles (valid count, first), RX / RW addresses, the row's NoC x (far end,
    // first core), my row's NoC y, the router cores' NoC [RB], their HOLD, their counter, my column has experts)
    const uint32_t etc = se_ra(R0 + 2), re0 = se_ra(R0 + 3);
    const uint32_t rxa = se_ra(R0 + 4), rwa = se_ra(R0 + 5);
    {
        // K chunks of SE_KC tiles, at most two rounds, a buffer each (no slot reused: no acks): the row's first core's
        // BRISC multicasts the row's x along the row (NOC0), the column's first core multicasts its W_r^T expert tiles
        // down the column (here, NOC1 - same-NoC row and column multicasts stall each other); a lead reads round
        // rd + 1 while round rd's multicast goes; every core pushes a round once its words say it landed
        DeviceZoneScopedN("SE-RTRLD");
        // (the column's W_r^T lead on row NoC x % GY, not row 0: the NOC1 read responses come back along the
        // lead's row (-y, then -x), and 11 leads on one row shared its links - ~3 us a round's reads vs ~0.6 for the
        // x leads, who sit on different rows)
        const uint32_t wly = se_ra(ys0 + cxv % GY);
        const bool lw = etc && cy == cxv % GY;
        const bool hw = se_ra(R0 + 9);
        const auto wg2 = TensorAccessor(rw_args, se_ra(R0 + 1), 2048);
        constexpr uint32_t NRD = SE_KTO / SE_KC, WS = SE_ETC * SE_KC * 2048;
        // (two buffers: past two rounds a buffer is reused - every receiver acks a round once its compute popped it
        // (its next reserve went through): +1 on the row's x lead and the column's W lead, who wait for all of them)
        const uint32_t xlx = se_ra(R0 + 7), xly = se_ra(R0 + 8);
        for (uint32_t rd = 0; rd < NRD; ++rd) {
            cb_reserve_back(SE_RX, SE_KC * SE_MTR);
            cb_reserve_back(SE_RW, SE_ETC * SE_KC);
            if (NRD > 2 && rd >= 2) {
                noc_semaphore_inc(get_noc_addr(xlx, xly, SE_EW + XACK_ID), 1);
                if (hw) {
                    noc_semaphore_inc(get_noc_addr(cxv, wly, SE_EW + WACK_ID), 1);
                }
            }
            if (lw) {  // (the column's first core: the chunk of its expert tiles of W_r^T down the column)
                const uint32_t wa = rwa + (rd & 1) * WS;
                if (NRD > 2 && rd >= 2) {
                    while (*ew(WACK_ID) < (rd - 1) * GY) {
                        invalidate_l1_cache();
                    }
                }
                for (uint32_t e = 0; e < etc; ++e) {
                    for (uint32_t k = 0; k < SE_KC; ++k) {
                        noc_async_read(
                            wg2.get_noc_addr((re0 + e) * SE_KTO + rd * SE_KC + k), wa + (e * SE_KC + k) * 2048, 2048);
                    }
                }
                noc_async_read_barrier();
                noc_async_write_multicast_loopback_src(
                    wa, get_noc_multicast_addr(cxv, by, cxv, ty, wa), etc * SE_KC * 2048, GY);
                noc_async_writes_flushed();
                *ew(RWR_ID) = rd + 1;
                noc_semaphore_set_multicast_loopback_src(
                    (SE_EW + RWR_ID), get_noc_multicast_addr(cxv, by, cxv, ty, (SE_EW + RWR_ID)), GY);
            }
            while (*ew(RXR_ID) < rd + 1) {
                invalidate_l1_cache();
            }
            cb_push_back(SE_RX, SE_KC * SE_MTR);
            while (hw && *ew(RWR_ID) < rd + 1) {
                invalidate_l1_cache();
            }
            cb_push_back(SE_RW, SE_ETC * SE_KC);
        }
        noc_async_writes_flushed();  // (my sets' source words read: re-arm them)
        *ew(RXR_ID) = 0;             // (re-armed: no more sets come this call - the x lead sets from its own word, RXS)
        *ew(RWR_ID) = 0;
        if (NRD > 2 && lw) {  // (every ack landed: the last were waited for)
            noc_semaphore_inc(get_noc_addr(SE_EW + WACK_ID), 0u - (NRD - 2) * GY);
            noc_async_atomic_barrier();
        }
    }
    {  // my partial logits (the compute's, a few us later) to their router cores' HOLD - before my W stream starts
        DeviceZoneScopedN("SE-RTROUT");
        cb_wait_front(SE_RO, SE_ETC * SE_MTR);
        // (row tile r of mine: my own owner's block into its router core's HOLD, another owner's straight into the
        // port sub-block it leaves from - (NoC, base, count) a row tile; the count ETC tiles, after they landed)
        const uint32_t ro = get_read_ptr(SE_RO);
        for (uint32_t r = 0; r < SE_MTR; ++r) {
            const uint32_t dw = se_ra(R0 + 10 + 3 * r), dx = dw & 0xFFFF, dy = dw >> 16;
            const uint32_t base = se_ra(R0 + 11 + 3 * r), ctr = se_ra(R0 + 12 + 3 * r);
            for (uint32_t e = 0; e < etc; ++e) {
                noc_async_write(ro + (e * SE_MTR + r) * 4096, get_noc_addr(dx, dy, base + (re0 + e) * 4096), 4096);
            }
            noc_async_write_barrier();  // (landed before the count)
            if (etc) {
                noc_semaphore_inc(get_noc_addr(dx, dy, ctr), etc);
            }
        }
        cb_pop_front(SE_RO, SE_ETC * SE_MTR);
    }
#ifdef SE_RDONE
    {  // my router phase is over - x rows / W_r^T read, the partials out: +1 on every port's "router done" word (the
       // ports credit this chip's arrival zones / tell the latent's senders "started" once every core's is: the
       // router's buffers overlay them - rtr_overlay)
        const uint32_t t0 = ord0 + se_ra(ys0 + GY + 1), np_ = se_ra(t0), rdw = se_ra(t0 + 3 + 2 * np_);
        for (uint32_t i = 0; i < np_; ++i) {
            noc_semaphore_inc(get_noc_addr(se_ra(t0 + 2 + 2 * i), se_ra(t0 + 3 + 2 * i), rdw), 1);
        }
    }
#endif
#endif
    {
        {  // (after my core's x load: the DRAM serves the x first - the gather is waiting on it)
            const uint32_t t0 = ord0 + se_ra(ys0 + GY + 1);  // (past the shard order: the tail)
            const uint32_t xld = se_ra(t0 + 2 + 2 * se_ra(t0));
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(xld)) == 0) {
                invalidate_l1_cache();
            }
        }
    }
    // the weights' DRAM reads go in stream order a window ahead of the sends (WPF of my blocks in flight): spread
    // over gate / up at the rate the stream uses them - a burst of all 35 MB at the start starved the gather
    constexpr uint8_t WNOC = noc_index;  // (the W DRAM reads: the NCRISC's own NoC)
    constexpr uint32_t WPF = 3;
    constexpr uint32_t WLA = 6;            // (the window in rounds: K3 head traced 440 -> 397 us at 6, flat over 5-8)
    uint8_t trid_of[NB / GY + 1];          // (a core's blocks: every GY-th of the chunk's NB)
    uint32_t wi = 0, ni = 0, pending = 0;  // (prefetch cursor over the first chunk's rounds; issued; issued unsent)
#ifdef SE_WRING
    // (one chunk: my W-own a ring of SE_WRING block slots - a block read into slot (its issue index % SE_WRING) once
    // the block SE_WRING before it was sent (multicast flushed); the DRAM region keeps every block)
    uint8_t slot_of[NB / GY + 1];
    uint32_t nsent = 0;
#endif
#ifdef SE_WRING
    constexpr uint32_t WPN = NCH * NB;  // (rounds the prefetch walks: every chunk reads its blocks again)
#else
    constexpr uint32_t WPN = NB;  // (the first chunk's: resident after it)
#endif
    auto offset_of = [&](uint32_t b) { return static_cast<uint32_t>(woffs[b / GY]); };  // (b: one of mine)
    uint32_t wla = 0;                     // (rounds landed here - the window's base)
    auto prefetch = [&](uint32_t upto) {  // (issue my blocks of rounds < upto, and the window)
        // (the window in the column's rounds, not my blocks - a block only once its round is within WLA of
        // the stream; 3 of mine in flight on every core put ~30 rounds of the whole grid in the DRAM queues at once,
        // and the first-needed block landed ~50 us after the burst was issued)
        while (wi < WPN && (wi < upto || (pending < WPF && wi < wla + WLA))) {
            const uint32_t b = blk(wi);
            if (b % GY != cy) {
                ++wi;
                continue;
            }
#ifdef SE_WRING
            if (ni >= nsent + SE_WRING)
                break;  // (its slot still sending)
            const uint32_t lo = (ni % SE_WRING) * WSLOT;
            slot_of[b / GY] = ni % SE_WRING;
#endif
            ++wi;
            const uint32_t t = (ni++ % 14) + 1, o = offset_of(b);
#ifndef SE_WRING
            const uint32_t lo = o;
#endif
            trid_of[b / GY] = t;
            noc_async_read_set_trid(t, WNOC);
            noc_async_read(
                get_noc_addr_from_bank_id<true>(wbank, w_addr + wbase + o * WTB, WNOC),
                wown + lo * WTB,
                bsize(b) * WTB,
                WNOC);
            ++pending;
        }
    };
    const uint32_t NR = NCH * NB;                                // (W rounds a call)
    uint32_t ws = 0, woff = 0, wr = 0, wc = NWS;                 // (send, my offset, pushed, credited)
    uint32_t nf[NWS] = {};
    while (ws < NR && blk(ws) % GY != cy) {
        ++ws;  // (skip rounds I do not hold)
    }
    // (down + reduce-scatter) arrivals per direction in zone-slot order: slot i credited (its pages pushed - the arg's
    // high half: P, or with compact zones my tiles of its block) once my arrival counter reached its threshold (the
    // low half; 0: nothing comes - a block I hold no tile of); the fin sub-block to y
    const uint32_t t0r = ord0 + se_ra(ys0 + GY + 1);
    const uint32_t q0 = t0r + 4 + 2 * se_ra(t0r);  // (past the tail)
    const uint32_t naf = se_ra(q0), af0 = q0 + 1, nab = se_ra(af0 + naf), ab0 = af0 + naf + 1;
    const uint32_t f0 = ab0 + nab;
    const uint32_t fcount = se_ra(f0), fy0 = se_ra(f0 + 1);
    const uint32_t fstride = se_ra(f0 + 2), fw = se_ra(f0 + 3);
    volatile tt_l1_ptr uint32_t* arrf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(se_ra(f0 + 4));
    volatile tt_l1_ptr uint32_t* arrb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(se_ra(f0 + 5));
    uint32_t af = 0, ab = 0, fo = 0;
// (chunks: the counters count up over them - a chunk's thresholds + its index x the arrivals a chunk; args at
// f0 + 6: the fin's y0 of chunks 1 .., the ports facing my fwd / bwd upstream (NoC x | y << 16), their zone-free
// word, my FINB-free word, the ports reading my FINB)
#ifdef SE_FIN_ROWS  // (a core's MT rows straddle owner blocks: each (chunk, row)'s own y base, at f0 + 6 + NCH + 4)
    constexpr uint32_t CHX = NCH + 4 + NCH * MT, NFF = NCH - 1;
#else
    constexpr uint32_t CHX = NCH > 1 ? NCH + 4 : 0, NFF = NCH > 1 ? NCH - 1 : 0;
#endif
    uint32_t nfmax = 0, nbmax = 0, zuf = 0, zub = 0;  // (the thresholds' max; the zone pages my slots take)
    for (uint32_t i = 0; i < naf; ++i) {
        const uint32_t a = se_ra(af0 + i);
        nfmax = (a & 0xFFFF) > nfmax ? (a & 0xFFFF) : nfmax;
        zuf += a >> 16;
    }
    for (uint32_t i = 0; i < nab; ++i) {
        const uint32_t a = se_ra(ab0 + i);
        nbmax = (a & 0xFFFF) > nbmax ? (a & 0xFFFF) : nbmax;
        zub += a >> 16;
    }
    uint32_t zsf = 0, zsb = 0, ff = 0;  // (zone-free signals sent a direction; FINB-free credits pushed)
    const uint32_t zpf = NCH > 1 ? se_ra(f0 + 6 + NCH - 1) : 0, zpb = NCH > 1 ? se_ra(f0 + 6 + NCH) : 0;
    const uint32_t zfa = NCH > 1 ? se_ra(f0 + 6 + NCH + 1) : 0;
    volatile tt_l1_ptr uint32_t* ffw =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(NCH > 1 ? se_ra(f0 + 6 + NCH + 2) : 0);
    const uint32_t fnr = NCH > 1 ? se_ra(f0 + 6 + NCH + 3) : 0;
#ifdef SE_LAT
    // (the latent: once every other chip's slice of my rows has landed (lctr: lnsrc arrivals a chunk), the compute's
    // prep may untilize them)
    const uint32_t L0 = f0 + 6 + CHX, lnr = se_ra(L0), lnsrc = se_ra(L0 + 1);
    volatile tt_l1_ptr uint32_t* lctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(se_ra(L0 + 2));
    const uint32_t NLI = lnr == 0 || lnsrc == 0 ? 0 : NCH;
    uint32_t li = 0;
#else
    constexpr uint32_t NLI = 0, li = 0;
#endif
#ifdef SE_LAT
    const uint32_t P0 = f0 + 6 + 3 + CHX;
#else
    const uint32_t P0 = f0 + 6 + CHX;
#endif
    const uint32_t pnu = se_ra(P0);
    const auto pg = TensorAccessor(p_args, se_ra(P0 + 1), SE_PREP_ROW);
    uint32_t fsrc = 0, frow = 0,
             fco = 0;  // (the last prep unit: the fence's source / rows - mine: nothing else writes them)
    uint32_t pu = 0;
    while (ws < NR || wr < NR || wc < NR || af < NCH * naf || ab < NCH * nab || fo < NCH * fcount || ff < NFF ||
           li < NLI || pu < pnu) {
        invalidate_l1_cache();
#ifdef SE_LAT
        if (li < NLI && *lctr >= (li + 1) * lnsrc) {
            cb_reserve_back(SE_LRDY, 1);  // (the compute's prep may untilize the arrivals)
            cb_push_back(SE_LRDY, 1);
            ++li;
        }
#endif
        if (pu < pnu && cb_pages_available_at_front(SE_PO, SE_PREP_FULL)) {  // a prep unit's 32 rows to the buffer
            const uint32_t src = get_read_ptr(SE_PO), q = P0 + 2 + 3 * pu;
            const uint32_t row = se_ra(q), co = se_ra(q + 1), nb = se_ra(q + 2);
            // (posted: non-posted DRAM writes on NOC1 - their acks - under the W stream's NOC1 DRAM reads wedged a
            // chip's NOC1, ~1 in 200 calls with the prep in every chunk; landed by the fence at the end:
            // test_posted_fence)
            for (uint32_t i = 0; i < 32; ++i) {
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                    src + i * SE_PREP_FULL * 64, pg.get_noc_addr(row + i) + co, nb);
            }
            noc_async_posted_writes_flushed();  // (PO may be refilled once they left L1)
            fsrc = src, frow = row, fco = co;
            cb_pop_front(SE_PO, SE_PREP_FULL);
            ++pu;
        }
        if (af < NCH * naf) {
            const uint32_t sl = af % naf, mc = af / naf;
            if (NCH > 1 && sl == 0 && mc > zsf) {  // (chunk mc's first slot: chunk mc - 1's zone all read (popped) -
                // +1 on the zone-free word of the port facing my fwd upstream: it credits the upstream's sends)
                if (cb_pages_reservable_at_back(SE_ARRF, SE_ZF)) {
                    // (only a core something lands on signals: one with only "nothing comes" slots runs ahead of
                    // the credit - its signals would count for a slower one's)
                    if (nfmax) {
                        noc_semaphore_inc(get_noc_addr(zpf & 0xFFFF, zpf >> 16, zfa), 1);
                    }
                    ++zsf;
                }
            } else {
                const uint32_t a = se_ra(af0 + sl), thr = a & 0xFFFF, pz = a >> 16;
                if (thr == 0 || *arrf >= mc * nfmax + thr) {
                    if (pz) {
                        cb_reserve_back(SE_ARRF, pz);
                        cb_push_back(SE_ARRF, pz);
                    }
                    ++af;
                    if (NCH > 1 && sl + 1 == naf && SE_ZF > zuf) {  // (the chunk padded to the whole zone: the
                        // compute pops it all, the next chunk's slots from the zone's start again)
                        cb_reserve_back(SE_ARRF, SE_ZF - zuf);
                        cb_push_back(SE_ARRF, SE_ZF - zuf);
                    }
                }
            }
        }
        if (ab < NCH * nab) {
            const uint32_t sl = ab % nab, mc = ab / nab;
            if (NCH > 1 && sl == 0 && mc > zsb) {
                if (cb_pages_reservable_at_back(SE_ARRB, SE_ZB)) {
                    if (nbmax) {
                        noc_semaphore_inc(get_noc_addr(zpb & 0xFFFF, zpb >> 16, zfa), 1);
                    }
                    ++zsb;
                }
            } else {
                const uint32_t a = se_ra(ab0 + sl), thr = a & 0xFFFF, pz = a >> 16;
                if (thr == 0 || *arrb >= mc * nbmax + thr) {
                    if (pz) {
                        cb_reserve_back(SE_ARRB, pz);
                        cb_push_back(SE_ARRB, pz);
                    }
                    ++ab;
                    if (NCH > 1 && sl + 1 == nab && SE_ZB > zub) {
                        cb_reserve_back(SE_ARRB, SE_ZB - zub);
                        cb_push_back(SE_ARRB, SE_ZB - zub);
                    }
                }
            }
        }
        if (fo < NCH * fcount && cb_pages_available_at_front(SE_OUTF, 1)) {
            const uint32_t mc = fo / fcount, i = fo % fcount;
#ifdef SE_FIN_ROWS
            const uint32_t ya = se_ra(f0 + 6 + NCH + 4 + mc * MT + i / fw) + i % fw;
#elif defined(SE_RS_PM)
            // (the rows scatter's pass-major sub-block: pass q's nr x pw tiles after the earlier passes' - widths
            // _split(fw, SE_RS_PM))
            const uint32_t y0 = mc == 0 ? fy0 : se_ra(f0 + 6 + mc - 1);  // (chunk mc's rows)
            const uint32_t nr = fcount / fw, bw = fw / SE_RS_PM, rm = fw % SE_RS_PM;
            uint32_t q = 0, j = i, c0 = 0;
            while (j >= nr * (bw + (q < rm ? 1 : 0))) {
                j -= nr * (bw + (q < rm ? 1 : 0));
                c0 += bw + (q < rm ? 1 : 0);
                ++q;
            }
            const uint32_t pw = bw + (q < rm ? 1 : 0);
            const uint32_t ya = y0 + (j / pw) * fstride + c0 + j % pw;
#else
            const uint32_t y0 = mc == 0 ? fy0 : se_ra(f0 + 6 + mc - 1);  // (chunk mc's rows)
            const uint32_t ya = y0 + (i / fw) * fstride + i % fw;
#endif
            cb_wait_front(SE_OUTF, 1);
            noc_async_write(get_read_ptr(SE_OUTF), yg.get_noc_addr(ya), 2048);
            noc_async_writes_flushed();  // (the page read out: free for the next fin tiles - its ack is the kernel
                                         // end's barrier; a barrier a tile serialized ~35 DRAM round trips behind the
                                         // last fin)
            cb_pop_front(SE_OUTF, 1);
            ++fo;
        }
        if (ff < NFF && *ffw >= (ff + 1) * fnr) {  // (the ports read chunk ff's FINB: the compute may write it again)
            cb_reserve_back(SE_FFREE, 1);
            cb_push_back(SE_FFREE, 1);
            ++ff;
        }
        if (wi < WPN) {
            prefetch(0);  // (keep the window full)
        }
        if (ws < NR && (ws < NWS || *sem(WFREE + ws % NWS) >= GY * (nf[ws % NWS] + 1))) {  // my block: down the column
            const uint32_t b = blk(ws), n = bsize(b), kb = b / GY;  // (kb: its index among my blocks)
            woff = 0;                                               // (its offset in my W-own: my blocks before it)
#ifdef SE_WRING
            woff = slot_of[kb] * WSLOT;
#else
            woff = offset_of(b);
#endif
            if (ws < WPN) {  // (first chunk, or every chunk with the ring: its reads issued and landed)
                prefetch(ws + 1);
                noc_async_read_barrier_with_trid(trid_of[kb], WNOC);
                --pending;
            }
            const uint32_t dst = win + (ws % NWS) * WSLOT * WTB, vw = words + (ws & 3) * 16;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vw) = ws + 1;
            noc_async_write_multicast_loopback_src(
                wown + woff * WTB, get_noc_multicast_addr(cxv, by, cxv, ty, dst), n * WTB, GY);
            noc_async_writes_flushed();
            noc_semaphore_set_multicast_loopback_src(
                vw, get_noc_multicast_addr(cxv, by, cxv, ty, get_semaphore(WV + ws % NWS)), GY);
            noc_async_writes_flushed();
            if (ws >= NWS) {
                ++nf[ws % NWS];
            }
            ++ws;
#ifdef SE_WRING
            ++nsent;  // (flushed: its slot may take a later block)
#endif
            while (ws < NR && blk(ws) % GY != cy) {
                ++ws;
            }
        }
        if (wr < NR && (*sem(WV + wr % NWS) >= wr + 1)) {  // W round wr landed here
            cb_reserve_back(CB_WIN, WSLOT);
            cb_push_back(CB_WIN, WSLOT);
            wla = ++wr;
        }
        if (wc < NR && wc < wr + NWS &&
            cb_pages_reservable_at_back(CB_WIN, WSLOT * (wc - wr + 1))) {  // slot of wc free
            noc_semaphore_inc(get_noc_addr(cxv, se_ra(ys0 + blk(wc) % GY), get_semaphore(WFREE + wc % NWS)), 1);
            ++wc;
        }
    }
    if (pu) {  // (the prep's fence: a non-posted rewrite of 4 B in every DRAM bank - the last unit's rows, one a bank -
        // after the posted writes on the same VC: its ack means they all landed)
        for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
            noc_async_write(fsrc + i * SE_PREP_FULL * 64, pg.get_noc_addr(frow + i) + fco, 4);
        }
    }
    noc_async_write_barrier();
    {  // re-arm my arrival counters (every increment of the call landed: each was waited to its final count)
        if (nfmax) {
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrf)), 0u - NCH * nfmax);
        }
        if (nbmax) {
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrb)), 0u - NCH * nbmax);
        }
        if (NFF && fnr) {
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(ffw)), 0u - NFF * fnr);
        }
    }
#ifdef SE_LAT
    if (lnr && lnsrc) {
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(lctr)), 0u - NCH * lnsrc);  // (re-arm)
    }
#endif
    noc_async_atomic_barrier();
}
