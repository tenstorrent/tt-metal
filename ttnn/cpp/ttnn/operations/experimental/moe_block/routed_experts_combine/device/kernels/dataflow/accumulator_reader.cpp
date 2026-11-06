// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Accumulator reader: its local rows (the L1 hand-off ring) and its N - 1 rings (one per source chip, each up to this
// call's count in the device plan); a consumed slot is one credit, counted on the port that faces the row's source on
// this accumulator's link to it (that port sends the credits back). Each row goes to the compute with its gate weight.
// It also zeroes this core's state for the call (combine_zero_gate.hpp); accumulator 0's reader is its chip's gate.
// REC_SHARED (the shared landing pool): no rings - the pool's slots granted to sources by id (REC_G grants outstanding a
// source at most), each grant written into the home port's grant ring (REC_GIDH: per (source, accumulator) G words,
// slot | tag << 24) and counted (RET / MRTOT, as the credits are with rings): the port forwards them in its snapshots.
// Taken slots go back on the free list and to the sources with rows still to grant.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"
#include "combine_pool.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/combine_plan_words.hpp"

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
constexpr uint32_t RET = get_compile_time_arg_val(10), MISC = get_compile_time_arg_val(11),
                   RINGS = get_compile_time_arg_val(12), LAYO = get_compile_time_arg_val(13);
constexpr uint32_t cb_w = get_compile_time_arg_val(14);
constexpr uint32_t MTAKEN = get_compile_time_arg_val(15), MRTOT = get_compile_time_arg_val(16);
constexpr uint32_t ALAND = get_compile_time_arg_val(17);    // (128, or 160 when ports share the core)
constexpr uint32_t ALBASE = get_compile_time_arg_val(18);   // (101, or 103 when ports share the core)
constexpr uint32_t LOC_OFF = get_compile_time_arg_val(19);  // the local ring (arena-relative)
constexpr uint32_t WORDS_OFF = get_compile_time_arg_val(20);
constexpr uint32_t BAR_SLOT = get_compile_time_arg_val(21);
constexpr auto plan_args = TensorAccessorArgs<CMB_PLAN_CT>();  // (the device plan)
constexpr uint32_t LASTPB = row_bytes;  // the row's 32 B trailer (one fabric packet a row)
#ifdef REC_SHARED
constexpr uint32_t SHARED = 1, G = REC_G, GIDH = REC_GIDH;
static_assert(PSL <= 1024, "shared landing: slot ids 16-bit, the owners / free list in CB_TAB (acc_tab_bytes)");
#else
constexpr uint32_t SHARED = 0, G = 1, GIDH = 0;
#endif
#define GTAG(q) (0x80u | ((q) & 0x7Fu))  // (a grant's tag: never 0 - a zeroed ring entry is no grant)
// REC_POOL (CombineLanding::FlatCores): the rings' slots on the flat cells - ring slot i is pool slot g = me x PSL + i,
// of the flat cells' free arena (combine_pool.hpp; its args after the ports' x, y). A
// row landed once its trailer reads so (32 B reads, several slots a source at once); taking it clears the trailer there
// before the credit goes back (trailers zeroed each call).
#ifdef REC_POOL
constexpr uint32_t POOL = 1, PROWS = REC_POOL_ROWS;
#else
constexpr uint32_t POOL = 0, PROWS = 2;
#endif

void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);  // the arena (same address on every core and chip)
    const uint32_t plan = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t hc = get_arg_val<uint32_t>(a++);  // this chip's index (its own ring is not there)
    const uint32_t bar = get_arg_val<uint32_t>(a++);
    const uint32_t gate_x = get_arg_val<uint32_t>(a++), gate_y = get_arg_val<uint32_t>(a++);
    // this call's zeroing: the state block and every landing slot's trailer (a slot is taken when its trailer flags
    // are set)
    const uint32_t call = rec::next_call(bar, BAR_SLOT);
    rec::zero_l1(area, RINGS);
    const uint32_t pool_a = 7 + 2 * N;             // (REC_POOL: the pool's args, after the ports' x, y)
    auto pool_noc = [&](uint32_t i) -> uint64_t {  // my ring slot i's NoC address on its flat cell
        return rec_pool_noc(pool_a, area, SLOT, me * PSL + i);
    };
    if constexpr (POOL) {  // (no row in any of my slots: their trailers zeroed, from the zeroed state's first words)
        for (uint32_t i = 0; i < PSL; ++i) {
            noc_async_write(area, pool_noc(i) + LASTPB, 32);
        }
        noc_async_write_barrier();
    } else {
        rec::zero_l1_strided(area + RINGS + LASTPB, 32, SLOT, PSL);
    }
    rec::count_in(gate_x, gate_y, bar);
    uint32_t want[N], px[N], py[N];
    for (uint32_t k = 0; k < N; ++k) {
        px[k] = get_arg_val<uint32_t>(a++);
        py[k] = get_arg_val<uint32_t>(a++);
    }
    if constexpr (POOL) {
        a += get_arg_val<uint32_t>(a);
    }
    {
        const uint32_t is_gate = get_arg_val<uint32_t>(a++), nz = get_arg_val<uint32_t>(a++);
        const uint32_t n_cores = get_arg_val<uint32_t>(a++);
        if (is_gate) {
            rec::release(bar, call, nz, n_cores, a);
        }
        a += 2 * n_cores;
    }
    const uint32_t QL = get_arg_val<uint32_t>(a++), WL = get_arg_val<uint32_t>(a++);
    const uint32_t ldid = get_arg_val<uint32_t>(a++);
    const uint32_t LB = area + LOC_OFF, lwords = area + WORDS_OFF;
    const uint32_t wa = a;  // (the y writers' x, y)
    cb_reserve_back(cb_tab, 1);
    const uint32_t sc = (get_write_ptr(cb_tab) + 63) & ~63u;  // (CB_TAB: scratch - the plan's words, the free list)
    uint32_t n_local = 0;
    {  // (the device plan: n_local and want[] of this accumulator)
        namespace pw = moe_block_plan_words;  // (n_local[kAccMax] at sc, want[me] at sc + 128: below gq at sc + 256)
        const uint64_t plan_noc = TensorAccessor(plan_args, plan, pw::kPlanBytes).get_noc_addr(0);
        noc_async_read(plan_noc + pw::kPlanNloc * 4, sc, 4 * pw::kAccMax);  // n_local[kAccMax]
        noc_async_read(plan_noc + (pw::kPlanWant + me * pw::kMaxChips) * 4, sc + 4 * pw::kAccMax, 128);  // want[me]
        noc_async_read_barrier();
        n_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[me];
        for (uint32_t k = 0; k < N; ++k) {
            want[k] = k == hc ? 0 : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc + 4 * pw::kAccMax)[k];
        }
    }
    uint32_t n_rows = n_local;
    for (uint32_t k = 0; k < N; ++k) {
        n_rows += want[k];
    }
    rec::wait_released(bar, call);  // (this chip's ports and writers zeroed: layouts, credits and grants may go out)
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* taken = misc + MTAKEN;  // per ring (source chip), cumulative
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
                if (rdep[k] && (bk == N || rem[k] > rem[bk])) {
                    bk = k;
                }
            }
            if (bk == N) {
                break;
            }
            ++rdep[bk];
            rem[bk] = 0;
        }
        uint32_t b = 0;
        for (uint32_t k = 0; k < N; ++k) {
            uint32_t d = rdep[k] < want[k] ? rdep[k] : want[k];  // (no more slots than rows)
            if constexpr (SHARED) {
                d = d < G ? d : G;  // (the first grants: the carve's shares, G at most)
            }
            rbase[k] = b;
            rdep[k] = d;
            b += d;
            t0[k] = taken[k];
            if constexpr (!SHARED) {
                if (k != hc && px[k]) {
                    noc_inline_dw_write(
                        get_noc_addr(px[k], py[k], area + LAYO + 4 * (k * AP + me)), tag << 24 | rbase[k] << 12 | d);
                }
            }
        }
    }
    // shared: each slot's owner (source, 0xFF free) and the free list in CB_TAB's scratch past the plan's words; per
    // source the grants outstanding and issued this call
    // (CB_TAB from sc: the plan's words, the grant queues at 256 (2 N G B), the trailer probes at 512 (<= 1 KB), 32 B
    // of zeros at 1536, the owners at 1600 (PSL B), the free list after them (2 PSL B) - acc_tab_bytes)
    volatile tt_l1_ptr uint8_t* own = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(sc + 1600);
    volatile tt_l1_ptr uint16_t* fstack =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc + 1600 + ((PSL + 3u) & ~3u));
    // a source's granted slots in grant order - it fills them in that order (one port, one path), so its oldest
    // outstanding grant, at (issued - outstanding) % G, is the only one to check
    volatile tt_l1_ptr uint16_t* gq = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc + 256);
    static_assert(2 * N * G <= 256, "shared landing: the grant queues in CB_TAB");
    // (REC_POOL: each source's next slots' trailers, 32 B a slot - past the shared landing's grant queues; then 32 B of
    // zeros, a taken slot's trailer)
    const uint32_t probe = sc + 512;
    const uint32_t zero32 = sc + 1536;
    if constexpr (POOL) {
        for (uint32_t w = 0; w < 8; ++w) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero32)[w] = 0;
        }
    }
    uint32_t gout[N], giss[N], nfree = 0;
    auto grant = [&](uint32_t k, uint32_t slot) {  // (the id into the home port's grant ring, then its count)
        const uint32_t seq = t0[k] + giss[k];
        noc_inline_dw_write(
            get_noc_addr(px[k], py[k], area + GIDH + 4 * ((k * AP + me) * G + seq % G)), slot | GTAG(seq) << 24);
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + RET + 4 * (k * AP + me)), 1);
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + MISC + 4 * (MRTOT + k)), 1);
        own[slot] = static_cast<uint8_t>(k);
        gq[k * G + giss[k] % G] = static_cast<uint16_t>(slot);
        ++giss[k];
        ++gout[k];
    };
    // a freed slot to the source with rows still to grant (want - outstanding), fewest outstanding first
    auto regrant = [&]() {
        while (nfree) {
            uint32_t bk = N;
            for (uint32_t k = 0; k < N; ++k) {
                if (k == hc || gout[k] >= G || want[k] <= gout[k]) {
                    continue;
                }
                if (bk == N || gout[k] < gout[bk] ||
                    (gout[k] == gout[bk] && want[k] - gout[k] > want[bk] - gout[bk])) {
                    bk = k;
                }
            }
            if (bk == N) {
                break;
            }
            grant(bk, fstack[--nfree]);
        }
    };
    if constexpr (SHARED) {
        for (uint32_t i = 0; i < PSL; ++i) {
            own[i] = 0xFF;
        }
        for (uint32_t k = 0; k < N; ++k) {
            gout[k] = giss[k] = 0;
        }
        for (uint32_t k = 0; k < N; ++k) {
            if (k == hc || !px[k]) {
                continue;
            }
            for (uint32_t j = 0; j < rdep[k]; ++j) {
                grant(k, rbase[k] + j);
            }
        }
        for (uint32_t i = PSL; i-- > 0;) {
            if (own[i] == 0xFF) {
                fstack[nfree++] = static_cast<uint16_t>(i);
            }
        }
        regrant();  // (the slots the first grants left: to whoever has rows still to grant)
    }
    auto ring_slot = [&](uint32_t k, uint32_t tk) { return area + RINGS + (rbase[k] + (tk - t0[k]) % rdep[k]) * SLOT; };
    auto push_row = [&](uint32_t slot, uint32_t t) {  // a row: its gate weight (element 0), the row into the CB
        cb_reserve_back(cb_w, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + LASTPB)[1] & 0xFFFF;
        cb_push_back(cb_w, 1);
        cb_reserve_back(cb_row, NT);
        cb_reserve_back(cb_meta, 1);
        noc_async_read(get_noc_addr(slot), get_write_ptr(cb_row), row_bytes);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = t / A;
        noc_async_read_barrier();
        cb_push_back(cb_row, NT);
        cb_push_back(cb_meta, 1);
    };
    uint32_t done = 0, li = 0;
    // L1 hand-off: local rows land in a ring of QL slots at LB (the y writers, y_row_major.hpp SE_Y_L1), granted from
    // here
    volatile tt_l1_ptr uint32_t* landed = misc + ALAND;
    const uint32_t lbase = misc[ALBASE];
    uint32_t lgranted = 0;
    auto lgrant_to = [&](uint32_t g) {
        for (uint32_t w = 0; w < WL; ++w) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(wa + 2 * w), get_arg_val<uint32_t>(wa + 1 + 2 * w), lwords + 4 * ldid),
                g - lgranted);
        }
        lgranted = g;
    };
    if (n_local) {
        lgrant_to(n_local < QL ? n_local : QL);
    }
    auto take_local = [&]() -> bool {  // the next local row (L1 hand-off ring), if every writer landed it
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < WL; ++w) {
            const uint32_t v = landed[w] - lbase;
            m = v < m ? v : m;
        }
        if (li >= m) {
            return false;
        }
        const uint32_t sl = LB + (li % QL) * SLOT;
        push_row(sl, (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl + LASTPB)[0] >> 8) & 0xFFFF);
        ++li;
        ++done;
        const uint32_t g = li + QL < n_local ? li + QL : n_local;
        // (grants in batches of up to 4 - at most the ring: a batch larger than QL never fills, and the writers stall)
        if (g > lgranted && (g - lgranted >= (QL < 4 ? QL : 4) || g == n_local)) {
            lgrant_to(g);
        }
        return true;
    };
    auto take_ring = [&](uint32_t k) -> bool {  // the next row of source k's ring, if it landed
        const uint32_t tk = taken[k];
        const uint32_t s = ring_slot(k, tk);
        volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB);
        if (tr[3] == 0 || tr[7] == 0) {
            return false;
        }
        push_row(s, (tr[0] >> 8) & 0xFFFF);
        tr[3] = 0;
        tr[7] = 0;
        taken[k] = tk + 1;
        --want[k];
        ++done;
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + RET + 4 * (k * AP + me)), 1);
        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + MISC + 4 * (MRTOT + k)), 1);
        return true;
    };
    // REC_POOL: the landed rows of the probed slots, PROWS row reads in flight (the row CB's depth); taking a row clears
    // its trailer there (the caller fences those writes before the credits go back)
    constexpr uint32_t PP = N > 16 ? 1 : (N > 8 ? 2 : 4);  // (slots probed a source: N x PP x 32 B of CB_TAB)
    uint32_t inflight = 0;
    auto flush_rows = [&]() {
        noc_async_read_barrier();
        for (uint32_t i = 0; i < inflight; ++i) {  // (a row at a time: a push wraps the CB only at its exact end)
            cb_push_back(cb_row, NT);
            cb_push_back(cb_meta, 1);
        }
        inflight = 0;
    };
    auto take_pool = [&](uint32_t k, volatile tt_l1_ptr uint32_t* tr) {
        const uint32_t tk = taken[k];
        const uint64_t sn = pool_noc(rbase[k] + (tk - t0[k]) % rdep[k]);
        cb_reserve_back(cb_w, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] = tr[1] & 0xFFFF;
        cb_push_back(cb_w, 1);
        cb_reserve_back(cb_row, NT * (inflight + 1));
        cb_reserve_back(cb_meta, inflight + 1);
        uint32_t rp = get_write_ptr(cb_row) + inflight * NT * get_local_cb_interface(cb_row).fifo_page_size;
        if (rp >= get_local_cb_interface(cb_row).fifo_limit) {
            rp -= get_local_cb_interface(cb_row).fifo_size;
        }
        uint32_t mp = get_write_ptr(cb_meta) + inflight * get_local_cb_interface(cb_meta).fifo_page_size;
        if (mp >= get_local_cb_interface(cb_meta).fifo_limit) {
            mp -= get_local_cb_interface(cb_meta).fifo_size;
        }
        noc_async_read(sn, rp, row_bytes);
        noc_async_write(zero32, sn + LASTPB, 32);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp)[0] = ((tr[0] >> 8) & 0xFFFF) / A;
        taken[k] = tk + 1;
        --want[k];
        ++done;
        if (++inflight == PROWS) {
            flush_rows();
        }
    };
    // shared, REC_POOL: source k's oldest granted slot i on its flat cell (its trailer probed into tr): the row read,
    // its trailer cleared there, the slot free (granted again only once the caller has fenced the clears)
    auto take_pool_slot = [&](uint32_t i, uint32_t k, volatile tt_l1_ptr uint32_t* tr) {
        const uint64_t sn = pool_noc(i);
        cb_reserve_back(cb_w, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] = tr[1] & 0xFFFF;
        cb_push_back(cb_w, 1);
        cb_reserve_back(cb_row, NT * (inflight + 1));
        cb_reserve_back(cb_meta, inflight + 1);
        uint32_t rp = get_write_ptr(cb_row) + inflight * NT * get_local_cb_interface(cb_row).fifo_page_size;
        if (rp >= get_local_cb_interface(cb_row).fifo_limit) {
            rp -= get_local_cb_interface(cb_row).fifo_size;
        }
        uint32_t mp = get_write_ptr(cb_meta) + inflight * get_local_cb_interface(cb_meta).fifo_page_size;
        if (mp >= get_local_cb_interface(cb_meta).fifo_limit) {
            mp -= get_local_cb_interface(cb_meta).fifo_size;
        }
        noc_async_read(sn, rp, row_bytes);
        noc_async_write(zero32, sn + LASTPB, 32);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp)[0] = ((tr[0] >> 8) & 0xFFFF) / A;
        own[i] = 0xFF;
        fstack[nfree++] = static_cast<uint16_t>(i);
        taken[k] = taken[k] + 1;
        --want[k];
        --gout[k];
        ++done;
        if (++inflight == PROWS) {
            flush_rows();
        }
    };
    auto take_slot = [&](uint32_t i) -> bool {  // shared: slot i's row, if it landed
        const uint32_t s = area + RINGS + i * SLOT;
        volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB);
        if (tr[3] == 0 || tr[7] == 0) {
            return false;
        }
        push_row(s, (tr[0] >> 8) & 0xFFFF);
        tr[3] = 0;
        tr[7] = 0;
        const uint32_t k = own[i];
        own[i] = 0xFF;
        fstack[nfree++] = static_cast<uint16_t>(i);
        taken[k] = taken[k] + 1;
        --want[k];
        --gout[k];
        ++done;
        return true;
    };
    while (done < n_rows) {
        invalidate_l1_cache();
        while (li < n_local) {
            if (!take_local()) {
                break;
            }
        }
        if constexpr (SHARED && POOL) {  // (each source's oldest PP grants probed at once, the landed rows taken)
            uint32_t np[N];
            for (uint32_t k = 0; k < N; ++k) {
                const uint32_t n = gout[k] < PP ? gout[k] : PP;
                np[k] = n;
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read(
                        pool_noc(gq[k * G + (giss[k] - gout[k] + j) % G]) + LASTPB, probe + 32 * (k * PP + j), 32);
                }
            }
            noc_async_read_barrier();
            bool any = false;
            for (uint32_t k = 0; k < N; ++k) {
                for (uint32_t j = 0; j < np[k]; ++j) {
                    volatile tt_l1_ptr uint32_t* tr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe + 32 * (k * PP + j));
                    if (tr[3] == 0 || tr[7] == 0) {
                        break;
                    }
                    take_pool_slot(gq[k * G + (giss[k] - gout[k]) % G], k, tr);
                    any = true;
                }
            }
            if (any) {  // (the slots granted again once their rows are read and their trailers cleared)
                if (inflight) {
                    flush_rows();
                }
                noc_async_write_barrier();
                regrant();
            }
        } else if constexpr (SHARED) {  // (each source's oldest grant only; freed slots granted again at once)
            bool any = false;
            for (uint32_t k = 0; k < N; ++k) {
                while (gout[k] && take_slot(gq[k * G + (giss[k] - gout[k]) % G])) {
                    any = true;
                }
            }
            if (any) {
                regrant();
            }
        } else if constexpr (POOL) {  // (each source's next PP slots probed at once, then the landed rows taken)
            uint32_t np[N];
            for (uint32_t k = 0; k < N; ++k) {
                uint32_t n = want[k] < PP ? want[k] : PP;
                n = n < rdep[k] ? n : rdep[k];
                np[k] = n;
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read(
                        pool_noc(rbase[k] + (taken[k] + j - t0[k]) % rdep[k]) + LASTPB, probe + 32 * (k * PP + j), 32);
                }
            }
            noc_async_read_barrier();
            uint32_t got[N], any = 0;
            for (uint32_t k = 0; k < N; ++k) {
                got[k] = 0;
                for (uint32_t j = 0; j < np[k]; ++j) {
                    volatile tt_l1_ptr uint32_t* tr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe + 32 * (k * PP + j));
                    if (tr[3] == 0 || tr[7] == 0) {
                        break;
                    }
                    take_pool(k, tr);
                    ++got[k];
                }
                any |= got[k];
            }
            if (any) {  // (credits once the rows are read and their trailers cleared: the sources write the slots again)
                if (inflight) {
                    flush_rows();
                }
                noc_async_write_barrier();
                for (uint32_t k = 0; k < N; ++k) {
                    if (got[k]) {
                        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + RET + 4 * (k * AP + me)), got[k]);
                        noc_semaphore_inc(get_noc_addr(px[k], py[k], area + MISC + 4 * (MRTOT + k)), got[k]);
                    }
                }
            }
        } else {
            for (uint32_t k = 0; k < N; ++k) {
                if (want[k]) {
                    take_ring(k);
                }
            }
        }
    }
    cb_reserve_back(cb_meta, 1);  // (the compute's end marker: it does not know the row count)
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = 0xFFFFFFFF;
    cb_push_back(cb_meta, 1);
    misc[ALBASE] = lbase + n_local;
    noc_async_atomic_barrier();
}
