// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, reader (RISCV_0) of one core of one expert group.
// 1. Routing, built once per call over the NP cores of the program: core i resolves the tokens
//    [i T / NP, (i + 1) T / NP) with moe_compute's mapping rules (local experts = ids the device's own mapping row maps
//    to itself, ascending; a (token, k) is local when the token's source row maps the id to this device; a token that
//    names one expert at several k gives one entry per k, as moe_compute lists them) and sends its local entries
//    (slot, token, score, k) to the root core (core 0). The root counting-sorts them by slot in core order (= token
//    order) into the routing table: per slot the routed rows in token order, at most `row_cap` (the slot's e_t page and
//    double-buffer half; later entries are dropped), and jobs of at most JR rows of one slot. Every other core reads
//    the table back.
//    The root also writes the table block to DRAM for the metadata program.
// 2. Streams this core's share of each job's expert weights in compute order P1(0), P1(1), P2(0), P1(2), P2(1), ...:
//    P1 = this core's W0/W1 column groups, P2 = its W2 output groups, both in moe_compute's prepared layout (one run
//    per group, the K padding rows at each group's end are not read).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t cb_rt = get_named_compile_time_arg_val("cb_rt");
    constexpr uint32_t cb_ctl = get_named_compile_time_arg_val("cb_ctl");
    constexpr uint32_t BT = get_named_compile_time_arg_val("block_tiles");
    constexpr uint32_t BP = get_named_compile_time_arg_val("block_packets");
    constexpr uint32_t TB = get_named_compile_time_arg_val("tile_bytes");
    constexpr uint32_t RS = get_named_compile_time_arg_val("block_slots");
    constexpr uint32_t IF = get_named_compile_time_arg_val("blocks_in_flight");
    constexpr uint32_t T = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t E = get_named_compile_time_arg_val("local_experts");
    constexpr uint32_t K = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t NID = get_named_compile_time_arg_val("global_experts");
    constexpr uint32_t IDX_PAGE = get_named_compile_time_arg_val("index_page_bytes");
    constexpr uint32_t MAP_PAGE = get_named_compile_time_arg_val("mapping_page_bytes");
    constexpr uint32_t TPS = get_named_compile_time_arg_val("tokens_per_source");
    constexpr uint32_t SPC = get_named_compile_time_arg_val("sources_per_core");
    constexpr uint32_t G = get_named_compile_time_arg_val("expert_groups");
    constexpr uint32_t JR = get_named_compile_time_arg_val("job_rows");
    constexpr uint32_t CAP = get_named_compile_time_arg_val("row_cap");
    constexpr uint32_t GW = get_named_compile_time_arg_val("grid_w");
    constexpr uint32_t GH = get_named_compile_time_arg_val("grid_h");
    constexpr uint32_t MW = get_named_compile_time_arg_val("mask_words");
    constexpr uint32_t SEM_A = get_named_compile_time_arg_val("sem_entries");
    constexpr uint32_t SEM_B = get_named_compile_time_arg_val("sem_table");
    // cb_rt byte offsets (MoEExpertRowsRoutingLayout)
    constexpr uint32_t OWN = get_named_compile_time_arg_val("rt_own");
    constexpr uint32_t MPO = get_named_compile_time_arg_val("rt_maps");
    constexpr uint32_t CIO = get_named_compile_time_arg_val("rt_ids");
    constexpr uint32_t CSO = get_named_compile_time_arg_val("rt_scores");
    constexpr uint32_t CTO = get_named_compile_time_arg_val("rt_ctl");
    constexpr uint32_t LSO = get_named_compile_time_arg_val("rt_slots");
    constexpr uint32_t CNO = get_named_compile_time_arg_val("rt_counts");
    constexpr uint32_t OFO = get_named_compile_time_arg_val("rt_offsets");
    constexpr uint32_t RTH = get_named_compile_time_arg_val("rt_table");
    constexpr uint32_t RTC = get_named_compile_time_arg_val("rt_table_counts");
    constexpr uint32_t RTO = get_named_compile_time_arg_val("rt_table_offsets");
    constexpr uint32_t RWO = get_named_compile_time_arg_val("rt_rows");
    constexpr uint32_t TES = get_named_compile_time_arg_val("rt_entry_slots");
    constexpr uint32_t TET = get_named_compile_time_arg_val("rt_entry_tokens");
    constexpr uint32_t TEC = get_named_compile_time_arg_val("rt_entry_scores");
    constexpr uint32_t TEK = get_named_compile_time_arg_val("rt_entry_k");
    constexpr uint32_t RTBS = get_named_compile_time_arg_val("rt_table_bytes");
    constexpr uint32_t JBO = get_named_compile_time_arg_val("rt_jobs");
    constexpr uint32_t ENO = get_named_compile_time_arg_val("rt_areas");
    // weights in moe_compute's prepared layout (tiles)
    constexpr uint32_t P1_RUN = get_named_compile_time_arg_val("w0_w1_run_tiles");
    constexpr uint32_t P1_STRIDE = get_named_compile_time_arg_val("w0_w1_group_tiles");
    constexpr uint32_t P1_GROUPS = get_named_compile_time_arg_val("w0_w1_groups_per_core");
    constexpr uint32_t P2_RUN = get_named_compile_time_arg_val("w2_run_tiles");
    constexpr uint32_t P2_STRIDE = get_named_compile_time_arg_val("w2_group_tiles");
    constexpr uint32_t P2_GROUPS = get_named_compile_time_arg_val("w2_groups_per_core");

    constexpr auto idx_args = TensorAccessorArgs<0>();
    constexpr auto sc_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto map_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto tab_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();

    constexpr uint32_t BB = BT * TB;
    constexpr uint32_t PT = (BT + BP - 1) / BP;  // tiles per NoC packet of a block
    static_assert(PT * TB <= NOC_MAX_BURST_SIZE, "a block's packets fit a NoC burst each");
    static_assert(K >= 1 && K <= 32, "top-k");
    constexpr uint32_t ME = K;              // local entries per token at most (one per k)
    constexpr uint32_t KST = IDX_PAGE / 2;  // u16 per index page

    uint32_t a = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(a++);
    const uint32_t idx_addr = get_arg_val<uint32_t>(a++);
    const uint32_t sc_addr = get_arg_val<uint32_t>(a++);
    const uint32_t map_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w01_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t g0 = get_arg_val<uint32_t>(a++), ng = get_arg_val<uint32_t>(a++);
    const uint32_t q0 = get_arg_val<uint32_t>(a++), nq = get_arg_val<uint32_t>(a++);
    const uint32_t g = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++), NP = get_arg_val<uint32_t>(a++);
    const uint32_t rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t device_id = get_arg_val<uint32_t>(a++);
    // common args: mapping row of each source (count first), virtual x / y of the grid, mask of the program's cores
    const uint32_t src_at = 0;
    const uint32_t vx_at = get_common_arg_val<uint32_t>(0) + 1, vy_at = vx_at + GW, all_at = vy_at + GH;
    const bool root = me == 0;

    // ---- routing ----
    const uint32_t t0 = me * T / NP, t1 = (me + 1) * T / NP;
    const uint32_t cape = ((T + NP - 1) / NP) * ME;
    const uint32_t estr = (4 + 8 * cape + 15) & ~15u;  // area: u32 n, u16 slot[cape], tok[cape], score[cape], k[cape]
    cb_reserve_back(cb_rt, 1);
    const uint32_t rt = get_write_ptr(cb_rt);
    const auto mp = TensorAccessor(map_args, map_addr, MAP_PAGE);
    noc_async_read(mp.get_noc_addr(device_id), rt + OWN, NID * 2);
    const uint32_t s_lo = t1 > t0 ? t0 / TPS : 0;
    const uint32_t s_n = t1 > t0 ? (t1 - 1) / TPS - s_lo + 1 : 0;
    for (uint32_t s = 0; s < s_n; ++s) {
        noc_async_read(
            mp.get_noc_addr(get_common_arg_val<uint32_t>(src_at + 1 + s_lo + s)), rt + MPO + s * NID * 2, NID * 2);
    }
    {
        const auto idx = TensorAccessor(idx_args, idx_addr, IDX_PAGE);
        const auto sc = TensorAccessor(sc_args, sc_addr, IDX_PAGE);
        for (uint32_t t = t0; t < t1; ++t) {
            noc_async_read(idx.get_noc_addr(t), rt + CIO + (t - t0) * IDX_PAGE, IDX_PAGE);
            noc_async_read(sc.get_noc_addr(t), rt + CSO + (t - t0) * IDX_PAGE, IDX_PAGE);
        }
    }
    noc_async_read_barrier();
    // this RISC alone works on the scratch below after the reads: plain pointers
    tt_l1_ptr uint16_t* rank = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + OWN);
    {
        uint32_t r = 0;
        for (uint32_t e = 0; e < NID; ++e) {
            rank[e] = rank[e] == device_id ? r++ : 0xFFFFu;
        }
    }
    static_assert(SPC >= 1, "sources per core");
    for (uint32_t s = 0; s < s_n; ++s) {
        tt_l1_ptr uint16_t* m = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + MPO + s * NID * 2);
        for (uint32_t e = 0; e < NID; ++e) {
            m[e] = m[e] == device_id ? rank[e] : 0xFFFFu;
        }
    }
    const tt_l1_ptr uint16_t* ci = reinterpret_cast<const tt_l1_ptr uint16_t*>(rt + CIO);
    const tt_l1_ptr uint16_t* cs = reinterpret_cast<const tt_l1_ptr uint16_t*>(rt + CSO);
    auto area = [&](uint32_t i) -> uint32_t { return rt + ENO + i * estr; };
    {
        const uint32_t ar = area(me);
        tt_l1_ptr uint16_t* es = reinterpret_cast<tt_l1_ptr uint16_t*>(ar + 4);
        tt_l1_ptr uint16_t* et = es + cape;
        tt_l1_ptr uint16_t* ec = et + cape;
        tt_l1_ptr uint16_t* ek = ec + cape;
        uint32_t n = 0;
        for (uint32_t t = t0; t < t1; ++t) {
            const tt_l1_ptr uint16_t* m =
                reinterpret_cast<const tt_l1_ptr uint16_t*>(rt + MPO + (t / TPS - s_lo) * NID * 2);
            const tt_l1_ptr uint16_t* c = ci + (t - t0) * KST;
            uint16_t v[K];
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t gid = c[k];
                v[k] = gid < NID ? m[gid] : 0xFFFFu;
            }
            for (uint32_t k = 0; k < K; ++k) {
                if (v[k] < E) {
                    es[n] = v[k];
                    et[n] = t;
                    ec[n] = cs[(t - t0) * KST + k];
                    ek[n] = k;
                    ++n;
                }
            }
        }
        *reinterpret_cast<tt_l1_ptr uint32_t*>(ar) = n;
        asm volatile("" ::: "memory");
        if (!root) {
            noc_async_write(ar, get_noc_addr(rx, ry, ar), estr);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(rx, ry, get_semaphore(SEM_A)), 1);
            noc_async_atomic_barrier();
        }
    }
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt + CTO);
    tt_l1_ptr uint16_t* rows = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + RWO);
    tt_l1_ptr uint16_t* jobs = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + JBO);
    uint32_t nj = 0;
    if (root) {
        volatile tt_l1_ptr uint32_t* sa = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_A));
        while (*sa < NP - 1) {
            invalidate_l1_cache();
        }
        invalidate_l1_cache();
        tt_l1_ptr uint16_t* list = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + LSO);
        tt_l1_ptr uint16_t* ofs = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + OFO);
        tt_l1_ptr uint16_t* cnt = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + CNO);
        for (uint32_t l = 0; l < E; ++l) {
            cnt[l] = 0;
        }
        for (uint32_t i = 0; i < NP; ++i) {
            const uint32_t ar = area(i);
            const uint32_t n = *reinterpret_cast<const tt_l1_ptr uint32_t*>(ar);
            const tt_l1_ptr uint16_t* es = reinterpret_cast<const tt_l1_ptr uint16_t*>(ar + 4);
            for (uint32_t e = 0; e < n; ++e) {
                if (cnt[es[e]] < CAP) {
                    cnt[es[e]] = cnt[es[e]] + 1;
                }
            }
        }
        uint32_t d = 0, nrow = 0;
        for (uint32_t l = 0; l < E; ++l) {
            if (cnt[l]) {
                list[d++] = l;
                ofs[l] = nrow;
                nrow += cnt[l];
                cnt[l] = 0;  // fill cursor
            }
        }
        // areas in core order = tokens ascending within a slot; the same entries in token order go to the table
        tt_l1_ptr uint16_t* tes = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + TES);
        tt_l1_ptr uint16_t* tet = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + TET);
        tt_l1_ptr uint16_t* tec = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + TEC);
        tt_l1_ptr uint16_t* tek = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + TEK);
        uint32_t q = 0;
        for (uint32_t i = 0; i < NP; ++i) {
            const uint32_t ar = area(i);
            const uint32_t n = *reinterpret_cast<const tt_l1_ptr uint32_t*>(ar);
            const tt_l1_ptr uint16_t* es = reinterpret_cast<const tt_l1_ptr uint16_t*>(ar + 4);
            const tt_l1_ptr uint16_t* et = es + cape;
            const tt_l1_ptr uint16_t* ec = et + cape;
            const tt_l1_ptr uint16_t* ek = ec + cape;
            for (uint32_t e = 0; e < n; ++e) {
                const uint32_t l = es[e];
                if (cnt[l] == CAP) {
                    continue;  // past the slot's cap, as in the count above
                }
                rows[ofs[l] + cnt[l]] = et[e];
                cnt[l] = cnt[l] + 1;
                tes[q] = l;
                tet[q] = et[e];
                tec[q] = ec[e];
                tek[q] = ek[e];
                ++q;
            }
        }
        for (uint32_t i = 0; i < d; ++i) {
            const uint32_t l = list[i];
            for (uint32_t r0 = 0; r0 < cnt[l]; r0 += JR) {
                jobs[3 * nj] = l;
                jobs[3 * nj + 1] = ofs[l] + r0;
                jobs[3 * nj + 2] = cnt[l] - r0 < JR ? cnt[l] - r0 : JR;
                ++nj;
            }
        }
        ctl[0] = nj;
        ctl[1] = d;
        ctl[2] = nrow;
        *reinterpret_cast<tt_l1_ptr uint32_t*>(rt + RTH) = nrow;
        tt_l1_ptr uint16_t* tc = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + RTC);
        tt_l1_ptr uint16_t* to = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + RTO);
        uint32_t run = 0;
        for (uint32_t l = 0; l < E; ++l) {
            tc[l] = cnt[l];
            to[l] = run;
            run += cnt[l];
        }
        asm volatile("" ::: "memory");
        const uint32_t sb = get_semaphore(SEM_B);
        for (uint32_t w = 0; w < MW; ++w) {
            uint32_t bits = get_common_arg_val<uint32_t>(all_at + w);
            while (bits) {
                const uint32_t b = __builtin_ctz(bits);
                bits &= bits - 1;
                const uint32_t li = w * 32 + b;
                const uint32_t cx = get_common_arg_val<uint32_t>(vx_at + li / GH);
                const uint32_t cy = get_common_arg_val<uint32_t>(vy_at + li % GH);
                if (cx != rx || cy != ry) {
                    noc_semaphore_inc(get_noc_addr(cx, cy, sb), 1);
                }
            }
        }
        noc_async_atomic_barrier();
        const auto tab = TensorAccessor(tab_args, tab_addr, RTBS);
        noc_async_write(rt + RTH, tab.get_noc_addr(0), RTBS);  // drained before this kernel ends
    } else {
        volatile tt_l1_ptr uint32_t* sb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_B));
        while (*sb < 1) {
            invalidate_l1_cache();
        }
        noc_async_read(get_noc_addr(rx, ry, rt + CTO), rt + CTO, 16);
        noc_async_read_barrier();
        nj = ctl[0];
        const uint32_t nrow = ctl[2];
        if (nj) {
            noc_async_read(get_noc_addr(rx, ry, rt + JBO), rt + JBO, (3 * nj * 2 + 15) & ~15u);
            noc_async_read(get_noc_addr(rx, ry, rt + RWO), rt + RWO, (nrow * 2 + 15) & ~15u);
            noc_async_read_barrier();
        }
    }
    asm volatile("" ::: "memory");  // every table store above is issued before the CB is pushed
    cb_push_back(cb_rt, 1);
    cb_reserve_back(cb_ctl, 1);
    volatile tt_l1_ptr uint32_t* cc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));
    const uint32_t D = nj > g ? (nj - g + G - 1) / G : 0;  // this group's jobs: g, g + G, ...
    cc[0] = D;
    cb_push_back(cb_ctl, 1);
    if (D == 0) {
        noc_async_write_barrier();  // the root's table block
        return;
    }

    // ---- weight stream: passes P1(0), P1(1), P2(0), P1(2), P2(1), ..., P2(D - 1); one run per group ----
    const uint32_t npass = 2 * D;
    auto pass_of = [&](uint32_t s, uint32_t& e, uint32_t& ph) {
        if (s == 0) {
            e = 0;
            ph = 0;
        } else if (s == npass - 1) {
            e = D - 1;
            ph = 1;
        } else {
            const uint32_t ep = (s + 1) >> 1;
            if (s & 1) {
                e = ep;
                ph = 0;
            } else {
                e = ep - 1;
                ph = 1;
            }
        }
    };
    const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank_id, 0);
    const uint32_t bank_lo = (uint32_t)bank_noc;
    const uint32_t slots = get_write_ptr(cb_w);
    noc_async_read_one_packet_set_state<true>(bank_noc, PT * TB, 0);
    // cursor over (pass, run, block)
    uint32_t pass = 0, run = 0, runs = 0, run_base = 0, run_stride = 0, run_addr = 0, run_left = 0, run_tiles = 0;
    auto open_pass = [&]() {
        uint32_t e, ph;
        pass_of(pass, e, ph);
        const uint32_t slot = jobs[3 * (g + G * e)];
        if (ph == 0) {
            runs = ng;
            run_tiles = P1_RUN;
            run_stride = P1_STRIDE * TB;
            run_base = w01_addr + ((slot * P1_GROUPS) + g0) * P1_STRIDE * TB;
        } else {
            runs = nq;
            run_tiles = P2_RUN;
            run_stride = P2_STRIDE * TB;
            run_base = w2_addr + ((slot * P2_GROUPS) + q0) * P2_STRIDE * TB;
        }
        run = 0;
        run_addr = run_base;
        run_left = runs ? run_tiles : 0;
    };
    auto advance = [&]() {
        while (run_left == 0) {
            if (++run < runs) {
                run_addr = run_base + run * run_stride;
                run_left = run_tiles;
            } else if (++pass < npass) {
                open_pass();
            } else {
                return;
            }
        }
    };
    open_pass();
    advance();
    uint32_t total = 0;
    for (uint32_t s = 0; s < npass; ++s) {
        uint32_t e, ph;
        pass_of(s, e, ph);
        total += ph == 0 ? ng * ((P1_RUN + BT - 1) / BT) : nq * ((P2_RUN + BT - 1) / BT);
    }
    uint32_t issued = 0, pushed = 0;
    while (pushed < total) {
        if (issued < total && issued - pushed < IF && cb_pages_reservable_at_back(cb_w, BT * (issued - pushed + 1))) {
            const uint32_t slot = issued % RS;
            const uint32_t nt = run_left < BT ? run_left : BT;
            noc_async_read_set_trid(slot + 1);
            for (uint32_t done = 0; done < nt; done += PT) {  // the block's packets share its transaction id
                const uint32_t pt = nt - done < PT ? nt - done : PT;
                if (pt != PT) {
                    noc_async_read_one_packet_set_state<true>(bank_noc, pt * TB, 0);
                }
                noc_async_read_one_packet_with_state_with_trid(
                    bank_lo, run_addr + done * TB, slots + slot * BB + done * TB, slot + 1);
                if (pt != PT) {
                    noc_async_read_one_packet_set_state<true>(bank_noc, PT * TB, 0);
                }
            }
            ++issued;
            run_addr += nt * TB;
            run_left -= nt;
            advance();
        }
        if (pushed < issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, pushed % RS + 1)) {
            cb_push_back(cb_w, BT);
            ++pushed;
        }
    }
    noc_async_read_set_trid(0);
    noc_async_write_barrier();  // the root's table block
}
