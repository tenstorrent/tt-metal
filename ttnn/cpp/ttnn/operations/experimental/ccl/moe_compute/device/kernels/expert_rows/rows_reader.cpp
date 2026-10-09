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
//    double-buffer half; later entries are dropped), and jobs of at most JR rows of one slot, dealt to the expert
//    groups: with one row tile per job every job has the same predicted time and job j of the slot order runs on group
//    j % G (the table stays in slot order); with several each job goes to the group with the earliest predicted finish
//    and the jobs are listed group by group. Every other core reads the table back.
//    The root also writes the table block to DRAM for the metadata program.
// 2. Streams this core's share of each job's expert weights in compute order P1(0), P1(1), P2(0), P1(2), P2(1), ...:
//    P1 = this core's W0/W1 column groups, P2 = its W2 output groups, both in moe_compute's prepared layout (the K
//    padding rows at each group's end are not read): a W0/W1 group is read from the bank pieces of the (layer, expert)
//    stream it lies in (a ring position's slice can go on in the next bank), a W2 group from this ring position's own
//    bank; a half block-column and a half-width last W2 group hold 2 tiles per K row. A unit is what the compute holds
//    at once: one group's run for a job of one row tile; for a job of several (M > 1) one chunk of one group, P1
//    chunk-major (chunk 0 of every column group, then chunk 1, ...), P2 group-major, each unit padded to the same
//    number of CB blocks.
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
    // weights in moe_compute's prepared layout (tiles, PreparedLayout in moe_expert_rows.cpp): the stride of a ring
    // position's W0/W1 column groups and a bank piece of W0/W1 per (layer, expert); the stride of its W2 output groups
    // and its W2 per (layer, expert)
    constexpr uint32_t P1_STRIDE = get_named_compile_time_arg_val("w0_w1_group_tiles");
    constexpr uint32_t P1_PIECE = get_named_compile_time_arg_val("w0_w1_piece_tiles");
    constexpr uint32_t P2_STRIDE = get_named_compile_time_arg_val("w2_group_tiles");
    constexpr uint32_t P2_SLICE = get_named_compile_time_arg_val("w2_slice_tiles");
    constexpr uint32_t M = get_named_compile_time_arg_val("row_tiles");
    constexpr uint32_t KC = get_named_compile_time_arg_val("chunk_tiles");
    constexpr uint32_t SB = get_named_compile_time_arg_val("chunk_blocks");
    constexpr uint32_t KC2 = get_named_compile_time_arg_val("w2_chunk_rows");
    constexpr uint32_t KT = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t NT = get_named_compile_time_arg_val("intermediate_tiles");
    constexpr uint32_t BIAS = get_named_compile_time_arg_val("has_bias");
    constexpr uint32_t CTL_WORDS = get_named_compile_time_arg_val("ctl_words");
    // M > 1: the planner's predicted time of a job (host-scaled cycles): its weight stream while the G groups share
    // DRAM, the busiest core's matmuls and partial reloads per row tile of a held job, and those of a one-row-tile job
    constexpr uint32_t DEAL_STREAM = get_named_compile_time_arg_val("deal_stream");
    constexpr uint32_t DEAL_ROW_TILE = get_named_compile_time_arg_val("deal_row_tile");
    constexpr uint32_t DEAL_SINGLE = get_named_compile_time_arg_val("deal_single");
    constexpr bool HELD = M > 1;  // jobs of several row tiles: held units, padded passes, per-job row tiles
    // cb_rt ctl block: job count, touched slots, rows, then (M > 1) the first job of each group and the job count
    constexpr uint32_t CTL_BYTES = HELD ? (16 + 2 * (G + 1) + 15) & ~15u : 16;

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
    const uint32_t bank_id = get_arg_val<uint32_t>(a++);  // this ring position's bank (its W2)
    const uint32_t idx_addr = get_arg_val<uint32_t>(a++);
    const uint32_t sc_addr = get_arg_val<uint32_t>(a++);
    const uint32_t map_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w01_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t s0 = get_arg_val<uint32_t>(a++);  // first W0/W1 tile of this core's groups in the expert stream
    const uint32_t ng = get_arg_val<uint32_t>(a++);
    const uint32_t q0 = get_arg_val<uint32_t>(a++), nq = get_arg_val<uint32_t>(a++);
    const uint32_t g = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++), NP = get_arg_val<uint32_t>(a++);
    const uint32_t rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t device_id = get_arg_val<uint32_t>(a++);
    // the last W0/W1 group is a half block-column, the last W2 group the half-width last a2a iteration
    const bool half_col = get_arg_val<uint32_t>(a++) == 1;
    const bool half_out = get_arg_val<uint32_t>(a++) == 1;
    // common args: mapping row of each source (count first), virtual x / y of the grid, mask of the program's cores,
    // bank of each W0/W1 piece (shard)
    const uint32_t src_at = 0;
    const uint32_t vx_at = get_common_arg_val<uint32_t>(0) + 1, vy_at = vx_at + GW, all_at = vy_at + GH;
    const uint32_t piece_bank_at = all_at + MW;
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
    tt_l1_ptr uint16_t* gs = reinterpret_cast<tt_l1_ptr uint16_t*>(rt + CTO + 16);
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
        if constexpr (HELD) {
            // Deal: jobs of more row tiles first (every slot's full jobs of M row tiles as the slots are walked, then
            // the slots' last partial jobs by row tiles, from bucket lists), each to the group with the earliest
            // predicted finish, on a tie the one with the least matmul work (stream-bound jobs tie), then the lower
            // group; at most CTL_WORDS - 1 per group (cb_ctl's list; G (CTL_WORDS - 1) is at least the most jobs of a
            // call). Each group then runs its jobs in slot order, so that jobs of many and of few row tiles run at
            // the same time on different groups. The deal is recorded per job (slot order) in the entry areas (read
            // by now), then the jobs are written group by group (group k's from gs[k]).
            static_assert(G <= 32, "expert groups");
            constexpr uint32_t END = 0xFFFF;
            // until the jobs are written: bucket lists of the slots with a partial job, that job's index
            tt_l1_ptr uint16_t* next = jobs;
            tt_l1_ptr uint16_t* part = jobs + d;
            tt_l1_ptr uint32_t* dealt = reinterpret_cast<tt_l1_ptr uint32_t*>(rt + ENO);  // first | slot, rows | group
            uint32_t head[M + 1];
            for (uint32_t t = 0; t <= M; ++t) {
                head[t] = END;
            }
            uint32_t load[G], work[G], fill[G];
#pragma GCC unroll 32
            for (uint32_t k = 0; k < G; ++k) {
                load[k] = 0;
                work[k] = 0;
                fill[k] = 0;
            }
            auto deal = [&](uint32_t id, uint32_t l, uint32_t first, uint32_t n) {
                const uint32_t m = (n + 31) / 32;
                const uint32_t w = m > 1 ? m * DEAL_ROW_TILE : DEAL_SINGLE;
                uint32_t to = G, best_load = 0xFFFFFFFF, best_work = 0xFFFFFFFF;
#pragma GCC unroll 32
                for (uint32_t k = 0; k < G; ++k) {
                    const uint32_t lk = load[k], wk = work[k];
                    if (fill[k] + 1 < CTL_WORDS && (lk < best_load || (lk == best_load && wk < best_work))) {
                        to = k;
                        best_load = lk;
                        best_work = wk;
                    }
                }
                load[to] += w > DEAL_STREAM ? w : DEAL_STREAM;
                work[to] += w;
                ++fill[to];
                dealt[2 * id] = first | (l << 16);
                dealt[2 * id + 1] = n | (to << 16);
            };
            for (uint32_t i = 0; i < d; ++i) {
                const uint32_t l = list[i], c = cnt[l], o = ofs[l];
                const uint32_t full = c / JR, r = c - full * JR;
                for (uint32_t f = 0; f < full; ++f) {
                    deal(nj++, l, o + f * JR, JR);
                }
                if (r) {
                    const uint32_t t = (r + 31) / 32;
                    next[i] = head[t];
                    head[t] = i;
                    part[i] = nj++;
                }
            }
            for (uint32_t t = M; t >= 1; --t) {
                for (uint32_t i = head[t]; i != END; i = next[i]) {
                    const uint32_t l = list[i], c = cnt[l], r = c % JR;
                    deal(part[i], l, ofs[l] + c - r, r);
                }
            }
            gs[0] = 0;
            for (uint32_t k = 0; k < G; ++k) {
                gs[k + 1] = gs[k] + fill[k];
                fill[k] = gs[k];  // write cursor
            }
            for (uint32_t q = 0; q < nj; ++q) {
                const uint32_t lo = dealt[2 * q], hi = dealt[2 * q + 1];
                const uint32_t p = fill[hi >> 16]++;
                jobs[3 * p] = lo >> 16;
                jobs[3 * p + 1] = lo & 0xFFFF;
                jobs[3 * p + 2] = hi & 0xFFFF;
            }
        } else {
            for (uint32_t i = 0; i < d; ++i) {
                const uint32_t l = list[i];
                for (uint32_t r0 = 0; r0 < cnt[l]; r0 += JR) {
                    jobs[3 * nj] = l;
                    jobs[3 * nj + 1] = ofs[l] + r0;
                    jobs[3 * nj + 2] = cnt[l] - r0 < JR ? cnt[l] - r0 : JR;
                    ++nj;
                }
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
        noc_async_read(get_noc_addr(rx, ry, rt + CTO), rt + CTO, CTL_BYTES);
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
    // this group's jobs (local index e): g, g + G, ... with one row tile per job, else as dealt above
    const uint32_t D = HELD ? gs[g + 1] - gs[g] : (nj > g ? (nj - g + G - 1) / G : 0);
    auto job_at = [&](uint32_t e) -> uint32_t { return HELD ? gs[g] + e : g + G * e; };
    cc[0] = D;
    if constexpr (HELD) {
        // row tiles of each of this group's jobs
        for (uint32_t e = 0; e < D && e + 1 < CTL_WORDS; ++e) {
            cc[1 + e] = (jobs[3 * job_at(e) + 2] + 31) / 32;
        }
    }
    cb_push_back(cb_ctl, 1);
    if (D == 0) {
        noc_async_write_barrier();  // the root's table block
        return;
    }

    // ---- weight stream: passes P1(0), P1(1), P2(0), P1(2), P2(1), ..., P2(D - 1) ----
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
    // A pass streams one job's P1 or P2 share. Jobs of several row tiles hold each unit for all their row tiles: x
    // chunk c of column group u (chunk-major), W2 chunk p of output group q, every unit padded to SB blocks. A job of
    // one row tile streams whole W2 runs and x chunks as large as an x slot of M row tiles holds (all of x when
    // M = 1); when held units exist (M > 1) its pass is padded to whole units so that the next held unit starts at a
    // unit boundary of the CB.
    constexpr uint32_t KC1 = KC * M < KT ? KC * M : KT;  // x chunk of a one-row-tile job
    constexpr uint32_t C1 = (KT + KC - 1) / KC;
    constexpr uint32_t C1S = (KT + KC1 - 1) / KC1;
    constexpr uint32_t C2 = (NT + KC2 - 1) / KC2;
    constexpr uint32_t UNIT_BLOCKS = HELD ? SB : 1;
    auto pad_to_unit = [](uint32_t b) { return (UNIT_BLOCKS - b % UNIT_BLOCKS) % UNIT_BLOCKS; };
    // tiles per K row of W0/W1 group u and of W2 group q: 4, or 2 for a half group (always a core's last)
    auto p1_width = [&](uint32_t u) -> uint32_t { return half_col && u + 1 == ng ? 2 : 4; };
    auto p2_width = [&](uint32_t q) -> uint32_t { return half_out && q + 1 == nq ? 2 : 4; };
    // K rows of chunk c of `chunks` chunks of kc of the k rows (the bias row ends the last)
    auto chunk_rows = [](uint32_t c, uint32_t kc, uint32_t chunks, uint32_t k) {
        return (c + 1 < chunks ? kc : k - c * kc) + (c + 1 == chunks ? BIAS : 0);
    };
    // blocks of a one-row-tile pass before its padding
    auto stream_blocks = [&](uint32_t ph) {
        uint32_t b = 0;
        if (ph == 1) {
            for (uint32_t q = 0; q < nq; ++q) {
                b += (p2_width(q) * (NT + BIAS) + BT - 1) / BT;
            }
            return b;
        }
        for (uint32_t u = 0; u < ng; ++u) {
            for (uint32_t c = 0; c < C1S; ++c) {
                b += (p1_width(u) * chunk_rows(c, KC1, C1S, KT) + BT - 1) / BT;
            }
        }
        return b;
    };
    // read cursor: byte address in its bank, that bank, the W0/W1 piece and the tiles to its end (a W2 run never
    // leaves its bank)
    uint32_t addr = 0, rd_bank = bank_id, piece = 0, piece_left = 0;
    // unit i of a pass: read cursor, real tiles, CB blocks; unit `units` of a padded pass carries no data
    auto unit_of = [&](uint32_t ph, bool held, uint32_t slot, uint32_t i, uint32_t& tiles, uint32_t& blocks) {
        if (ph == 0) {
            const uint32_t kc = held ? KC : KC1, chunks = held ? C1 : C1S;
            const uint32_t u = i % ng, c = i / ng, w = p1_width(u);
            const uint32_t at = s0 + u * P1_STRIDE + w * c * kc;  // tile of the (layer, expert) stream
            piece = at / P1_PIECE;
            piece_left = (piece + 1) * P1_PIECE - at;
            rd_bank = get_common_arg_val<uint32_t>(piece_bank_at + piece);
            addr = w01_addr + (slot * P1_PIECE + at - piece * P1_PIECE) * TB;
            tiles = w * chunk_rows(c, kc, chunks, KT);
        } else {
            const uint32_t kc = held ? KC2 : NT, chunks = held ? C2 : 1;
            const uint32_t q = i / chunks, p = i % chunks, w = p2_width(q);
            rd_bank = bank_id;
            piece_left = 0xFFFFFFFF;
            addr = w2_addr + (slot * P2_SLICE + (q0 + q) * P2_STRIDE + w * p * kc) * TB;
            tiles = w * chunk_rows(p, kc, chunks, NT);
        }
        blocks = held ? SB : (tiles + BT - 1) / BT;
    };
    const uint32_t slots = get_write_ptr(cb_w);
    // the bank and packet size the read command buffer is set to (set_state), changed only when a packet needs others
    uint32_t state_bank = 0, state_tiles = 0, bank_lo = 0;
    auto set_read_state = [&](uint32_t bank, uint32_t tiles) {
        const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank, 0);
        noc_async_read_one_packet_set_state<true>(bank_noc, tiles * TB, 0);
        bank_lo = (uint32_t)bank_noc;
        state_bank = bank;
        state_tiles = tiles;
    };
    set_read_state(bank_id, PT);
    // cursor over (pass, unit, block)
    uint32_t pass = 0, ph = 0, slot = 0, unit = 0, units = 0, left = 0, block = 0, blocks = 0;
    bool held = false;
    auto open_unit = [&]() {
        uint32_t tiles = 0;
        if (unit < units) {
            unit_of(ph, held, slot, unit, tiles, blocks);
        } else {
            blocks = pad_to_unit(stream_blocks(ph));
        }
        left = tiles;
        block = 0;
    };
    auto open_pass = [&]() {
        uint32_t e;
        pass_of(pass, e, ph);
        const uint32_t j = job_at(e);
        slot = jobs[3 * j];
        held = HELD && jobs[3 * j + 2] > 32;
        units = ph == 0 ? ng * (held ? C1 : C1S) : (held ? nq * C2 : nq);
        unit = 0;
        open_unit();
    };
    // units of the open pass, plus the padding unit of a one-row-tile pass when M > 1
    auto last_unit = [&]() { return units + (HELD && !held ? 1 : 0); };
    auto advance = [&]() {
        while (block == blocks) {
            if (++unit < last_unit()) {
                open_unit();
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
        uint32_t e, p;
        pass_of(s, e, p);
        if (HELD && jobs[3 * job_at(e) + 2] > 32) {
            total += (p == 0 ? ng * C1 : nq * C2) * SB;
        } else {
            const uint32_t b = stream_blocks(p);
            total += b + (HELD ? pad_to_unit(b) : 0);
        }
    }
    // one transaction id per block in flight (ids 1..NOC_MAX_TRANSACTION_ID; IF blocks are in flight at most)
    constexpr uint32_t TRIDS = RS < NOC_MAX_TRANSACTION_ID ? RS : NOC_MAX_TRANSACTION_ID;
    static_assert(IF < TRIDS, "transaction ids");
    uint32_t issued = 0, pushed = 0;
    while (pushed < total) {
        if (issued < total && issued - pushed < IF && cb_pages_reservable_at_back(cb_w, BT * (issued - pushed + 1))) {
            const uint32_t bslot = issued % RS;
            const uint32_t trid = issued % TRIDS + 1;
            const uint32_t nt = left < BT ? left : BT;
            if (nt) {  // a unit's padding blocks carry no data
                noc_async_read_set_trid(trid);
                // the block's packets share its transaction id; a packet ends where the cursor's bank piece does
                for (uint32_t done = 0; done < nt;) {
                    if (piece_left == 0) {  // the stream goes on at this (layer, expert)'s start in the next bank
                        rd_bank = get_common_arg_val<uint32_t>(piece_bank_at + ++piece);
                        addr -= P1_PIECE * TB;
                        piece_left = P1_PIECE;
                    }
                    uint32_t pt = nt - done < PT ? nt - done : PT;
                    pt = pt < piece_left ? pt : piece_left;
                    if (rd_bank != state_bank || pt != state_tiles) {
                        set_read_state(rd_bank, pt);
                    }
                    noc_async_read_one_packet_with_state_with_trid(bank_lo, addr, slots + bslot * BB + done * TB, trid);
                    done += pt;
                    addr += pt * TB;
                    piece_left -= pt;
                }
            }
            ++issued;
            left -= nt;
            ++block;
            advance();
        }
        if (pushed < issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, pushed % TRIDS + 1)) {
            cb_push_back(cb_w, BT);
            ++pushed;
        }
    }
    noc_async_read_set_trid(0);
    noc_async_write_barrier();  // the root's table block
}
