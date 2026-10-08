// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows metadata and placement (data movement only). Reads the routing table the expert-row program wrote and
// writes moe_compute's outputs:
//   slot 0  per-expert token counts, one row in every grid core's shard (same L1 address on every core)
//   slot 1  one activation row per token with >= 1 local expert, ascending: [token, k per expert (k + 1 = not
//           routed), score bits per expert], then rows [-1, k + 1 .., 0 ..]
//   slot 2  page e: expert e's tokens ascending, one u32 per 16 B entry, then -1
//   slot 3  the last two local experts' rows in the double buffer, half e % 2, with moe_compute's geometry: expert e's
//           n_e rows split evenly over the ntp height shards (earlier shards take the remainder), row dt of shard h at
//           half * BLOCK + dt * SEG, bytes [w SEG, (w + 1) SEG) of the row.
// The metadata work is split over the NP cores of the program. ComputeOnly runs on the combine cores (ntp x dp, core
// i = shard (i / dp, i % dp)) and each core places its own slot-3 shard. With the combine (place_dense = 0) it runs
// on the feeder cores, feed.cpp fills slot 3, and once every core's writes landed core 0 increments the combine sync
// core's metadata semaphore once (the combine reader waits for exactly 1).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_s = get_named_compile_time_arg_val("cb_scratch");
    constexpr uint32_t T = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t K = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t E = get_named_compile_time_arg_val("local_experts");
    constexpr uint32_t NTP = get_named_compile_time_arg_val("height_shards");
    constexpr uint32_t SEG = get_named_compile_time_arg_val("segment_bytes");
    constexpr uint32_t BLOCK = get_named_compile_time_arg_val("half_bytes");
    constexpr uint32_t EAL = get_named_compile_time_arg_val("count_words");
    constexpr uint32_t ETW = get_named_compile_time_arg_val("e_t_page_bytes");
    constexpr uint32_t AROW = get_named_compile_time_arg_val("activation_row_bytes");
    constexpr uint32_t GW = get_named_compile_time_arg_val("grid_w");
    constexpr uint32_t GH = get_named_compile_time_arg_val("grid_h");
    constexpr uint32_t NP = get_named_compile_time_arg_val("program_cores");
    constexpr uint32_t ROW_BYTES = get_named_compile_time_arg_val("row_bytes");
    constexpr bool PLACE_DENSE = get_named_compile_time_arg_val("place_dense") == 1;
    constexpr uint32_t DONE_SEM = get_named_compile_time_arg_val("done_semaphore_id");
    constexpr uint32_t META_SEM = get_named_compile_time_arg_val("metadata_semaphore_id");
    // routing table block (rows_reader.cpp): u32 nrow; u16 counts / first rows [E]; u16 rows [T K] in expert-row
    // order; the same entries in token order: slot, token, score, k
    constexpr uint32_t P_RTC = get_named_compile_time_arg_val("table_counts");
    constexpr uint32_t P_RTO = get_named_compile_time_arg_val("table_offsets");
    constexpr uint32_t P_RTR = get_named_compile_time_arg_val("table_rows");
    constexpr uint32_t P_TES = get_named_compile_time_arg_val("table_entry_slots");
    constexpr uint32_t P_TET = get_named_compile_time_arg_val("table_entry_tokens");
    constexpr uint32_t P_TEC = get_named_compile_time_arg_val("table_entry_scores");
    constexpr uint32_t P_TEK = get_named_compile_time_arg_val("table_entry_k");
    constexpr uint32_t P_RTBS = get_named_compile_time_arg_val("table_bytes");
    // scratch layout (byte offsets in cb_s)
    constexpr uint32_t CNO = get_named_compile_time_arg_val("s_counts");
    constexpr uint32_t OFO = get_named_compile_time_arg_val("s_offsets");
    constexpr uint32_t ETB = get_named_compile_time_arg_val("s_e_t");
    constexpr uint32_t ARB = get_named_compile_time_arg_val("s_activation");
    constexpr uint32_t CRB = get_named_compile_time_arg_val("s_count_row");
    constexpr uint32_t RTB = get_named_compile_time_arg_val("s_table");
    constexpr auto rows_args = TensorAccessorArgs<0>();
    constexpr auto et_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    constexpr auto ac_args = TensorAccessorArgs<et_args.next_compile_time_args_offset()>();
    constexpr auto tab_args = TensorAccessorArgs<ac_args.next_compile_time_args_offset()>();
    constexpr uint32_t NAB = 8;  // activation rows in flight before a write barrier
    constexpr uint32_t KSEL = K + 1;

    uint32_t a = 0;
    const uint32_t rows_addr = get_arg_val<uint32_t>(a++);
    const uint32_t et_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ac_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++);  // slot 0: L1 address of every grid core's shard
    const uint32_t dense = get_arg_val<uint32_t>(a++);     // slot 3: this core's shard
    const uint32_t tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h = get_arg_val<uint32_t>(a++);
    const uint32_t w = get_arg_val<uint32_t>(a++);
    const uint32_t pi = get_arg_val<uint32_t>(a++);
    const uint32_t lead_x = get_arg_val<uint32_t>(a++);  // core 0 of the program and the combine sync core
    const uint32_t lead_y = get_arg_val<uint32_t>(a++);
    const uint32_t sync_x = get_arg_val<uint32_t>(a++);
    const uint32_t sync_y = get_arg_val<uint32_t>(a++);
    const uint32_t vx_at = a, vy_at = a + GW;

    cb_reserve_back(cb_s, 1);
    const uint32_t s = get_write_ptr(cb_s);
    tt_l1_ptr uint32_t* cnt = reinterpret_cast<tt_l1_ptr uint32_t*>(s + CNO);
    tt_l1_ptr uint32_t* ofs = reinterpret_cast<tt_l1_ptr uint32_t*>(s + OFO);
    const tt_l1_ptr uint16_t* lst = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + RTB + P_RTR);

    const auto tab = TensorAccessor(tab_args, tab_addr, P_RTBS);
    const uint64_t tb = tab.get_noc_addr(0);
    noc_async_read(tb, s + RTB, P_RTR);
    noc_async_read_barrier();
    const uint32_t nrow = *reinterpret_cast<const tt_l1_ptr uint32_t*>(s + RTB);
    const uint32_t rb = (nrow * 2 + 63) & ~63u;
    if (nrow) {
        const uint32_t parts[5] = {P_RTR, P_TES, P_TET, P_TEC, P_TEK};
        for (uint32_t i = 0; i < 5; ++i) {
            noc_async_read(tb + parts[i], s + RTB + parts[i], rb);
        }
    }
    {
        const tt_l1_ptr uint16_t* tc = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + RTB + P_RTC);
        const tt_l1_ptr uint16_t* to = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + RTB + P_RTO);
        for (uint32_t l = 0; l < E; ++l) {
            cnt[l] = tc[l];
            ofs[l] = to[l];
        }
    }
    noc_async_read_barrier();

    // ---- slot 0: counts, one row per grid core, cores pi, pi + NP, ... of the row-major grid ----
    volatile tt_l1_ptr uint32_t* crow = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + CRB);
    for (uint32_t i = 0; i < EAL; ++i) {
        crow[i] = i < E ? cnt[i] : 0;
    }
    for (uint32_t li = pi; li < GW * GH; li += NP) {
        const uint32_t vx = get_arg_val<uint32_t>(vx_at + li % GW);
        const uint32_t vy = get_arg_val<uint32_t>(vy_at + li / GW);
        noc_async_write(s + CRB, get_noc_addr(vx, vy, cnt_addr), EAL * 4);
    }
    // ---- slot 2: page e for the experts e with e % NP == pi ----
    const auto et = TensorAccessor(et_args, et_addr, ETW);
    volatile tt_l1_ptr uint32_t* eb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + ETB);
    for (uint32_t e = pi; e < E; e += NP) {
        noc_async_write_barrier();  // eb is reused per page
        const uint32_t n = cnt[e];
        for (uint32_t i = 0; i <= n; ++i) {
            eb[4 * i] = i < n ? uint32_t(lst[ofs[e] + i]) : 0xFFFFFFFFu;
            eb[4 * i + 1] = 0;
            eb[4 * i + 2] = 0;
            eb[4 * i + 3] = 0;
        }
        noc_async_write(s + ETB, et.get_noc_addr(e), (n + 1) * 16);
    }
    // ---- slot 1: row r is written by core r % NP; activated tokens first (ascending), then empty rows ----
    const auto ac = TensorAccessor(ac_args, ac_addr, T * AROW);
    uint32_t r = 0, buf = 0;
    auto row_buf = [&]() -> volatile tt_l1_ptr uint32_t* {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + ARB + buf * AROW);
    };
    const tt_l1_ptr uint16_t* tes = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + RTB + P_TES);
    const tt_l1_ptr uint16_t* tet = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + RTB + P_TET);
    const tt_l1_ptr uint16_t* tec = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + RTB + P_TEC);
    const tt_l1_ptr uint16_t* tek = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + RTB + P_TEK);
    uint32_t q0 = 0;
    while (q0 < nrow) {
        const uint32_t t = tet[q0];
        uint32_t q1 = q0;
        while (q1 < nrow && tet[q1] == t) {
            ++q1;
        }
        if (r % NP == pi) {
            if (buf == 0) {
                noc_async_write_barrier();  // all NAB buffers are about to be refilled
            }
            volatile tt_l1_ptr uint32_t* row = row_buf();
            row[0] = t;
            for (uint32_t e = 0; e < E; ++e) {
                row[1 + e] = KSEL;
                row[1 + E + e] = 0;
            }
            for (uint32_t i = 1 + 2 * E; i < AROW / 4; ++i) {
                row[i] = 0;  // alignment padding
            }
            for (uint32_t q = q0; q < q1; ++q) {
                row[1 + tes[q]] = tek[q];
                row[1 + E + tes[q]] = uint32_t(tec[q]);
            }
            noc_async_write(s + ARB + buf * AROW, ac.get_noc_addr(0, r * AROW), AROW);
            buf = (buf + 1) % NAB;
        }
        ++r;
        q0 = q1;
    }
    noc_async_write_barrier();
    buf = 0;
    {
        volatile tt_l1_ptr uint32_t* row = row_buf();
        row[0] = 0xFFFFFFFFu;
        for (uint32_t e = 0; e < E; ++e) {
            row[1 + e] = KSEL;
            row[1 + E + e] = 0;
        }
        for (uint32_t i = 1 + 2 * E; i < AROW / 4; ++i) {
            row[i] = 0;
        }
    }
    for (uint32_t rr = r; rr < T; ++rr) {
        if (rr % NP == pi) {
            noc_async_write(s + ARB, ac.get_noc_addr(0, rr * AROW), AROW);
        }
    }

    // ---- slot 3: the last two experts' rows into this core's shard ----
    if constexpr (PLACE_DENSE) {
        const auto rw = TensorAccessor(rows_args, rows_addr, ROW_BYTES);
        const uint32_t e_first = E >= 2 ? E - 2 : 0;
        for (uint32_t e = e_first; e < E; ++e) {
            const uint32_t n = cnt[e];
            const uint32_t q = n / NTP, rem = n % NTP;
            const uint32_t c = q + (h < rem ? 1 : 0);
            const uint32_t off = h * q + (h < rem ? h : rem);
            for (uint32_t dt = 0; dt < c; ++dt) {
                noc_async_read(rw.get_noc_addr(ofs[e] + off + dt, w * SEG), dense + (e % 2) * BLOCK + dt * SEG, SEG);
            }
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    if constexpr (!PLACE_DENSE) {
        const uint32_t done = get_semaphore(DONE_SEM);
        noc_semaphore_inc(get_noc_addr(lead_x, lead_y, done), 1);
        if (pi == 0) {
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done);
            noc_semaphore_wait(d, NP);
            noc_semaphore_set(d, 0);
            noc_semaphore_inc(get_noc_addr(sync_x, sync_y, get_semaphore(META_SEM)), 1);
        }
        noc_async_atomic_barrier();
    }
}
