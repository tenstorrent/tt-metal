// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — matmul-core reader (NCRISC, NoC0).
//
// Per entry (transport order): load_operand_block — Kt/kb K-block windows of A (shared along an n-line) and W (shared
// along an m-line), each delivered by Mcast1D families. A window may be split into two contiguous parts (A: by m
// rows, W: by k rows), each read from DRAM and multicast by a different sender of the line (families "a"/"a2",
// "b"/"b2"), so one RISC never serializes a whole window's read issue + multicast. The cross-block operand is
// resident after entry 0: its windows are re-credited (reserve + push, no NoC) because the CB holds the whole Kt
// slice and the ring wraps exactly. Loading is software-pipelined: a sender issues step s+1's DRAM reads while step
// s's multicast drains (send<CallerManaged>; every slot is flushed before it is overwritten).
// Then land_arrivals_block for the lagged (pending) entry: wait the global arrival counter of each arriving direction
// until it covers this entry (host threshold), credit the zone slot (the upstream port's fabric writes already landed
// the bytes). At the end: re-arm both counters by decrementing what arrived.
//
// Raw-API note (design "Helpers considered and rejected"): land_arrivals_block is a semaphore wait + CB credit of
// bytes a remote chip wrote; no helper covers a fabric fused-inc counter with host thresholds.

#include <cstdint>
#ifdef MRS_ZONES  // perf instrumentation only
#include <tools/profiler/kernel_profiler.hpp>
#define MRS_ZONE(name) DeviceZoneScopedN(name)
#else
#define MRS_ZONE(name)
#endif
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_act_k_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_k_window = get_compile_time_arg_val(1);
    constexpr uint32_t cb_arrivals_fwd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_arrivals_bwd = get_compile_time_arg_val(3);
    constexpr uint32_t core_m = get_compile_time_arg_val(4);
    constexpr uint32_t core_n = get_compile_time_arg_val(5);
    constexpr uint32_t kb = get_compile_time_arg_val(6);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t Kt = get_compile_time_arg_val(8);
    constexpr uint32_t Nt = get_compile_time_arg_val(9);
    constexpr uint32_t a_row_per_block = get_compile_time_arg_val(10);  // scatter -2: block_m, else 0
    constexpr uint32_t w_col_per_block = get_compile_time_arg_val(11);  // scatter -1: block_n, else 0
    constexpr bool resident_a = get_compile_time_arg_val(12) != 0;
    constexpr bool resident_w = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t P = get_compile_time_arg_val(14);
    constexpr uint32_t num_entries = get_compile_time_arg_val(15);
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t a_cap_windows = get_compile_time_arg_val(18);  // CB capacity in K-block windows
    constexpr uint32_t w_cap_windows = get_compile_time_arg_val(19);
    constexpr uint32_t a_split = get_compile_time_arg_val(20);         // m rows of part 0 (core_m: one part)
    constexpr uint32_t w_split = get_compile_time_arg_val(21);         // k rows of part 0 (kb: one part)
    constexpr uint32_t prefetch_steps = get_compile_time_arg_val(22);  // K-block windows a sender reads ahead
    constexpr auto a_args = TensorAccessorArgs<23>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto mc_a =
        McastArgs<get_named_compile_time_arg_val("a_ct_offset"), get_named_compile_time_arg_val("a_rt_offset")>();
    constexpr auto mc_a2 =
        McastArgs<get_named_compile_time_arg_val("a2_ct_offset"), get_named_compile_time_arg_val("a2_rt_offset")>();
    constexpr auto mc_b =
        McastArgs<get_named_compile_time_arg_val("b_ct_offset"), get_named_compile_time_arg_val("b_rt_offset")>();
    constexpr auto mc_b2 =
        McastArgs<get_named_compile_time_arg_val("b2_ct_offset"), get_named_compile_time_arg_val("b2_rt_offset")>();
    static_assert((a_split < core_m) == mc_a2.active, "A part 1 exists iff the A window is split");
    static_assert((w_split < kb) == mc_b2.active, "W part 1 exists iff the W window is split");

    constexpr uint32_t a_window_tiles = core_m * kb;
    constexpr uint32_t w_window_tiles = kb * core_n;
    constexpr uint32_t a_win_bytes = a_window_tiles * a_tile_bytes;
    constexpr uint32_t w_win_bytes = w_window_tiles * w_tile_bytes;
    // part p of a window: contiguous bytes [off, off + size)
    constexpr uint32_t a_part_off[2] = {0, a_split * kb * a_tile_bytes};
    constexpr uint32_t a_part_bytes[2] = {a_split * kb * a_tile_bytes, (core_m - a_split) * kb * a_tile_bytes};
    constexpr uint32_t w_part_off[2] = {0, w_split * core_n * w_tile_bytes};
    constexpr uint32_t w_part_bytes[2] = {w_split * core_n * w_tile_bytes, (kb - w_split) * core_n * w_tile_bytes};

    uint32_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t row0 = get_arg_val<uint32_t>(arg++);  // gm * core_m (block-local)
    const uint32_t col0 = get_arg_val<uint32_t>(arg++);  // gn * core_n (block-local)
    const uint32_t m_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t n_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t arr_fwd_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t arr_bwd_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t total_fwd = get_arg_val<uint32_t>(arg++);  // increments my own counters receive (groups I lead)
    const uint32_t total_bwd = get_arg_val<uint32_t>(arg++);
    const uint32_t observations = get_arg_val<uint32_t>(arg++);  // member acks of my counters before I re-arm
    // per entry: [block | arrival_mask << 16 | lag << 20, leader_fwd, leader_bwd]; leader = x | y << 8 | thr << 16:
    // the arrival has landed once that core's counter (incremented once per group of cores) reaches thr
    const uint32_t entries_idx = arg;
    constexpr uint32_t ENTRY_WORDS = 3;
    constexpr uint32_t POLL_TRID = 0xF;  // read transaction id of the leader-counter polls (prefetches use 1..8)
    const uint32_t poll_scratch = get_semaphore(2);
    const uint32_t observed_addr = get_semaphore(3);  // same id on every matmul core

    const auto a_acc = TensorAccessor(a_args, a_addr, a_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
    volatile tt_l1_ptr uint32_t* arr_fwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_fwd_addr);
    volatile tt_l1_ptr uint32_t* arr_bwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_bwd_addr);

    Noc noc;
    auto a_tx = mc_a.optional_sender(noc);
    auto a_rx = mc_a.optional_receiver(noc);
    auto a2_tx = mc_a2.optional_sender(noc);
    auto a2_rx = mc_a2.optional_receiver(noc);
    auto w_tx = mc_b.optional_sender(noc);
    auto w_rx = mc_b.optional_receiver(noc);
    auto w2_tx = mc_b2.optional_sender(noc);
    auto w2_rx = mc_b2.optional_receiver(noc);
    const bool a_sends[2] = {a_tx.has_value(), a2_tx.has_value()};
    const bool w_sends[2] = {w_tx.has_value(), w2_tx.has_value()};
    const bool a_sender = a_sends[0] || a_sends[1];
    const bool w_sender = w_sends[0] || w_sends[1];

    const uint32_t a_base = get_write_ptr(cb_act_k_window);  // window slot 0 (the write pointer is at the base)
    const uint32_t w_base = get_write_ptr(cb_weight_k_window);
    constexpr uint32_t num_steps = num_entries * num_k_blocks;
    auto block_of = [&](uint32_t step) {
        return get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * (step / num_k_blocks)) & 0xFFFF;
    };
    // the resident operand moves only in entry 0 (later entries re-credit the same windows)
    auto a_moves = [&](uint32_t step) { return !(resident_a && step >= num_k_blocks); };
    auto w_moves = [&](uint32_t step) { return !(resident_w && step >= num_k_blocks); };
    // DRAM reads of the parts this core sends, into window `dst`
    auto issue_a = [&](uint32_t step, uint32_t dst) {
#ifndef MRS_SKIP_DRAM  // perf ablation only
        MRS_ZONE("RD_DRAM");
        const uint32_t page0 = (block_of(step) * a_row_per_block + row0) * Kt + (step % num_k_blocks) * kb;
        for (uint32_t part = 0; part < 2; ++part) {
            if (!a_sends[part]) {
                continue;
            }
            const uint32_t i0 = part == 0 ? 0 : a_split;
            const uint32_t i1_nom = part == 0 ? a_split : core_m;
            const uint32_t i1 = i1_nom < m_valid ? i1_nom : m_valid;
            for (uint32_t i = i0; i < i1; ++i) {
                for (uint32_t kk = 0; kk < kb; ++kk) {
                    noc_async_read(
                        a_acc.get_noc_addr(page0 + i * Kt + kk), dst + (i * kb + kk) * a_tile_bytes, a_tile_bytes);
                }
            }
        }
#endif
    };
    auto issue_w = [&](uint32_t step, uint32_t dst) {
#ifndef MRS_SKIP_DRAM  // perf ablation only
        MRS_ZONE("RD_DRAM");
        const uint32_t page0 = (step % num_k_blocks) * kb * Nt + block_of(step) * w_col_per_block + col0;
        for (uint32_t part = 0; part < 2; ++part) {
            if (!w_sends[part]) {
                continue;
            }
            const uint32_t k0 = part == 0 ? 0 : w_split;
            const uint32_t k1 = part == 0 ? w_split : kb;
            for (uint32_t kk = k0; kk < k1; ++kk) {
                for (uint32_t j = 0; j < n_valid; ++j) {
                    noc_async_read(
                        w_acc.get_noc_addr(page0 + kk * Nt + j), dst + (kk * core_n + j) * w_tile_bytes, w_tile_bytes);
                }
            }
        }
#endif
    };

    // load_operand_block, software-pipelined over the flat (entry, K-block) step sequence. A sender keeps up to
    // prefetch_steps future windows reserved with their DRAM reads in flight (each step's reads carry their own
    // transaction id, so a step waits only for its own reads). Look-ahead windows are reserved only when free
    // (never blocking the current step's multicast on compute). Receivers never look ahead: their slot order is the
    // senders', so a multicast destination is always their current write pointer. Every core walks the parts in
    // the same order (a, a2, b, b2).
    constexpr uint32_t NUM_TRIDS = 8;
    auto trid_of = [](uint32_t step) { return 1 + step % NUM_TRIDS; };
    uint32_t a_slot = 0, w_slot = 0;      // window slot of the current step
    uint32_t a_issued = 0, w_issued = 0;  // steps whose window is reserved (and, if moving, whose reads are issued)
    auto advance = [&](uint32_t s,
                       bool sender,
                       bool moves_s,
                       uint32_t& issued,
                       uint32_t cb,
                       uint32_t win_tiles,
                       uint32_t slot,
                       uint32_t cap,
                       auto&& moves,
                       auto&& issue,
                       uint32_t base,
                       uint32_t win_bytes) {
        if (issued == s) {  // the current step: reserve (blocking) and, as a sender of a moving step, issue its reads
            cb_reserve_back(cb, win_tiles);
            if (sender && moves_s) {
                noc_async_read_set_trid(trid_of(s));
                issue(s, base + slot * win_bytes);
            }
            issued = s + 1;
        }
        if (!sender) {
            return;
        }
        while (issued < num_steps && issued <= s + prefetch_steps && moves(issued)) {
            const uint32_t ahead = issued - s + 1;  // windows from the current write pointer
            if (ahead > cap || !cb_pages_reservable_at_back(cb, ahead * win_tiles)) {
                break;
            }
            cb_reserve_back(cb, ahead * win_tiles);  // free: returns immediately
            uint32_t t_slot = slot + (issued - s);
            t_slot = t_slot >= cap ? t_slot - cap : t_slot;
            noc_async_read_set_trid(trid_of(issued));
            issue(issued, base + t_slot * win_bytes);
            ++issued;
        }
    };
    auto load_step = [&](uint32_t s) {
        // Every earlier multicast has left L1 before this step issues any read: a slot read into at step s (for a
        // step t <= s + cap - 1) was last a multicast source at step t - cap <= s - 1.
        noc_async_writes_flushed();
        {
            MRS_ZONE("RD_RESV");
            advance(
                s,
                a_sender,
                a_moves(s),
                a_issued,
                cb_act_k_window,
                a_window_tiles,
                a_slot,
                a_cap_windows,
                a_moves,
                issue_a,
                a_base,
                a_win_bytes);
            advance(
                s,
                w_sender,
                w_moves(s),
                w_issued,
                cb_weight_k_window,
                w_window_tiles,
                w_slot,
                w_cap_windows,
                w_moves,
                issue_w,
                w_base,
                w_win_bytes);
            if ((a_sender && a_moves(s)) || (w_sender && w_moves(s))) {
                noc_async_read_barrier_with_trid(trid_of(s));  // this step's reads only
            }
        }
        {
            MRS_ZONE("RD_MCAST");
            const uint32_t a_dst = a_base + a_slot * a_win_bytes;
            const uint32_t w_dst = w_base + w_slot * w_win_bytes;
            // CallerManaged: send() returns once the multicast is issued; the source slot is protected by the flush at
            // the start of the next step (no read targets it before then).
            if (a_moves(s)) {
                if (a_tx) {
                    a_tx->template send<SourceL1Guard::CallerManaged>(
                        a_dst + a_part_off[0], a_dst + a_part_off[0], a_part_bytes[0]);
                } else {
                    a_rx->receive();
                }
                if constexpr (mc_a2.active) {
                    if (a2_tx) {
                        a2_tx->template send<SourceL1Guard::CallerManaged>(
                            a_dst + a_part_off[1], a_dst + a_part_off[1], a_part_bytes[1]);
                    } else {
                        a2_rx->receive();
                    }
                }
            }
            if (w_moves(s)) {
                if (w_tx) {
                    w_tx->template send<SourceL1Guard::CallerManaged>(
                        w_dst + w_part_off[0], w_dst + w_part_off[0], w_part_bytes[0]);
                } else {
                    w_rx->receive();
                }
                if constexpr (mc_b2.active) {
                    if (w2_tx) {
                        w2_tx->template send<SourceL1Guard::CallerManaged>(
                            w_dst + w_part_off[1], w_dst + w_part_off[1], w_part_bytes[1]);
                    } else {
                        w2_rx->receive();
                    }
                }
            }
        }
        cb_push_back(cb_act_k_window, a_window_tiles);
        cb_push_back(cb_weight_k_window, w_window_tiles);
        a_slot = a_slot + 1 == a_cap_windows ? 0 : a_slot + 1;
        w_slot = w_slot + 1 == w_cap_windows ? 0 : w_slot + 1;
    };

    // wait until the leader's arrival counter (same address on every core) covers `thr`: a local wait when this core
    // leads its group, else poll the leader's counter over the NoC
    auto wait_leader = [&](uint32_t lead, uint32_t counter_addr) {
        const uint32_t x = lead & 0xFF, y = (lead >> 8) & 0xFF, thr = lead >> 16;
        if (x == my_x[noc_index] && y == my_y[noc_index]) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr), thr);
            return;
        }
        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll_scratch);
        const uint64_t src = get_noc_addr(x, y, counter_addr);
        while (true) {
            noc_async_read_set_trid(POLL_TRID);
            noc_async_read(src, poll_scratch, sizeof(uint32_t));
            noc_async_read_barrier_with_trid(POLL_TRID);
            if (*scratch >= thr) {
                break;
            }
        }
        noc_semaphore_inc(
            get_noc_addr(x, y, observed_addr), 1);  // the leader may re-arm only after every member saw it
    };
    // land_arrivals_block: the zone slot(s) of this entry are credited once their group's counter covers them.
    auto land_arrivals_block = [&](uint32_t entry_idx) {
        const uint32_t word0 = get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * entry_idx);
        const uint32_t mask = (word0 >> 16) & 0x3;
        MRS_ZONE("ARR_WAIT");
        if (mask & 1) {
            wait_leader(get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * entry_idx + 1), arr_fwd_addr);
            cb_reserve_back(cb_arrivals_fwd, P);
            cb_push_back(cb_arrivals_fwd, P);
        }
        if (mask & 2) {
            wait_leader(get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * entry_idx + 2), arr_bwd_addr);
            cb_reserve_back(cb_arrivals_bwd, P);
            cb_push_back(cb_arrivals_bwd, P);
        }
    };
    // host decision: finish this entry behind the next entry's matmul (its arrival is expected late)
    auto lagged = [&](uint32_t entry_idx) {
        return ((get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * entry_idx) >> 20) & 1) != 0;
    };

    constexpr uint32_t NONE = 0xFFFFFFFF;
    uint32_t pending = NONE;
    for (uint32_t e = 0; e < num_entries; ++e) {
        for (uint32_t kblk = 0; kblk < num_k_blocks; ++kblk) {
            load_step(e * num_k_blocks + kblk);
        }
        if (pending != NONE) {
            land_arrivals_block(pending);
            pending = NONE;
        }
        if (lagged(e)) {
            pending = e;  // stall shadow: the add waits behind the next entry's matmul
        } else {
            land_arrivals_block(e);
        }
    }

    noc_async_write_barrier();  // the last multicasts have landed before this kernel retires
    // Re-arm the global arrival counters for the next call (decrement what arrived, never store 0) once every group
    // member has observed them.
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(observed_addr), observations);
    if (total_fwd > 0) {
        noc_semaphore_wait_min(arr_fwd, total_fwd);
        noc_semaphore_inc(get_noc_addr(arr_fwd_addr), 0u - total_fwd);
    }
    if (total_bwd > 0) {
        noc_semaphore_wait_min(arr_bwd, total_bwd);
        noc_semaphore_inc(get_noc_addr(arr_bwd_addr), 0u - total_bwd);
    }
    noc_async_atomic_barrier();
}
