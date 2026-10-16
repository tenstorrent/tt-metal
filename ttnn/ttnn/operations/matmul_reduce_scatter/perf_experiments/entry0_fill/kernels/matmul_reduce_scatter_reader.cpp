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
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"  // permanent stage zones (free when the profiler is off)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

#ifndef MRS_FILL_SPLIT
#define MRS_FILL_SPLIT 0  // 0: baseline walk | 1: ack all at mcast start, sends then waits | 2: ack right after
// reserve (before reads), sends then waits | 3: ack right after reserve, then the baseline a, a2, b, b2 order (send or
// wait) | 5: early ack of the A families only (W families keep the fused ack + wait, so every W multicast still starts
// after its whole line received A: grid-wide A-then-W phases) | 4: ack right after reserve, then per operand: send my
// parts, wait the others (A phase, then W phase)
#endif
#ifndef MRS_FILL_SPLIT_ENTRIES
#define MRS_FILL_SPLIT_ENTRIES 0xFFFF  // split-phase walk only for the first N entries' steps (1: the fill only)
#endif
#ifndef MRS_FILL_FULL_STEPS
#define MRS_FILL_FULL_STEPS \
    0  // the first N steps use the full split (mode 2: every family acked early, sends then waits)
#endif
#ifndef MRS_FILL_FAST_ADDR
#define MRS_FILL_FAST_ADDR \
    0  // 1: DRAM read addresses from a per-step table (one accessor lookup per bank, not per tile)
#endif
#ifndef MRS_FILL_LA0_W_LATE
#define MRS_FILL_LA0_W_LATE 0  // step 0 only: W look-ahead reads after step 0's W phase (not between barrier and send)
#endif
#ifndef MRS_FILL_POLL_ARR
#define MRS_FILL_POLL_ARR 0  // 1: land arrivals by polling (between / inside load steps) instead of blocking the loader
#endif
#ifndef MRS_FILL_LA_AFTER_SEND
#define MRS_FILL_LA_AFTER_SEND 0  // split steps: issue the look-ahead reads after my sends instead of before them
#endif
#ifdef MRS_FILL_ZONES
constexpr bool fill_zones = true;
#else
constexpr bool fill_zones = false;
#endif
// zone a call only on instrumented steps (`zs` in scope)
#define FZ(name, ...)               \
    if (zs) {                       \
        MaybeDeviceZoneScope(name); \
        __VA_ARGS__;                \
    } else {                        \
        __VA_ARGS__;                \
    }
// split-phase halves of ReceiverPipe::receive (Flag signal, pre-handshake, fixed sender); no-ops for an absent family
template <typename MC>
FORCE_INLINE void rx_ack(const MC& mc, const Noc& noc) {
    if constexpr (MC::active) {
        static_assert(MC::pre_handshake && MC::signal == DataReadySignal::Flag && !MC::rotating);
        Semaphore(detail::mcast_semaphore_id(MC::consumer_ready)).up(noc, mc.sender_x(), mc.sender_y(), 1);
    }
}
template <typename MC, typename Poll>
FORCE_INLINE void rx_wait(const MC&, Poll&& poll) {
    if constexpr (MC::active) {
        Semaphore sem(detail::mcast_semaphore_id(MC::data_ready));
        while (sem.value() != VALID) {
            poll();
        }
        sem.set(INVALID);
    }
}

void kernel_main() {
    constexpr uint32_t cb_act_k_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_k_window = get_compile_time_arg_val(1);
    constexpr uint32_t cb_arrivals_fwd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_arrivals_bwd = get_compile_time_arg_val(3);
    // streamed windows of one wave: A core_m x kb, W kb x core_n (with waves, the split axis is a wave's extent; the
    // resident operand's axis is never split)
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
    constexpr uint32_t num_entries = get_compile_time_arg_val(15);  // G x num_waves
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t a_cap_windows = get_compile_time_arg_val(18);  // CB capacity in K-block windows
    constexpr uint32_t w_cap_windows = get_compile_time_arg_val(19);
    constexpr uint32_t a_split = get_compile_time_arg_val(20);                   // m rows of part 0 (core_m: one part)
    constexpr uint32_t w_split = get_compile_time_arg_val(21);                   // k rows of part 0 (kb: one part)
    constexpr uint32_t prefetch_steps = get_compile_time_arg_val(22);            // K-block windows a sender reads ahead
    constexpr bool lookahead_after_barrier = get_compile_time_arg_val(23) != 0;  // issue look-ahead reads after the
    // current step's read barrier (else before it)
    // NoC of each operand's DRAM reads. This kernel owns every read on both NoCs of its core (the writer only writes),
    // so the second NoC adds DRAM-response links for the fill
    constexpr uint8_t a_read_noc = get_compile_time_arg_val(24);
    constexpr uint8_t w_read_noc = get_compile_time_arg_val(25);
    constexpr bool two_read_nocs = a_read_noc != w_read_noc || a_read_noc != noc_index;
    constexpr uint32_t num_waves = get_compile_time_arg_val(26);  // entries per block (entry e is wave e mod num_waves)
    constexpr auto a_args = TensorAccessorArgs<27>();
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
    const uint32_t total_fwd = get_arg_val<uint32_t>(arg++);  // arrival increments my counters receive this call
    const uint32_t total_bwd = get_arg_val<uint32_t>(arg++);
    // per wave: row_off | col_off << 8 | rows << 16 | cols << 24 inside my per-core block
    const uint32_t waves_idx = arg;
    // per entry: [block | arrival_mask << 16 | lag << 20, thr_fwd | thr_bwd << 16]; an arrival has landed once my
    // counter of its direction reaches its threshold (one fused increment per (core, unit), in the upstream's order)
    const uint32_t entries_idx = waves_idx + num_waves;
    constexpr uint32_t ENTRY_WORDS = 2;

    const auto a_acc = TensorAccessor(a_args, a_addr, a_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
    volatile tt_l1_ptr uint32_t* arr_fwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_fwd_addr);
    volatile tt_l1_ptr uint32_t* arr_bwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_bwd_addr);

    // ---- entry0_fill: polled arrival landing (MRS_FILL_POLL_ARR) -------------------------------------------------
    // The baseline lands an entry's arrivals with a BLOCKING wait between two entries' loads, so a late arrival
    // stalls this core's next-entry multicasts (and, through the handshakes, its lines). Polled: land, in entry order,
    // every entry whose counters already cover it, at every step start and inside every wait of the loader. Never
    // blocks on the fabric; the final blocking drain after the last load keeps the baseline's end state.
    uint32_t land_next = 0;  // next entry (in order) to land
    auto try_land = [&]() {
        while (land_next < num_entries) {
            const uint32_t word0 = get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * land_next);
            const uint32_t word1 = get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * land_next + 1);
            const uint32_t mask = (word0 >> 16) & 0x3;
            invalidate_l1_cache();
            if ((mask & 1) && (*arr_fwd < (word1 & 0xFFFF) || !cb_pages_reservable_at_back(cb_arrivals_fwd, P))) {
                return;
            }
            if ((mask & 2) && (*arr_bwd < (word1 >> 16) || !cb_pages_reservable_at_back(cb_arrivals_bwd, P))) {
                return;
            }
            if (mask & 1) {
                cb_reserve_back(cb_arrivals_fwd, P);
                cb_push_back(cb_arrivals_fwd, P);
            }
            if (mask & 2) {
                cb_reserve_back(cb_arrivals_bwd, P);
                cb_push_back(cb_arrivals_bwd, P);
            }
            ++land_next;
        }
    };
    auto poll_arrivals = [&]() {
        if constexpr (MRS_FILL_POLL_ARR) {
            try_land();
        }
    };

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
    auto wave_of = [&](uint32_t step) { return get_arg_val<uint32_t>(waves_idx + (step / num_k_blocks) % num_waves); };
    // valid extent of a wave along one axis: my valid tiles past the wave's offset, capped at its extent
    auto wave_valid = [](uint32_t valid, uint32_t off, uint32_t ext) {
        return valid <= off ? 0u : (valid - off < ext ? valid - off : ext);
    };
    // the resident operand moves only in entry 0 (later entries re-credit the same windows)
    auto a_moves = [&](uint32_t step) { return !(resident_a && step >= num_k_blocks); };
    auto w_moves = [&](uint32_t step) { return !(resident_w && step >= num_k_blocks); };
    // DRAM reads of the parts this core sends, into window `dst`
    auto issue_a = [&](uint32_t step, uint32_t dst) {
#ifndef MRS_SKIP_DRAM  // perf ablation only
        MaybeDeviceZoneScope("RD_DRAM");
        const uint32_t wave = wave_of(step);
        const uint32_t row_off = wave & 0xFF;
        const uint32_t m_valid_w = wave_valid(m_valid, row_off, (wave >> 16) & 0xFF);
        const uint32_t page0 = (block_of(step) * a_row_per_block + row0 + row_off) * Kt + (step % num_k_blocks) * kb;
        for (uint32_t part = 0; part < 2; ++part) {
            if (!a_sends[part]) {
                continue;
            }
            const uint32_t i0 = part == 0 ? 0 : a_split;
            const uint32_t i1_nom = part == 0 ? a_split : core_m;
            const uint32_t i1 = i1_nom < m_valid_w ? i1_nom : m_valid_w;
            if constexpr (MRS_FILL_FAST_ADDR && Kt % NUM_DRAM_BANKS == 0) {
                if (i0 >= i1) {
                    continue;
                }
                uint64_t first[NUM_DRAM_BANKS];
                constexpr uint32_t nb = kb < NUM_DRAM_BANKS ? kb : NUM_DRAM_BANKS;
                for (uint32_t kk = 0; kk < nb; ++kk) {
                    first[kk] = a_acc.get_noc_addr(page0 + i0 * Kt + kk, 0, a_read_noc);
                }
                constexpr uint32_t row_bytes = (Kt / NUM_DRAM_BANKS) * a_tile_bytes;
                uint32_t row_off = 0;
                for (uint32_t i = i0; i < i1; ++i, row_off += row_bytes) {
                    uint32_t b = 0, wrap_off = row_off;
                    for (uint32_t kk = 0; kk < kb; ++kk) {
                        noc_async_read(
                            first[b] + wrap_off, dst + (i * kb + kk) * a_tile_bytes, a_tile_bytes, a_read_noc);
                        if (++b == NUM_DRAM_BANKS) {
                            b = 0;
                            wrap_off += a_tile_bytes;
                        }
                    }
                }
                continue;
            }
            for (uint32_t i = i0; i < i1; ++i) {
                for (uint32_t kk = 0; kk < kb; ++kk) {
                    noc_async_read(
                        a_acc.get_noc_addr(page0 + i * Kt + kk, 0, a_read_noc),
                        dst + (i * kb + kk) * a_tile_bytes,
                        a_tile_bytes,
                        a_read_noc);
                }
            }
        }
#endif
    };
    auto issue_w = [&](uint32_t step, uint32_t dst) {
#ifndef MRS_SKIP_DRAM  // perf ablation only
        MaybeDeviceZoneScope("RD_DRAM");
        const uint32_t wave = wave_of(step);
        const uint32_t col_off = (wave >> 8) & 0xFF;
        const uint32_t n_valid_w = wave_valid(n_valid, col_off, wave >> 24);
        const uint32_t page0 = (step % num_k_blocks) * kb * Nt + block_of(step) * w_col_per_block + col0 + col_off;
        for (uint32_t part = 0; part < 2; ++part) {
            if (!w_sends[part]) {
                continue;
            }
            const uint32_t k0 = part == 0 ? 0 : w_split;
            const uint32_t k1 = part == 0 ? w_split : kb;
            if constexpr (MRS_FILL_FAST_ADDR && Nt % NUM_DRAM_BANKS == 0) {
                // interleaved DRAM: page p + NUM_DRAM_BANKS is page p's bank at + one page; a row Nt pages (a
                // multiple of the bank count) down is the same bank at + Nt / NUM_DRAM_BANKS pages. So one accessor
                // lookup per bank of the first row addresses the whole part.
                uint64_t first[NUM_DRAM_BANKS];
                const uint32_t nb = n_valid_w < NUM_DRAM_BANKS ? n_valid_w : NUM_DRAM_BANKS;
                for (uint32_t j = 0; j < nb; ++j) {
                    first[j] = w_acc.get_noc_addr(page0 + k0 * Nt + j, 0, w_read_noc);
                }
                constexpr uint32_t row_bytes = (Nt / NUM_DRAM_BANKS) * w_tile_bytes;
                uint32_t row_off = 0;
                for (uint32_t kk = k0; kk < k1; ++kk, row_off += row_bytes) {
                    uint32_t b = 0, wrap_off = row_off;
                    for (uint32_t j = 0; j < n_valid_w; ++j) {
                        noc_async_read(
                            first[b] + wrap_off, dst + (kk * core_n + j) * w_tile_bytes, w_tile_bytes, w_read_noc);
                        if (++b == NUM_DRAM_BANKS) {
                            b = 0;
                            wrap_off += w_tile_bytes;
                        }
                    }
                }
                continue;
            }
            for (uint32_t kk = k0; kk < k1; ++kk) {
                for (uint32_t j = 0; j < n_valid_w; ++j) {
                    noc_async_read(
                        w_acc.get_noc_addr(page0 + kk * Nt + j, 0, w_read_noc),
                        dst + (kk * core_n + j) * w_tile_bytes,
                        w_tile_bytes,
                        w_read_noc);
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
    auto set_trid = [](uint32_t trid) {
        noc_async_read_set_trid(trid, 0);
        if constexpr (two_read_nocs) {
            noc_async_read_set_trid(trid, 1);
        }
    };
    uint32_t a_slot = 0, w_slot = 0;      // window slot of the current step
    uint32_t a_issued = 0, w_issued = 0;  // steps whose window is reserved (and, if moving, whose reads are issued)
    // current step: reserve its window (blocking) and, as a sender of a moving step, issue its reads if no earlier
    // look-ahead did
    auto reserve_current = [&](uint32_t s,
                               bool sender,
                               bool moves_s,
                               uint32_t& issued,
                               uint32_t cb,
                               uint32_t win_tiles,
                               uint32_t slot,
                               auto&& issue,
                               uint32_t base,
                               uint32_t win_bytes) {
        if (issued == s) {
            if constexpr (MRS_FILL_POLL_ARR) {
                while (!cb_pages_reservable_at_back(cb, win_tiles)) {
                    try_land();
                }
            }
            cb_reserve_back(cb, win_tiles);
            if (sender && moves_s) {
                set_trid(trid_of(s));
                issue(s, base + slot * win_bytes);
            }
            issued = s + 1;
        }
    };
    // look-ahead: a sender reserves up to prefetch_steps future windows (only when free) and issues their reads
    auto look_ahead = [&](uint32_t s,
                          bool sender,
                          uint32_t& issued,
                          uint32_t cb,
                          uint32_t win_tiles,
                          uint32_t slot,
                          uint32_t cap,
                          auto&& moves,
                          auto&& issue,
                          uint32_t base,
                          uint32_t win_bytes) {
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
            set_trid(trid_of(issued));
            issue(issued, base + t_slot * win_bytes);
            ++issued;
        }
    };
    auto look_ahead_a = [&](uint32_t s) {
        look_ahead(
            s,
            a_sender,
            a_issued,
            cb_act_k_window,
            a_window_tiles,
            a_slot,
            a_cap_windows,
            a_moves,
            issue_a,
            a_base,
            a_win_bytes);
    };
    auto look_ahead_w = [&](uint32_t s) {
        look_ahead(
            s,
            w_sender,
            w_issued,
            cb_weight_k_window,
            w_window_tiles,
            w_slot,
            w_cap_windows,
            w_moves,
            issue_w,
            w_base,
            w_win_bytes);
    };
    auto look_ahead_both = [&](uint32_t s) {
        look_ahead(
            s,
            a_sender,
            a_issued,
            cb_act_k_window,
            a_window_tiles,
            a_slot,
            a_cap_windows,
            a_moves,
            issue_a,
            a_base,
            a_win_bytes);
        look_ahead(
            s,
            w_sender,
            w_issued,
            cb_weight_k_window,
            w_window_tiles,
            w_slot,
            w_cap_windows,
            w_moves,
            issue_w,
            w_base,
            w_win_bytes);
    };
    // ---- entry0_fill experiment ----------------------------------------------------------------------------------
    // Split-phase receive (MRS_FILL_SPLIT): the helper's ReceiverPipe::receive() fuses "ack my free slot to the
    // sender" with "wait for its data", so a core walking a -> a2 -> b -> b2 cannot ack b before it has RECEIVED a and
    // a2, and b's sender (which waits all its line's acks) inherits the slowest A line's delivery: the four
    // multicasts of a step serialize grid-wide. Raw-API bypass of ReceiverPipe::receive (no split-phase helper API):
    // ack = remote +1 on the sender's consumer-ready cell, wait = poll my data-ready flag VALID then clear it --
    // exactly the helper's two halves (Flag signal, pre-handshake, fixed sender), only reordered.
    auto ack_one = [&](bool receives, const auto& mc) {
        if (receives) {
            rx_ack(mc, noc);
        }
    };
    auto ack_received = [&](uint32_t s, uint32_t mode) {
        if (a_moves(s)) {
            ack_one(a_rx.has_value(), mc_a);
            if constexpr (mc_a2.active) {
                ack_one(a2_rx.has_value(), mc_a2);
            }
        }
        if (mode != 5 && w_moves(s)) {
            ack_one(w_rx.has_value(), mc_b);
            if constexpr (mc_b2.active) {
                ack_one(w2_rx.has_value(), mc_b2);
            }
        }
    };
    auto load_step = [&](uint32_t s) {
        // Every earlier multicast has left L1 before this step issues any read: a slot read into at step s (for a
        // step t <= s + cap - 1) was last a multicast source at step t - cap <= s - 1.
        noc_async_writes_flushed();
        poll_arrivals();
        const bool split = MRS_FILL_SPLIT != 0 && s / num_k_blocks < MRS_FILL_SPLIT_ENTRIES;
        const uint32_t mode = s < MRS_FILL_FULL_STEPS ? 2 : MRS_FILL_SPLIT;
        const bool la_late = split && MRS_FILL_LA_AFTER_SEND;
        {
            MaybeDeviceZoneScope("RD_RESV");
            reserve_current(
                s,
                a_sender,
                a_moves(s),
                a_issued,
                cb_act_k_window,
                a_window_tiles,
                a_slot,
                issue_a,
                a_base,
                a_win_bytes);
            reserve_current(
                s,
                w_sender,
                w_moves(s),
                w_issued,
                cb_weight_k_window,
                w_window_tiles,
                w_slot,
                issue_w,
                w_base,
                w_win_bytes);
            if (split && mode >= 2) {
                ack_received(
                    s, mode);  // split-phase: my reserved slots are free -> tell every sender now, before my reads
            }
            if constexpr (!lookahead_after_barrier) {
                if (!la_late) {
                    look_ahead_both(s);
                }
            }
            if ((a_sender && a_moves(s)) || (w_sender && w_moves(s))) {
                MaybeDeviceZoneScope("RD_BARRIER");
                noc_async_read_barrier_with_trid(trid_of(s), 0);  // this step's reads only
                if constexpr (two_read_nocs) {
                    noc_async_read_barrier_with_trid(trid_of(s), 1);
                }
            }
            if constexpr (lookahead_after_barrier) {
                // the current step's reads get the DRAM to themselves (the fill: every sender reads step 0 at once);
                // the look-ahead reads then overlap this step's multicast
                if (!la_late) {
                    if (MRS_FILL_LA0_W_LATE && s == 0) {
                        look_ahead_a(s);
                    } else {
                        look_ahead_both(s);
                    }
                }
            }
        }
        {
            MaybeDeviceZoneScope("RD_MCAST");
            const uint32_t a_dst = a_base + a_slot * a_win_bytes;
            const uint32_t w_dst = w_base + w_slot * w_win_bytes;
            // CallerManaged: send() returns once the multicast is issued; the source slot is protected by the flush at
            // the start of the next step (no read targets it before then).
#ifndef MRS_SKIP_MCAST                                       // perf ablation only
            const bool zs = fill_zones && s < num_k_blocks;  // instrument entry 0's steps only (marker budget)
            if (split && mode == 5) {
                // early-acked A parts: wait only; W parts: the baseline fused receive (ack after my A landed). With
                // la_late, each operand's look-ahead reads are issued right after this core's own send of that operand
                // (the read issue loop is ~1.6 us of NCRISC time for a W part: off the multicast's critical path)
                if (a_moves(s)) {
                    if (a_tx) {
                        FZ("SND_A",
                           (a_tx->template send<SourceL1Guard::CallerManaged>(
                               a_dst + a_part_off[0], a_dst + a_part_off[0], a_part_bytes[0])));
                    }
                    if constexpr (mc_a2.active) {
                        if (a2_tx) {
                            FZ("SND_A2",
                               (a2_tx->template send<SourceL1Guard::CallerManaged>(
                                   a_dst + a_part_off[1], a_dst + a_part_off[1], a_part_bytes[1])));
                        }
                    }
                    if (la_late) {
                        look_ahead_a(s);
                    }
                    if (a_rx) {
                        FZ("RCV_A", rx_wait(mc_a, poll_arrivals));
                    }
                    if constexpr (mc_a2.active) {
                        if (a2_rx) {
                            FZ("RCV_A2", rx_wait(mc_a2, poll_arrivals));
                        }
                    }
                } else if (la_late) {
                    look_ahead_a(s);
                }
                if (w_moves(s)) {
                    if (w_tx) {
                        FZ("SND_B",
                           (w_tx->template send<SourceL1Guard::CallerManaged>(
                               w_dst + w_part_off[0], w_dst + w_part_off[0], w_part_bytes[0])));
                        if (la_late) {
                            look_ahead_w(s);
                        }
                    } else {
                        FZ("RCV_B", (rx_ack(mc_b, noc), rx_wait(mc_b, poll_arrivals)));
                    }
                    if constexpr (mc_b2.active) {
                        if (w2_tx) {
                            FZ("SND_B2",
                               (w2_tx->template send<SourceL1Guard::CallerManaged>(
                                   w_dst + w_part_off[1], w_dst + w_part_off[1], w_part_bytes[1])));
                            if (la_late) {
                                look_ahead_w(s);
                            }
                        } else {
                            FZ("RCV_B2", (rx_ack(mc_b2, noc), rx_wait(mc_b2, poll_arrivals)));
                        }
                    }
                } else if (la_late) {
                    look_ahead_w(s);
                }
            } else if (split && mode == 3) {
                // early acks (posted after reserve), then the baseline family order; a received part is a wait only
                if (a_moves(s)) {
                    if (a_tx) {
                        FZ("SND_A",
                           (a_tx->template send<SourceL1Guard::CallerManaged>(
                               a_dst + a_part_off[0], a_dst + a_part_off[0], a_part_bytes[0])));
                    } else {
                        FZ("RCV_A", rx_wait(mc_a, poll_arrivals));
                    }
                    if constexpr (mc_a2.active) {
                        if (a2_tx) {
                            FZ("SND_A2",
                               (a2_tx->template send<SourceL1Guard::CallerManaged>(
                                   a_dst + a_part_off[1], a_dst + a_part_off[1], a_part_bytes[1])));
                        } else {
                            FZ("RCV_A2", rx_wait(mc_a2, poll_arrivals));
                        }
                    }
                }
                if (la_late) {
                    FZ("RD_LA", look_ahead_both(s));
                }
                if (w_moves(s)) {
                    if (w_tx) {
                        FZ("SND_B",
                           (w_tx->template send<SourceL1Guard::CallerManaged>(
                               w_dst + w_part_off[0], w_dst + w_part_off[0], w_part_bytes[0])));
                    } else {
                        FZ("RCV_B", rx_wait(mc_b, poll_arrivals));
                    }
                    if constexpr (mc_b2.active) {
                        if (w2_tx) {
                            FZ("SND_B2",
                               (w2_tx->template send<SourceL1Guard::CallerManaged>(
                                   w_dst + w_part_off[1], w_dst + w_part_off[1], w_part_bytes[1])));
                        } else {
                            FZ("RCV_B2", rx_wait(mc_b2, poll_arrivals));
                        }
                    }
                }
            } else if (split && mode == 4) {
                // early acks, then per operand: my sends first, then my waits. A completes on this core before its W
                // sends (keeps the grid's A column multicasts and W row multicasts mostly apart in time), but part 1 of
                // a line no longer waits for part 0's delivery
                if (a_moves(s)) {
                    if (a_tx) {
                        FZ("SND_A",
                           (a_tx->template send<SourceL1Guard::CallerManaged>(
                               a_dst + a_part_off[0], a_dst + a_part_off[0], a_part_bytes[0])));
                    }
                    if constexpr (mc_a2.active) {
                        if (a2_tx) {
                            FZ("SND_A2",
                               (a2_tx->template send<SourceL1Guard::CallerManaged>(
                                   a_dst + a_part_off[1], a_dst + a_part_off[1], a_part_bytes[1])));
                        }
                    }
                    if (a_rx) {
                        FZ("RCV_A", rx_wait(mc_a, poll_arrivals));
                    }
                    if constexpr (mc_a2.active) {
                        if (a2_rx) {
                            FZ("RCV_A2", rx_wait(mc_a2, poll_arrivals));
                        }
                    }
                }
                if (w_moves(s)) {
                    if (w_tx) {
                        FZ("SND_B",
                           (w_tx->template send<SourceL1Guard::CallerManaged>(
                               w_dst + w_part_off[0], w_dst + w_part_off[0], w_part_bytes[0])));
                    }
                    if constexpr (mc_b2.active) {
                        if (w2_tx) {
                            FZ("SND_B2",
                               (w2_tx->template send<SourceL1Guard::CallerManaged>(
                                   w_dst + w_part_off[1], w_dst + w_part_off[1], w_part_bytes[1])));
                        }
                    }
                }
                if (la_late) {
                    FZ("RD_LA", look_ahead_both(s));
                }
                if (w_moves(s)) {
                    if (w_rx) {
                        FZ("RCV_B", rx_wait(mc_b, poll_arrivals));
                    }
                    if constexpr (mc_b2.active) {
                        if (w2_rx) {
                            FZ("RCV_B2", rx_wait(mc_b2, poll_arrivals));
                        }
                    }
                }
            } else if (split) {
                // split-phase walk: (acks already posted) send every part I own, then wait every part I receive, so no
                // line's multicast waits behind another line's delivery (the a -> a2 -> b -> b2 chain)
                if (mode == 1) {
                    ack_received(s, mode);
                }
                if (a_moves(s)) {
                    if (a_tx) {
                        FZ("SND_A",
                           (a_tx->template send<SourceL1Guard::CallerManaged>(
                               a_dst + a_part_off[0], a_dst + a_part_off[0], a_part_bytes[0])));
                    }
                    if constexpr (mc_a2.active) {
                        if (a2_tx) {
                            FZ("SND_A2",
                               (a2_tx->template send<SourceL1Guard::CallerManaged>(
                                   a_dst + a_part_off[1], a_dst + a_part_off[1], a_part_bytes[1])));
                        }
                    }
                }
                if (w_moves(s)) {
                    if (w_tx) {
                        FZ("SND_B",
                           (w_tx->template send<SourceL1Guard::CallerManaged>(
                               w_dst + w_part_off[0], w_dst + w_part_off[0], w_part_bytes[0])));
                    }
                    if constexpr (mc_b2.active) {
                        if (w2_tx) {
                            FZ("SND_B2",
                               (w2_tx->template send<SourceL1Guard::CallerManaged>(
                                   w_dst + w_part_off[1], w_dst + w_part_off[1], w_part_bytes[1])));
                        }
                    }
                }
                if (la_late) {
                    FZ("RD_LA", look_ahead_both(s));
                }
                if (a_moves(s)) {
                    if (a_rx) {
                        FZ("RCV_A", rx_wait(mc_a, poll_arrivals));
                    }
                    if constexpr (mc_a2.active) {
                        if (a2_rx) {
                            FZ("RCV_A2", rx_wait(mc_a2, poll_arrivals));
                        }
                    }
                }
                if (w_moves(s)) {
                    if (w_rx) {
                        FZ("RCV_B", rx_wait(mc_b, poll_arrivals));
                    }
                    if constexpr (mc_b2.active) {
                        if (w2_rx) {
                            FZ("RCV_B2", rx_wait(mc_b2, poll_arrivals));
                        }
                    }
                }
            } else {
                if (a_moves(s)) {
                    if (a_tx) {
                        FZ("SND_A",
                           (a_tx->template send<SourceL1Guard::CallerManaged>(
                               a_dst + a_part_off[0], a_dst + a_part_off[0], a_part_bytes[0])));
                    } else {
                        FZ("RCV_A", a_rx->receive());
                    }
                    if constexpr (mc_a2.active) {
                        if (a2_tx) {
                            FZ("SND_A2",
                               (a2_tx->template send<SourceL1Guard::CallerManaged>(
                                   a_dst + a_part_off[1], a_dst + a_part_off[1], a_part_bytes[1])));
                        } else {
                            FZ("RCV_A2", a2_rx->receive());
                        }
                    }
                }
                if (w_moves(s)) {
                    if (w_tx) {
                        FZ("SND_B",
                           (w_tx->template send<SourceL1Guard::CallerManaged>(
                               w_dst + w_part_off[0], w_dst + w_part_off[0], w_part_bytes[0])));
                    } else {
                        FZ("RCV_B", w_rx->receive());
                    }
                    if constexpr (mc_b2.active) {
                        if (w2_tx) {
                            FZ("SND_B2",
                               (w2_tx->template send<SourceL1Guard::CallerManaged>(
                                   w_dst + w_part_off[1], w_dst + w_part_off[1], w_part_bytes[1])));
                        } else {
                            FZ("RCV_B2", w2_rx->receive());
                        }
                    }
                }
            }
#endif
        }
        if (MRS_FILL_LA0_W_LATE && lookahead_after_barrier && !la_late && s == 0) {
            look_ahead_w(s);
        }
        cb_push_back(cb_act_k_window, a_window_tiles);
        cb_push_back(cb_weight_k_window, w_window_tiles);
        a_slot = a_slot + 1 == a_cap_windows ? 0 : a_slot + 1;
        w_slot = w_slot + 1 == w_cap_windows ? 0 : w_slot + 1;
    };

    // land_arrivals_block: the zone slot(s) of this entry are credited once the counter covers them.
    auto land_arrivals_block = [&](uint32_t entry_idx) {
        const uint32_t word0 = get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * entry_idx);
        const uint32_t word1 = get_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * entry_idx + 1);
        const uint32_t mask = (word0 >> 16) & 0x3;
        MaybeDeviceZoneScope("ARR_WAIT");
        if (mask & 1) {
            noc_semaphore_wait_min(arr_fwd, word1 & 0xFFFF);
            cb_reserve_back(cb_arrivals_fwd, P);
            cb_push_back(cb_arrivals_fwd, P);
        }
        if (mask & 2) {
            noc_semaphore_wait_min(arr_bwd, word1 >> 16);
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
        if constexpr (MRS_FILL_POLL_ARR) {
            continue;
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

    if constexpr (MRS_FILL_POLL_ARR) {
        MaybeDeviceZoneScope("ARR_WAIT");
        while (land_next < num_entries) {
            try_land();
        }
    }
    noc_async_write_barrier();  // the last multicasts have landed before this kernel retires
    // Re-arm the global arrival counters for the next call (decrement what arrived, never store 0).
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
