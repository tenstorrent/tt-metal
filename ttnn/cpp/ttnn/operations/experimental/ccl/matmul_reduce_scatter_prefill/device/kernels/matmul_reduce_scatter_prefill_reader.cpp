// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter_prefill — matmul-core reader (NCRISC, NoC0).
//
// Per entry (transport order): load_operand_block — Kt/kb K-block windows of A (shared along an n-line) and W (shared
// along an m-line), each delivered by Mcast1D families. A window may be split into two contiguous parts (A: by m
// rows, W: by k rows), each read from DRAM and multicast by a different sender of the line (families "a"/"a2",
// "b"/"b2"), so one RISC never serializes a whole window's read issue + multicast. The cross-block operand is
// resident after entry 0: its windows are re-credited (reserve + push, no NoC) because the CB holds the whole Kt
// slice and the ring wraps exactly. Loading is software-pipelined: a sender issues step s+1's DRAM reads while step
// s's multicast drains (send<CallerManaged>; every slot is flushed before it is overwritten).
// Arrivals are landed by POLLING (try_land): at every step start and inside every wait of the loader, every entry
// (in entry order) whose global arrival counter(s) already cover it (host threshold) gets its zone slot credited (the
// upstream port's fabric writes already landed the bytes). The loader never blocks on the fabric, so a late arrival
// no longer stalls this core's next-entry multicasts (and, through the handshakes, its lines). A final blocking drain
// follows the last load. At the end: re-arm both counters by decrementing what arrived.
// Chunks: the walk above runs once per output chunk (num_chunks x the per-chunk entries); a chunk moves the operand
// slices at its tile origin, and the resident operand reloads only when its own axis' chunk index changes. Zone slots
// are reused across chunks, so for every chunk the upstream must wait for, this core reports once its compute consumed
// that chunk's arrivals (CB pages acked) to the local port that forwards the credit upstream.
//
// Multicast walk: the A parts' receivers ack (consumer-ready) right after reserving their slot, before this core's own
// DRAM reads/barrier; an A receive is then a wait only. The W parts keep the fused ack + wait in the a, a2, b, b2
// order, so every W multicast still starts after its whole line received A (grid-wide A-then-W phases): concurrent
// multicasts on one line or crossing A-column / W-row multicasts stall in the NoC (measured up to 16 us).
//
// Raw-API notes (design "Helpers considered and rejected"): try_land is a semaphore poll + CB credit of bytes a remote
// chip wrote; no helper covers a fabric fused-inc counter with host thresholds. rx_ack / rx_wait are the two halves
// of ReceiverPipe::receive (Flag signal, pre-handshake, fixed sender) issued separately: the helper fuses them, and
// the fused ack made every A sender wait for the slowest W-reading receiver's DRAM barrier (the fill chain), and a
// fused wait cannot poll the arrival counters.
// RAW mcast-receive halves — measured helper bypass (do NOT "fix" back to ReceiverPipe::receive):
// focus 640x2048x7168 traced: helper receive 126.1-126.6 us -> split ack/wait + polled
// arrivals 123.1-124.0 us (first window lands 12.8 -> 10.9 us; entry-2 stall gone); GLM -5.0, MiMo -2.3, Linear
// -7.7..-11 us. Keep W's ack fused after A (concurrent A-column / W-row multicasts stalled in the NoC, +16..+32 us).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_args.hpp"

using namespace dataflow_kernel_lib;
namespace mrs = matmul_reduce_scatter_prefill;

// a family's semaphore, read from its compile-time block the same way McastArgs binds it
template <uint32_t CT_BASE, mcast_wire::SemaphoreRole ROLE>
FORCE_INLINE auto mc_semaphore() {
    return detail::make_mcast_semaphore<McastSemaphoreBinding{detail::positional_mcast_semaphore<CT_BASE, ROLE>()}>();
}

// split-phase halves of ReceiverPipe::receive (Flag signal, pre-handshake, fixed sender); no-ops for an absent family
// and for one this kernel never receives on (every core of each line is then its sender; the host omits the sender
// coordinates)
template <uint32_t CT_BASE, uint32_t RT_BASE>
FORCE_INLINE void rx_ack(const McastArgs<CT_BASE, RT_BASE>& mc, const Noc& noc) {
    using MC = McastArgs<CT_BASE, RT_BASE>;
    if constexpr (MC::active) {
        if constexpr (MC::receiver_available) {
            static_assert(MC::pre_handshake && MC::signal == DataReadySignal::Flag && !MC::rotating);
            mc_semaphore<CT_BASE, mcast_wire::CONSUMER_READY>().up(noc, mc.sender_x(), mc.sender_y(), 1);
        }
    }
}
template <uint32_t CT_BASE, uint32_t RT_BASE, typename Poll>
FORCE_INLINE void rx_wait(const McastArgs<CT_BASE, RT_BASE>&, Poll&& poll) {
    using MC = McastArgs<CT_BASE, RT_BASE>;
    if constexpr (MC::active) {
        if constexpr (MC::receiver_available) {
            auto sem = mc_semaphore<CT_BASE, mcast_wire::DATA_READY>();
            while (sem.value() != VALID) {
                poll();
            }
            sem.set(INVALID);
        }
    }
}

void kernel_main() {
    namespace ct = mrs::reader_ct;
    constexpr uint32_t cb_act_k_window = get_compile_time_arg_val(ct::CB_ACT_K_WINDOW);
    constexpr uint32_t cb_weight_k_window = get_compile_time_arg_val(ct::CB_WEIGHT_K_WINDOW);
    constexpr uint32_t cb_arrivals_fwd = get_compile_time_arg_val(ct::CB_ARRIVALS_FWD);
    constexpr uint32_t cb_arrivals_bwd = get_compile_time_arg_val(ct::CB_ARRIVALS_BWD);
    // streamed windows of one wave: A core_m x kb, W kb x core_n (with waves, the split axis is a wave's extent; the
    // resident operand's axis is never split)
    constexpr uint32_t core_m = get_compile_time_arg_val(ct::CORE_M);
    constexpr uint32_t core_n = get_compile_time_arg_val(ct::CORE_N);
    constexpr uint32_t kb = get_compile_time_arg_val(ct::KB);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(ct::NUM_K_BLOCKS);
    constexpr uint32_t Kt = get_compile_time_arg_val(ct::KT);
    constexpr uint32_t Nt = get_compile_time_arg_val(ct::NT);
    constexpr uint32_t a_row_per_block = get_compile_time_arg_val(ct::A_ROW_PER_BLOCK);  // scatter -2: block_m, else 0
    constexpr uint32_t w_col_per_block = get_compile_time_arg_val(ct::W_COL_PER_BLOCK);  // scatter -1: block_n, else 0
    constexpr bool resident_a = get_compile_time_arg_val(ct::RESIDENT_A) != 0;
    constexpr bool resident_w = get_compile_time_arg_val(ct::RESIDENT_W) != 0;
    constexpr uint32_t P = get_compile_time_arg_val(ct::P);
    constexpr uint32_t num_entries = get_compile_time_arg_val(ct::NUM_ENTRIES);  // G x num_waves
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(ct::A_TILE_BYTES);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(ct::W_TILE_BYTES);
    constexpr uint32_t a_cap_windows = get_compile_time_arg_val(ct::A_CAP_WINDOWS);  // CB capacity in K-block windows
    constexpr uint32_t w_cap_windows = get_compile_time_arg_val(ct::W_CAP_WINDOWS);
    constexpr uint32_t a_split = get_compile_time_arg_val(ct::A_SPLIT);  // m rows of part 0 (core_m: one part)
    constexpr uint32_t w_split = get_compile_time_arg_val(ct::W_SPLIT);  // k rows of part 0 (kb: one part)
    constexpr uint32_t prefetch_steps = get_compile_time_arg_val(ct::PREFETCH_STEPS);  // windows a sender reads ahead
    constexpr bool lookahead_after_barrier = get_compile_time_arg_val(ct::LOOKAHEAD_AFTER_BARRIER) != 0;  // issue
    // look-ahead reads after the current step's read barrier (else before it)
    // NoC of each operand's DRAM reads. This kernel owns every read on both NoCs of its core (the writer only writes),
    // so the second NoC adds DRAM-response links for the fill
    constexpr uint8_t a_read_noc = get_compile_time_arg_val(ct::A_READ_NOC);
    constexpr uint8_t w_read_noc = get_compile_time_arg_val(ct::W_READ_NOC);
    constexpr bool two_read_nocs = a_read_noc != w_read_noc || a_read_noc != noc_index;
    constexpr uint32_t num_waves = get_compile_time_arg_val(ct::NUM_WAVES);  // entries per block (e is wave e mod W)
    constexpr uint32_t num_chunks = get_compile_time_arg_val(ct::NUM_CHUNKS);
    constexpr uint32_t n_inner = get_compile_time_arg_val(ct::N_INNER);   // chunks along the inner axis
    constexpr bool m_outer = get_compile_time_arg_val(ct::M_OUTER) != 0;  // chunk c: (c / n_inner, c % n_inner)
    constexpr uint32_t chunk_m = get_compile_time_arg_val(ct::CHUNK_M);
    constexpr uint32_t chunk_n = get_compile_time_arg_val(ct::CHUNK_N);
    constexpr uint32_t block_m = get_compile_time_arg_val(ct::BLOCK_M);
    constexpr uint32_t block_n = get_compile_time_arg_val(ct::BLOCK_N);
    constexpr auto a_args = TensorAccessorArgs<ct::COUNT>();
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

    const uint32_t a_addr = get_common_arg_val<uint32_t>(mrs::reader_common::A_ADDR);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(mrs::reader_common::W_ADDR);
    const uint32_t row0 = get_arg_val<uint32_t>(mrs::reader_rt::ROW0);  // gm * core_m (chunk-local)
    const uint32_t col0 = get_arg_val<uint32_t>(mrs::reader_rt::COL0);  // gn * core_n (chunk-local)
    // my half's row and my (half, link)'s report row in the common args
    const uint32_t half_row = get_arg_val<uint32_t>(mrs::reader_rt::HALF_ROW);
    const uint32_t report_row = get_arg_val<uint32_t>(mrs::reader_rt::REPORT_ROW);
    const uint32_t arr_fwd_addr = get_common_arg_val<uint32_t>(mrs::reader_common::ARR_FWD_ADDR);
    const uint32_t arr_bwd_addr = get_common_arg_val<uint32_t>(mrs::reader_common::ARR_BWD_ADDR);
    // arrival increments my counters receive per chunk
    const uint32_t total_fwd = get_common_arg_val<uint32_t>(half_row + mrs::reader_half::TOTAL_FWD);
    const uint32_t total_bwd = get_common_arg_val<uint32_t>(half_row + mrs::reader_half::TOTAL_BWD);
    // arrival slots I consume per chunk, per direction
    const uint32_t slots_fwd = get_common_arg_val<uint32_t>(half_row + mrs::reader_half::SLOTS_FWD);
    const uint32_t slots_bwd = get_common_arg_val<uint32_t>(half_row + mrs::reader_half::SLOTS_BWD);
    // port counting my consumed fwd / bwd chunks
    const uint32_t rep_fwd_xy = get_common_arg_val<uint32_t>(report_row + mrs::reader_report::FWD_XY);
    const uint32_t rep_bwd_xy = get_common_arg_val<uint32_t>(report_row + mrs::reader_report::BWD_XY);
    // chunks the upstream waits for (num - zone sets)
    const uint32_t report_chunks = get_common_arg_val<uint32_t>(mrs::reader_common::REPORT_CHUNKS);
    // per wave (common): row_off | col_off << 8 | rows << 16 | cols << 24 inside my per-core block
    constexpr uint32_t waves_idx = mrs::reader_common::COUNT;
    // per entry: [block | arrival_mask << 16 | lag << 20, thr_fwd | thr_bwd << 16]; an arrival has landed once my
    // counter of its direction reaches its threshold (one fused increment per (core, unit), in the upstream's order)
    const uint32_t entries_idx = half_row + mrs::reader_half::COUNT;  // in my half's row
    constexpr uint32_t ENTRY_WORDS = mrs::reader_half::ENTRY_WORDS;
    constexpr uint32_t total_entries = num_chunks * num_entries;
    // chunk c's tile origin inside the block and my valid extents there
    auto chunk_mn = [](uint32_t c, uint32_t& im, uint32_t& in) {
        im = m_outer ? c / n_inner : c % n_inner;
        in = m_outer ? c % n_inner : c / n_inner;
    };
    // (the wave_valid below caps it at a wave's extent)
    auto valid_ext = [](uint32_t blk, uint32_t chunk, uint32_t ci, uint32_t off) {
        const uint32_t c0 = ci * chunk;
        const uint32_t ext = blk - c0 < chunk ? blk - c0 : chunk;  // this chunk's extent
        return ext <= off ? 0u : ext - off;
    };

    const auto a_acc = TensorAccessor(a_args, a_addr, a_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
    volatile tt_l1_ptr uint32_t* arr_fwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_fwd_addr);
    volatile tt_l1_ptr uint32_t* arr_bwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_bwd_addr);
    // try_land: credit, in entry order, every entry whose arrival counter(s) already cover it; never blocks
    // consumption reports: chunks of my arrivals compute has popped (16-bit CB ack counter, accumulated as deltas)
    const uint32_t freed_addr = get_semaphore(mrs::sem::FREED);
    uint32_t reported[2] = {0, 0};
    uint16_t acked_seen[2] = {
        (uint16_t)reg_read((uintptr_t)get_cb_tiles_acked_ptr(cb_arrivals_fwd)),
        (uint16_t)reg_read((uintptr_t)get_cb_tiles_acked_ptr(cb_arrivals_bwd))};
    uint32_t acked_total[2] = {0, 0};
    auto try_report = [&]() {
        for (uint32_t d = 0; d < 2; ++d) {
            const uint32_t per_chunk = (d == 0 ? slots_fwd : slots_bwd) * P;
            if (per_chunk == 0 || reported[d] >= report_chunks) {
                continue;
            }
            const uint16_t now =
                (uint16_t)reg_read((uintptr_t)get_cb_tiles_acked_ptr(d == 0 ? cb_arrivals_fwd : cb_arrivals_bwd));
            acked_total[d] += (uint16_t)(now - acked_seen[d]);
            acked_seen[d] = now;
            const uint32_t xy = d == 0 ? rep_fwd_xy : rep_bwd_xy;
            while (reported[d] < report_chunks && acked_total[d] >= (reported[d] + 1) * per_chunk) {
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, freed_addr), 1);
                ++reported[d];
            }
        }
    };
    uint32_t land_next = 0;
    auto try_land = [&]() {
        if (report_chunks > 0) {
            try_report();
        }
        while (land_next < total_entries) {
            const uint32_t le = land_next % num_entries;
            const uint32_t lc = land_next / num_entries;
            const uint32_t word0 = get_common_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * le);
            const uint32_t word1 = get_common_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * le + 1);
            const uint32_t mask = (word0 >> 16) & 0x3;
            invalidate_l1_cache();
            if ((mask & 1) &&
                (*arr_fwd < lc * total_fwd + (word1 & 0xFFFF) || !cb_pages_reservable_at_back(cb_arrivals_fwd, P))) {
                return;
            }
            if ((mask & 2) &&
                (*arr_bwd < lc * total_bwd + (word1 >> 16) || !cb_pages_reservable_at_back(cb_arrivals_bwd, P))) {
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

    Noc noc;
    auto a_tx = mc_a.optional_sender(noc);
    // receiver pipes: constructed for their data-ready flag init; the receive halves are issued by rx_ack / rx_wait
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
    constexpr uint32_t num_steps = total_entries * num_k_blocks;
    auto block_of = [&](uint32_t step) {
        return get_common_arg_val<uint32_t>(entries_idx + ENTRY_WORDS * ((step / num_k_blocks) % num_entries)) & 0xFFFF;
    };
    auto wave_of = [&](uint32_t step) {
        return get_common_arg_val<uint32_t>(waves_idx + ((step / num_k_blocks) % num_entries) % num_waves);
    };
    // valid extent of a wave along one axis: my valid tiles past the wave's offset, capped at its extent
    auto wave_valid = [](uint32_t valid, uint32_t off, uint32_t ext) {
        return valid <= off ? 0u : (valid - off < ext ? valid - off : ext);
    };
    // the resident operand moves only in the first entry of a chunk whose outer (resident-axis) index changed; later
    // entries re-credit the same windows
    auto reloads = [&](uint32_t step) {
        const uint32_t eg = step / num_k_blocks;
        return eg % num_entries == 0 && (eg / num_entries) % n_inner == 0;
    };
    auto a_moves = [&](uint32_t step) { return !resident_a || reloads(step); };
    auto w_moves = [&](uint32_t step) { return !resident_w || reloads(step); };
    // DRAM reads of the parts this core sends, into window `dst`
    auto issue_a = [&](uint32_t step, uint32_t dst) {
        const uint32_t wave = wave_of(step);
        const uint32_t row_off = wave & 0xFF;
        uint32_t im, in;
        chunk_mn(step / num_k_blocks / num_entries, im, in);
        const uint32_t m_valid = valid_ext(block_m, chunk_m, im, row0);
        const uint32_t m_valid_w = wave_valid(m_valid, row_off, (wave >> 16) & 0xFF);
        const uint32_t page0 =
            (block_of(step) * a_row_per_block + im * chunk_m + row0 + row_off) * Kt + (step % num_k_blocks) * kb;
        for (uint32_t part = 0; part < 2; ++part) {
            if (!a_sends[part]) {
                continue;
            }
            const uint32_t i0 = part == 0 ? 0 : a_split;
            const uint32_t i1_nom = part == 0 ? a_split : core_m;
            const uint32_t i1 = i1_nom < m_valid_w ? i1_nom : m_valid_w;
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
    };
    auto issue_w = [&](uint32_t step, uint32_t dst) {
        const uint32_t wave = wave_of(step);
        const uint32_t col_off = (wave >> 8) & 0xFF;
        uint32_t im, in;
        chunk_mn(step / num_k_blocks / num_entries, im, in);
        const uint32_t n_valid = valid_ext(block_n, chunk_n, in, col0);
        const uint32_t n_valid_w = wave_valid(n_valid, col_off, wave >> 24);
        const uint32_t page0 =
            (step % num_k_blocks) * kb * Nt + block_of(step) * w_col_per_block + in * chunk_n + col0 + col_off;
        for (uint32_t part = 0; part < 2; ++part) {
            if (!w_sends[part]) {
                continue;
            }
            const uint32_t k0 = part == 0 ? 0 : w_split;
            const uint32_t k1 = part == 0 ? w_split : kb;
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
            while (!cb_pages_reservable_at_back(cb, win_tiles)) {
                try_land();  // compute may be waiting for an arrival credit before it frees this window
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
    auto load_step = [&](uint32_t s) {
        // Every earlier multicast has left L1 before this step issues any read: a slot read into at step s (for a
        // step t <= s + cap - 1) was last a multicast source at step t - cap <= s - 1.
        noc_async_writes_flushed();
        try_land();
        reserve_current(
            s, a_sender, a_moves(s), a_issued, cb_act_k_window, a_window_tiles, a_slot, issue_a, a_base, a_win_bytes);
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
        if (a_moves(s)) {
            // my A slot is reserved (free): ack the A senders now, before my own reads land, so no A multicast
            // waits for a W reader's DRAM barrier
            if (a_rx) {
                rx_ack(mc_a, noc);
            }
            if constexpr (mc_a2.active) {
                if (a2_rx) {
                    rx_ack(mc_a2, noc);
                }
            }
        }
        if constexpr (!lookahead_after_barrier) {
            look_ahead_both(s);
        }
        if ((a_sender && a_moves(s)) || (w_sender && w_moves(s))) {
            noc_async_read_barrier_with_trid(trid_of(s), 0);  // this step's reads only
            if constexpr (two_read_nocs) {
                noc_async_read_barrier_with_trid(trid_of(s), 1);
            }
        }
        if constexpr (lookahead_after_barrier) {
            // the current step's reads get the DRAM to themselves (the fill: every sender reads step 0 at once);
            // the look-ahead reads then overlap this step's multicast
            look_ahead_both(s);
        }
        const uint32_t a_dst = a_base + a_slot * a_win_bytes;
        const uint32_t w_dst = w_base + w_slot * w_win_bytes;
        // CallerManaged: send() returns once the multicast is issued; the source slot is protected by the flush at
        // the start of the next step (no read targets it before then).
        if (a_moves(s)) {
            if (a_tx) {
                a_tx->template send<SourceL1Guard::CallerManaged>(
                    a_dst + a_part_off[0], a_dst + a_part_off[0], a_part_bytes[0]);
            } else {
                rx_wait(mc_a, try_land);  // acked after reserve
            }
            if constexpr (mc_a2.active) {
                if (a2_tx) {
                    a2_tx->template send<SourceL1Guard::CallerManaged>(
                        a_dst + a_part_off[1], a_dst + a_part_off[1], a_part_bytes[1]);
                } else {
                    rx_wait(mc_a2, try_land);
                }
            }
        }
        if (w_moves(s)) {
            if (w_tx) {
                w_tx->template send<SourceL1Guard::CallerManaged>(
                    w_dst + w_part_off[0], w_dst + w_part_off[0], w_part_bytes[0]);
            } else {
                rx_ack(mc_b, noc);  // fused order: W is acked only once my A landed (A-then-W phases)
                rx_wait(mc_b, try_land);
            }
            if constexpr (mc_b2.active) {
                if (w2_tx) {
                    w2_tx->template send<SourceL1Guard::CallerManaged>(
                        w_dst + w_part_off[1], w_dst + w_part_off[1], w_part_bytes[1]);
                } else {
                    rx_ack(mc_b2, noc);
                    rx_wait(mc_b2, try_land);
                }
            }
        }
        cb_push_back(cb_act_k_window, a_window_tiles);
        cb_push_back(cb_weight_k_window, w_window_tiles);
        a_slot = a_slot + 1 == a_cap_windows ? 0 : a_slot + 1;
        w_slot = w_slot + 1 == w_cap_windows ? 0 : w_slot + 1;
    };

    for (uint32_t step = 0; step < num_steps; ++step) {
        load_step(step);
    }
    while (land_next < total_entries) {
        try_land();
    }
    while (reported[0] < (slots_fwd ? report_chunks : 0) || reported[1] < (slots_bwd ? report_chunks : 0)) {
        try_report();  // the upstream ports wait for these credits
    }

    noc_async_write_barrier();  // the last multicasts have landed before this kernel retires
    // Re-arm the global arrival counters for the next call (decrement what arrived, never store 0).
    if (total_fwd > 0) {
        noc_semaphore_wait_min(arr_fwd, num_chunks * total_fwd);
        noc_semaphore_inc(get_noc_addr(arr_fwd_addr), 0u - num_chunks * total_fwd);
    }
    if (total_bwd > 0) {
        noc_semaphore_wait_min(arr_bwd, num_chunks * total_bwd);
        noc_semaphore_inc(get_noc_addr(arr_bwd_addr), 0u - num_chunks * total_bwd);
    }
    noc_async_atomic_barrier();
}
