// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch v3 planner (BRISC, planner cores): v2's plan part (kvd_planner.cpp), but every other needed row
// goes out as class-0 packets: the planner fills a staging slot of each TX port (d, l) in its own L1 (entries, rows
// read from the cache) and publishes each slot as it fills (at most `stage` a port waiting for TX), then the partial
// ones, then tells every TX it is done with class 0. Planner 0 is also the chip barrier: every core of the op counts in
// on its barrier word after zeroing its control block, and it releases them all.
//
// A slice (inbox slot i) is chip i's plan-table slot for this owner: plan_table header, then 5 bitmap words per local
// block (160 rows / 32). Row g of local block lb, word w, bit k: g = (me + lb * N) * stripe + w * 32 + k.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#include "kvd3_common.hpp"

#if defined(KVD_ABL_COPY_LATE) && !defined(KVD_ABL_NO_SUMMARY)
#error "KVD_ABL=COPY_LATE uses the planner scratch the summaries use: add NO_SUMMARY"
#endif

#ifdef KVD_ABL_FIRST_TILE
#error "KVD_ABL=FIRST_TILE was removed (measured noise; plan 10.13)"
#endif

#include <cstdint>

using namespace kvd;
using namespace kvd3;

namespace {
constexpr uint8_t kXRing[] = {KVD_XRING};
constexpr uint8_t kYRing[] = {KVD_YRING};
}  // namespace

#ifdef KVD_FUSED_PLAN
// the fused plan (kvd4): this kernel runs the chip's dispatch-only plan first (sparse_push_plan.cpp, PLAN_FUSED)
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/plan/device/kernels/sparse_push_plan.cpp"
#endif

void kernel_main() {
#ifdef KVD_FUSED_PLAN
    plan_prologue();
#endif
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr bool WX = get_compile_time_arg_val(3) != 0;
    constexpr bool WY = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t lanes = get_compile_time_arg_val(5);  // lanes (links x lanes per link)
    constexpr uint32_t stripe = get_compile_time_arg_val(6);
    constexpr uint32_t owner_slot_rows = get_compile_time_arg_val(7);  // output rows per owner (owner-major pages)
    constexpr uint32_t inbox_slot_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t planners = get_compile_time_arg_val(10);
    constexpr uint32_t stage = get_compile_time_arg_val(11);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(12);  // cache pages per slot
    constexpr bool has_batch = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t num_slots = get_compile_time_arg_val(14);
    constexpr uint32_t owner_blocks_max = get_compile_time_arg_val(15);
    constexpr uint32_t cb_inbox = get_compile_time_arg_val(16);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(17);
    constexpr uint32_t cb_row = get_compile_time_arg_val(18);
    constexpr uint32_t G = get_compile_time_arg_val(19);
    constexpr uint32_t B = get_compile_time_arg_val(20);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(21);
    constexpr uint32_t n_cores = get_compile_time_arg_val(22);
    constexpr uint32_t lanes3 = get_compile_time_arg_val(23);  // 1, or 2: producers split by index parity
    constexpr auto input_args = TensorAccessorArgs<24>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto inbox_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto scratch_args = TensorAccessorArgs<inbox_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<scratch_args.next_compile_time_args_offset()>();
    constexpr auto batch_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t words_per_block = stripe / 32;
    constexpr uint32_t bitmap_off = ::sparse_push::plan_table::kBitmapOff;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t inbox_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t scratch_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t batch_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(5);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(6);
    const uint32_t host_slot = get_common_arg_val<uint32_t>(7);

    const uint32_t t_k0 = stats_clock();
    uint32_t t_first_pub = 0;
    uint32_t a = 0;
    const uint32_t planner = get_arg_val<uint32_t>(a++);
    const uint32_t lane3 = lanes3 == 2 ? (planner & 1) : 0;  // (P even: planner index = producer index)
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t my_r = me / C, my_c = me % C;
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(a++)));
    const uint32_t ports_present = get_arg_val<uint32_t>(a++);  // bit q = d * lanes + l: TX(d, l) exists
    constexpr uint32_t kQ = 4 * kMaxLinks;
    uint32_t tx_x[kQ], tx_y[kQ];
    for (uint32_t q = 0; q < kQ; ++q) {
        tx_x[q] = get_arg_val<uint32_t>(a++);
        tx_y[q] = get_arg_val<uint32_t>(a++);
    }
    uint32_t rd_x[kQ], rd_y[kQ];  // this planner's lane reader of TX(d, l): the TX core (lane 0) or its helper
    for (uint32_t q = 0; q < kQ; ++q) {
        rd_x[q] = get_arg_val<uint32_t>(a++);
        rd_y[q] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t is_sync = get_arg_val<uint32_t>(a++);  // planner 0: owns the inbox word and the chip barrier
    const uint32_t bar_x = get_arg_val<uint32_t>(a++);
    const uint32_t bar_y = get_arg_val<uint32_t>(a++);
    const uint32_t go_sem_addr = reinterpret_cast<uint32_t>(go);
    const uint32_t planner_xy_arg = a;  // planners x (noc x, noc y), read when needed (no stack copy)
    a += 2 * planners;
    const uint32_t all_xy_arg = a;  // planner 0: every core of the op (noc x, noc y), the barrier's release
    const uint32_t table_addr = get_common_arg_val<uint32_t>(12);
#ifdef KVD_ACTIVE_FROM_BLOCK
    if (skip_chunk(table_addr, align64(get_write_ptr(cb_stage)))) {  // (stage: written only after the gate)
        return;
    }
#endif
    const uint32_t inbox_sem_addr = get_common_arg_val<uint32_t>(11);
    const uint32_t bgo_addr = get_common_arg_val<uint32_t>(9);       // arrived_semaphore: the release
    const uint32_t barrier_addr = get_common_arg_val<uint32_t>(10);  // credit_semaphore on planner 0: the count

    // ---- chip barrier: zero the control block, count in; planner 0 releases everyone once all are in
    const uint32_t ctrl = get_write_ptr(cb_ctrl);
    auto* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl);
#ifdef KVD_ABL_ZERO_LOOP
    for (uint32_t i = 0; i < kCtrlWords; ++i) {
        cw[i] = 0;
    }
#else
    zero_ctrl(ctrl);
#endif
    noc_semaphore_inc(get_noc_addr(bar_x, bar_y, barrier_addr), 1);
    noc_async_atomic_barrier();
    if (is_sync) {
        auto* count_in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);
        noc_semaphore_wait_min(count_in, n_cores);
        noc_semaphore_set(count_in, 0);
        for (uint32_t i = 0; i < n_cores; ++i) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(all_xy_arg + 2 * i), get_arg_val<uint32_t>(all_xy_arg + 2 * i + 1), bgo_addr),
                1);
        }
        noc_async_atomic_barrier();
    }
    {
        auto* bgo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bgo_addr);
        noc_semaphore_wait_min(bgo, 1);
        noc_semaphore_set(bgo, 0);
    }
    const uint32_t t_bar = stats_clock();
    uint32_t t_own = 0, t_inbox = 0;

    const auto input = TensorAccessor(input_args, input_addr, row_bytes);
    const auto output = TensorAccessor(output_args, output_addr, row_bytes);
    const auto inbox = TensorAccessor(inbox_args, inbox_addr, inbox_slot_bytes);
    (void)scratch_addr;

    // The cache slot: the host's, or (trace-safe) the user id read on device * num_layers + layer_idx.
    cb_reserve_back(cb_row, 1);
    const uint32_t row_l1 = align64(get_write_ptr(cb_row));
    uint32_t slot = host_slot;
    bool bad_user = false;
    if constexpr (has_batch) {
        const auto batch = TensorAccessor(batch_args, batch_addr);
        noc_async_read(batch.get_noc_addr(0), row_l1, sizeof(uint32_t));
        noc_async_read_barrier();
        const uint32_t user = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);
        bad_user = user >= num_slots / num_layers;
        slot = bad_user ? 0 : user * num_layers + layer_idx;
    }
    const uint32_t slot_base = slot * pages_per_slot;

    const uint32_t t_p0 = stats_clock();
    // Planner 0 copies this chip's own slice into the inbox, waits until the N - 1 other chips' slices are in (their
    // last packets increment its inbox word), takes the word back to zero (persistent), and releases every planner.
    if (is_sync) {
#if defined(KVD_FUSED_PLAN) && !defined(KVD_ABL_FUSED_NOPLAN)
        // (the fused plan: every owner's slot is in the table)
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(KVD_PLAN_READY)), N);
#endif
        cb_reserve_back(cb_inbox, 1);
        const uint32_t buf = align64(get_write_ptr(cb_inbox));
        const auto table = TensorAccessor(table_args, table_addr, inbox_slot_bytes);
        noc_async_read(table.get_noc_addr(0), buf, 64);  // slot 0's header holds this call's extent
        noc_async_read_barrier();
        const auto* h0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        const uint32_t active = h0[::sparse_push::plan_table::kWordActiveBlocks];
        // (and the chunk start: a dense plan writes only slot 0's header; the planners' hot split reads it)
        const uint32_t has_split = h0[::sparse_push::plan_table::kWordHasSplit];
        const uint32_t split_block = h0[::sparse_push::plan_table::kWordSplitBlock];
        noc_async_read(table.get_noc_addr(me), buf, inbox_slot_bytes);
        noc_async_read_barrier();
        auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        h[::sparse_push::plan_table::kWordActiveBlocks] = active;
        h[::sparse_push::plan_table::kWordHasSplit] = has_split;
        h[::sparse_push::plan_table::kWordSplitBlock] = split_block;
        noc_async_write(buf, inbox.get_noc_addr(me), inbox_slot_bytes);
        noc_async_write_barrier();
        t_own = stats_clock();
        auto* inbox_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inbox_sem_addr);
        noc_semaphore_wait_min(inbox_sem, N - 1);
        t_inbox = stats_clock();
        noc_semaphore_inc(get_noc_addr(inbox_sem_addr), 0u - (N - 1));
        for (uint32_t p = 0; p < planners; ++p) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(planner_xy_arg + 2 * p),
                    get_arg_val<uint32_t>(planner_xy_arg + 2 * p + 1),
                    go_sem_addr),
                1);
        }
        noc_async_atomic_barrier();
    }
    {
        KVD_ZONE("KVD-P-gowait");
        noc_semaphore_wait_min(go, 1);
    }
    noc_semaphore_set(go, 0);
    const uint32_t t_go = stats_clock();
    KVD_ZONE("KVD-P-plan");

    // Every requester's plan covers its own active blocks (the plan op derives them from that chip's highest requested
    // row), so they differ: a requester's words past its own count are stale and read as zero. The planners split this
    // chip's blocks up to the largest count.
    cb_reserve_back(cb_inbox, 1);
    const uint32_t hdr_l1 = align64(get_write_ptr(cb_inbox));
    // Each slice's nonzero-word summary (the plan's owner-major slots: one bit a bitmap word), read with the headers:
    // a planner then reads and loads only the (word, slice) lines with bits, ~5% of them (KVD_ABL_NO_SUMMARY: all).
    // Control words 256.. are scratch on a planner core (no remote writes): summaries, then the per-word slice masks.
    const uint32_t sum_l1 = align64(ctrl + 1024);
    const uint32_t wm_l1 = sum_l1 + N * ::sparse_push::plan_table::kSummaryWords * 4;
    constexpr uint32_t kMaxMaskWords = 512;
    static_assert(1024 + 63 + 32 * ::sparse_push::plan_table::kSummaryWords * 4 + kMaxMaskWords * 4 <= 2048 * 4);
    for (uint32_t i = 0; i < N; ++i) {
        noc_async_read(inbox.get_noc_addr(i), hdr_l1 + i * 64, 64);
#ifndef KVD_ABL_NO_SUMMARY
        noc_async_read(
            inbox.get_noc_addr(i) + ::sparse_push::plan_table::kSummaryOff * 4,
            sum_l1 + i * ::sparse_push::plan_table::kSummaryWords * 4,
            ::sparse_push::plan_table::kSummaryWords * 4);
#endif
    }
    noc_async_read_barrier();
    const uint32_t t_hdr = stats_clock();
    uint32_t owned_of[32];
    uint32_t owned = 0;
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t act = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            hdr_l1 + i * 64)[::sparse_push::plan_table::kWordActiveBlocks];
        owned_of[i] = act > me ? (act - me + N - 1) / N : 0;
        owned_of[i] =
            owned_of[i] < owner_blocks_max ? owned_of[i] : owner_blocks_max;  // a header never overruns a slot
        if (bad_user) {
            owned_of[i] = 0;  // an out-of-range user id: plan nothing rather than read another user's cache
        }
        owned = owned_of[i] > owned ? owned_of[i] : owned;
    }
    uint32_t owned_words_of[32];
    for (uint32_t i = 0; i < N; ++i) {
        owned_words_of[i] = owned_of[i] * words_per_block;
    }
#ifdef KVD_ABL_NO_SUMMARY
    bool use_sum = false;
#else
    bool use_sum = true;
    for (uint32_t i = 0; i < N; ++i) {
        use_sum = use_sum && (owned_of[i] == 0 || reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                                      hdr_l1 + i * 64)[::sparse_push::plan_table::kWordHasSummary] ==
                                                      ::sparse_push::plan_table::kSummaryMagic);
    }
#endif
    // Blocks are dealt to the planners in turn (lb = planner, planner + P, ...): the rows the current chunk's queries
    // pick cluster in a few blocks, so contiguous ranges would leave one planner with most of the work.
    // Slice i's words of the j-th block land at inbox_l1 + (j * N + i) * 128 (one 128 B window covers a block's 5
    // words from its 64 B aligned start).
    const uint32_t inbox_l1 = hdr_l1 + N * 64;
    // Words (32 rows each) are dealt to the planners in turn (u = planner, planner + P, ...): the current chunk's block
    // of this owner is needed by nearly every chip, so even whole blocks leave one planner with most of the work.
    // Slice i's 64 B line holding the j-th word lands at inbox_l1 + (j * N + i) * 64.
    const uint32_t owned_words = owned * words_per_block;
    // Dealt in groups of kGW words that share a 64 B line of every slice (absolute word index a = bitmap_off + u;
    // group g = a / kGW): one read a (group, slice) instead of one a (word, slice), and still fine-grained enough that
    // the hot blocks spread over many planners. Slice i's line of the j-th group lands at inbox_l1 + (j * N + i) * 64.
#ifdef KVD_ABL_PDEAL
    constexpr uint32_t kGW = KVD_ABL_PDEAL;
#else
    constexpr uint32_t kGW = 2;  // (L1 inbox: 2 beats 1 by ~5-10 us; 4/8/16 lose to planner imbalance)
#endif
    static_assert(kGW >= 1 && kGW <= 16 && (16 % kGW) == 0, "a group stays inside one 64 B line");
    // Knobs: KVD_ABL=PSPLIT=S fixes the cold groups' split (default adaptive), HSPLIT=S the hot groups' (default 8);
    // both 1 = whole groups. With splits, PFLUSH counts units and the KVD_STATS word counts count a word once a part.
    // Units: each group is split into kSplit parts of 32 / kSplit rows of every word, dealt to the planners in turn
    // (unit v = (g - g_lo) * kSplit + part): a hot word (rows nearly every chip wants, e.g. the current chunk's block,
    // or every word at small KV) spreads over kSplit planners instead of one.
#ifdef KVD_ABL_PSPLIT
    constexpr uint32_t kSplitFixed = KVD_ABL_PSPLIT;  // (0: adaptive, below)
#else
    constexpr uint32_t kSplitFixed = 0;
#endif
    static_assert(kSplitFixed <= 32 && (kSplitFixed == 0 || (32 % kSplitFixed) == 0));
    // Hot groups (KVD_ABL_HSPLIT=S): the groups from the current chunk's first block on (every chip's queries pick
    // their local window there, so its rows are needed nearly everywhere and are the slowest planners' work) are split
    // into S parts each; the others into kSplit. At small KV every block is the current chunk's.
#ifdef KVD_ABL_HSPLIT
    constexpr uint32_t kHSplitMax = KVD_ABL_HSPLIT;
#else
    constexpr uint32_t kHSplitMax = 8;  // (5k: 164 -> 108 us; 1M unchanged)
#endif
    static_assert(kHSplitMax >= 1 && kHSplitMax <= 32 && (32 % kHSplitMax) == 0);
    const uint32_t a_lo = bitmap_off, a_hi = bitmap_off + owned_words;
    const uint32_t g_lo = a_lo / kGW, g_hi = (a_hi + kGW - 1) / kGW;
    uint32_t g_hot = g_hi;
    {
        const auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_l1 + me * 64);
        if (h[::sparse_push::plan_table::kWordHasSplit] != 0) {
            const uint32_t sb = h[::sparse_push::plan_table::kWordSplitBlock];
            const uint32_t lb_hot = sb > me ? (sb - me + N - 1) / N : 0;
            const uint32_t a_hot = a_lo + lb_hot * words_per_block;
            g_hot = a_hot / kGW < g_lo ? g_lo : (a_hot / kGW > g_hi ? g_hi : a_hot / kGW);
        }
    }
    // Cold groups split only while there are few of them (small KV): up to 4 ways, at most ~2 units a planner
    // (50k L14: 166 -> 135 us with 4; at 1M a split costs line reads and loses).
    uint32_t cold_split = kSplitFixed, hot_split = kHSplitMax;
    if (cold_split == 0) {
        const uint32_t cold = g_hot - g_lo;
        cold_split = cold * 4 <= 2 * planners ? 4 : (cold * 2 <= 2 * planners ? 2 : 1);
    }
    // (a planner's line sets are KVD3_PLANNER_UNITS at most: past that, no split)
    if (((g_hot - g_lo) * cold_split + (g_hi - g_hot) * hot_split + planners - 1) / planners > KVD3_PLANNER_UNITS) {
        cold_split = hot_split = 1;
    }
    const uint32_t n_cold = (g_hot - g_lo) * cold_split;
    const uint32_t n_units = n_cold + (g_hi - g_hot) * hot_split;
    // unit v -> its group and the rows of each of its words it routes
    auto unit_group = [&](uint32_t v) { return v < n_cold ? g_lo + v / cold_split : g_hot + (v - n_cold) / hot_split; };
    auto unit_rows = [&](uint32_t v) -> uint32_t {
        const uint32_t s = v < n_cold ? cold_split : hot_split;
        const uint32_t part = v < n_cold ? v % cold_split : (v - n_cold) % hot_split;
        const uint32_t n = 32 / s;
        return s == 1 ? ~0u : ((1u << n) - 1) << (part * n);
    };
    if (planner < n_units && ((n_units - 1 - planner) / planners + 1) * kGW > kMaxMaskWords) {
        use_sum = false;  // (more words than the mask scratch: all lines)
    }
    static_assert(32 % kGW == 0 && ::sparse_push::plan_table::kBitmapOff % kGW == 0);
    if (use_sum) {
        // Group-outer: a group's per-word slice masks from the summaries (plain loads: they landed before the barrier
        // above), stored once; then a read only for each marked (group, slice) line.
        uint64_t page[32];
        for (uint32_t i = 0; i < N; ++i) {
            page[i] = inbox.get_noc_addr(i);
        }
        const tt_l1_ptr uint32_t* sums = reinterpret_cast<const tt_l1_ptr uint32_t*>(sum_l1);
        auto* wm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wm_l1);
        constexpr uint32_t kGMask = (1u << kGW) - 1;
        uint32_t jj = 0;
        for (uint32_t v = planner; v < n_units; v += planners, ++jj) {
            const uint32_t g = unit_group(v);
            const uint32_t first_u = g * kGW - a_lo;  // (a_lo % kGW == 0: a group never straddles a summary word)
            const uint32_t w32 = first_u / 32, sh = first_u % 32;
            uint32_t m[kGW];
            for (uint32_t b = 0; b < kGW; ++b) {
                m[b] = 0;
            }
#pragma GCC unroll 8
            for (uint32_t i = 0; i < N; ++i) {
                const uint32_t own = owned_words_of[i];
                const uint32_t valid =
                    own > first_u ? (own - first_u >= kGW ? kGMask : (1u << (own - first_u)) - 1) : 0;
                const uint32_t bits = (sums[i * ::sparse_push::plan_table::kSummaryWords + w32] >> sh) & valid;
                for (uint32_t b = 0; b < kGW; ++b) {
                    m[b] |= ((bits >> b) & 1) << i;
                }
            }
            uint32_t any = 0;
            for (uint32_t b = 0; b < kGW; ++b) {
                wm[jj * kGW + b] = m[b];
                any |= m[b];
            }
            for (; any != 0; any &= any - 1) {
                const uint32_t i = __builtin_ctz(any);
                noc_async_read(page[i] + ((g * kGW * 4) & ~63u), inbox_l1 + (jj * N + i) * 64, 64);
            }
        }
    } else {
        // Slice-outer: slice i is one page (one bank), so its NoC coordinate is set once and each of its lines is a
        // stateful read (~4x cheaper to issue than a full read; the planners' start waits on these).
        for (uint32_t ii = 0; ii < N; ++ii) {
            const uint32_t i = (ii + planner) % N;  // (staggered: the planners start on different banks)
            const uint64_t page = inbox.get_noc_addr(i);
            noc_async_read_one_packet_set_state(page, 64);
            const uint32_t page_lo = static_cast<uint32_t>(page);
            uint32_t jj = 0;
            for (uint32_t v = planner; v < n_units; v += planners, ++jj) {
                const uint32_t g = unit_group(v);
                const uint32_t first_u = (g * kGW > a_lo ? g * kGW : a_lo) - a_lo;
                if (owned_words_of[i] <= first_u) {
                    continue;
                }
                noc_async_read_one_packet_with_state(page_lo + ((g * kGW * 4) & ~63u), inbox_l1 + (jj * N + i) * 64);
            }
        }
    }
    const uint32_t t_issued = stats_clock();
    noc_async_read_barrier();

    const uint32_t t_read = stats_clock();
    // Class-0 staging: one open slot per port q = d * lanes + l (lazily: a slot is taken when its first row comes),
    // at most `stage` per port taken (open or waiting for TX). A full slot: rows read, barrier, publish.
    constexpr uint32_t sbytes = slot_bytes(B, row_bytes);
    constexpr uint32_t rows_off = slot_rows_offset(B);
    const uint32_t stage_l1 = align64(get_write_ptr(cb_stage));
    uint32_t pubq[kQ], openq[kQ];
    for (uint32_t q = 0; q < kQ; ++q) {
        pubq[q] = 0;
        openq[q] = 0;
    }
    auto qslot = [&](uint32_t q) { return stage_l1 + stage_offset(q, 0, pubq[q] % stage, 1, stage, sbytes); };
    auto publish = [&](uint32_t q) {  // close the open slot of q and publish it (its reads must have landed)
        auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(qslot(q));
        h[kHdrEntries] = openq[q];
        h[kHdrClass] = 0;
        h[kHdrEnd] = 0;
        openq[q] = 0;
        ++pubq[q];
        if (t_first_pub == 0) {
            t_first_pub = stats_clock();
        }
        noc_semaphore_inc(get_noc_addr(rd_x[q], rd_y[q], ctrl + tx_pub(planner, 0) * 4), 1);
        noc_semaphore_inc(get_noc_addr(rd_x[q], rd_y[q], ctrl + tx_cls_pub(lane3, 0) * 4), 1);
        noc_semaphore_inc(get_noc_addr(rd_x[q], rd_y[q], ctrl + tx_pub_total(lane3) * 4), 1);
    };
    uint32_t cy_wait = 0, cy_rdbar = 0, n_rows = 0;
    uint32_t cy_waitq[kQ], n_waitq[kQ], n_rowsq[kQ];  // (KVD_STATS: per port, to tell head-of-line stalls)
    for (uint32_t q = 0; q < kQ; ++q) {
        cy_waitq[q] = n_waitq[q] = n_rowsq[q] = 0;
    }
    // a row goes into the open slot of port q (one is taken if needed: TX has freed one)
    auto place_row = [&](uint32_t q, uint32_t w0, uint32_t m, uint64_t src) {
        ++n_rows;
        if (openq[q] == 0) {  // a new slot: wait until TX has freed one
            const uint32_t tw = stats_clock();
            bool waited = false;
            while (pubq[q] - cw[p_freed(q)] >= stage) {
                waited = true;
            }
            const uint32_t dw = stats_clock() - tw;
            cy_wait += dw;
#ifdef KVD_STATS
            ++n_rowsq[q];  // (KVDW: rows, stalls and their time per port; the time includes uncontended checks)
            n_waitq[q] += waited;
            cy_waitq[q] += dw;
#endif
            (void)waited;
        }
#ifdef KVD_STATS
        else {
            ++n_rowsq[q];
        }
#endif
        const uint32_t dst = qslot(q);
        auto* de = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + kSlotHeaderBytes);
        de[2 * openq[q]] = w0;
        de[2 * openq[q] + 1] = m;
        noc_async_read(src, dst + rows_off + openq[q] * row_bytes, row_bytes);
        if (++openq[q] == B) {
            const uint32_t tb = stats_clock();
            noc_async_read_barrier();
            cy_rdbar += stats_clock() - tb;
            publish(q);
        }
    };
#ifdef KVD_ABL_PDEFER
    // Deferred rows (KVD_ABL=PDEFER): a row for a port with no free slot waits in that port's list (in the row CB,
    // after the copy batch) while the planner routes on; a list is placed as its port frees slots, and only a full list
    // makes the planner wait. A list is first in, first out (entries stand alone, so order is free).
    constexpr uint32_t kDefer = 32;  // rows a port
    const uint32_t defer_l1 = row_l1 + 8 * row_bytes;
    uint32_t dh[kQ], dt[kQ];
    for (uint32_t q = 0; q < kQ; ++q) {
        dh[q] = dt[q] = 0;
    }
    uint32_t n_deferred = 0, any_deferred = 0;  // any_deferred: bit q, list q not empty
    auto dslot = [&](uint32_t q, uint32_t i) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(defer_l1 + (q * kDefer + i % kDefer) * 16);
    };
    auto room = [&](uint32_t q) { return openq[q] != 0 || pubq[q] - cw[p_freed(q)] < stage; };
    auto drain = [&](uint32_t q) {
        while (dh[q] != dt[q] && room(q)) {
            auto* e = dslot(q, dh[q]++);
            place_row(q, e[0], e[1], (static_cast<uint64_t>(e[3]) << 32) | e[2]);
        }
        if (dh[q] == dt[q]) {
            any_deferred &= ~(1u << q);
        }
    };
    auto add_row = [&](uint32_t q, uint32_t w0, uint32_t m, uint64_t src) {
        if (any_deferred >> q & 1) {
            drain(q);
        }
        if (dh[q] == dt[q] && room(q)) {
            place_row(q, w0, m, src);
            return;
        }
        while (dt[q] - dh[q] >= kDefer) {  // the list is full: wait for this port (as without PDEFER)
            place_row(
                q,
                dslot(q, dh[q])[0],
                dslot(q, dh[q])[1],
                (static_cast<uint64_t>(dslot(q, dh[q])[3]) << 32) | dslot(q, dh[q])[2]);
            ++dh[q];
        }
        auto* e = dslot(q, dt[q]++);
        e[0] = w0;
        e[1] = m;
        e[2] = static_cast<uint32_t>(src);
        e[3] = static_cast<uint32_t>(src >> 32);
        any_deferred |= 1u << q;
        ++n_deferred;
    };
    auto drain_all = [&]() {
        for (uint32_t m = any_deferred; m != 0; m &= m - 1) {
            drain(__builtin_ctz(m));
        }
    };
#else
    auto add_row = place_row;
    auto drain_all = []() {};
#endif

    // The rows this chip itself needs: local copies into its owner-major pages, kCopyBatch at a time.
    constexpr uint32_t kCopyBatch = 8;
    uint32_t copy_rows[kCopyBatch];
    uint32_t copies = 0;
    uint32_t n_words = 0, n_bits = 0, n_copies = 0, cy_copy = 0, cy_load = 0, cy_tree = 0, cy_add = 0;
    auto flush_copies = [&]() {
        const uint32_t tc = stats_clock();
        n_copies += copies;
#ifdef KVD_ABL_NO_COPIES
        copies = 0;
        return;
#endif
        for (uint32_t k = 0; k < copies; ++k) {
            noc_async_read(input.get_noc_addr(slot_base + copy_rows[k]), row_l1 + k * row_bytes, row_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < copies; ++k) {
            noc_async_write(
                row_l1 + k * row_bytes, output.get_noc_addr(me * owner_slot_rows + copy_rows[k]), row_bytes);
        }
        noc_async_writes_flushed();  // the buffers are reused next
        copies = 0;
        cy_copy += stats_clock() - tc;
    };

    const Routing<R, C, WX, WY> rt{kXRing, kYRing};
    const uint32_t me_bit = 1u << me;
#ifdef KVD_ABL_COPY_LATE
    // the rows this chip needs itself, copied after done(0) (control words 256..2047: no remote writes on a planner)
    constexpr uint32_t kLateMax = 2048 - 256;
    auto* late = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl) + 256;
    uint32_t n_late = 0;
#endif
    // Route the rows of word aw; slice i's 64 B line holding the word is at lines + i * 64.
    // part_mask: the rows of the word this unit routes (kSplit)
    auto route_word = [&](uint32_t lines,
                          uint32_t aw,
                          uint32_t smask,
                          uint32_t part_mask) {  // smask: slices with bits (~0: unknown)
        drain_all();
        const uint32_t u = aw - a_lo;
        const uint32_t lb = u / words_per_block, w = u % words_per_block;
        const uint32_t word_off = aw % 16;
        // only the slices with a bit in this word take part in the rows' masks
        uint32_t words[32], nz_of[32];
        uint32_t any = 0, nz = 0;
        ++n_words;
        const uint32_t tl = stats_clock();
        const volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lines) + word_off;
        if (smask != ~0u) {  // only the slices the summaries mark (their lines are the only ones read)
            for (uint32_t m = smask; m != 0; m &= m - 1) {
                const uint32_t i = __builtin_ctz(m);
                const uint32_t v = p[i * 16];
                words[nz] = v;
                nz_of[nz++] = i;
                any |= v;
            }
        } else {
            for (uint32_t i = 0; i < N; ++i, p += 16) {
                const uint32_t v = owned_words_of[i] > u ? *p : 0;
                if (v != 0) {
                    words[nz] = v;
                    nz_of[nz++] = i;
                    any |= v;
                }
            }
        }
        cy_load += stats_clock() - tl;
        any &= part_mask;
        while (any) {
            const uint32_t k = __builtin_ctz(any);
            any &= any - 1;
            ++n_bits;
            uint32_t mask = 0;
            for (uint32_t z = 0; z < nz; ++z) {
                mask |= ((words[z] >> k) & 1) << nz_of[z];
            }
            const uint32_t in_block = w * 32 + k;
            const uint32_t row = (me + lb * N) * stripe + in_block;
            if (mask & me_bit) {
#ifdef KVD_ABL_COPY_LATE
                if (n_late < kLateMax) {
                    late[n_late++] = lb * stripe + in_block;  // after done(0): no traffic waits on this chip's own rows
                } else
#endif
                {
                    copy_rows[copies++] = lb * stripe + in_block;
                    if (copies == kCopyBatch) {
                        flush_copies();
                    }
                }
            }
            const uint32_t m = mask & ~me_bit;
            if (m == 0) {
                continue;
            }
            const uint32_t tt0 = stats_clock();
            const RowTree<R, C, WX, WY> tree(m);
            const uint32_t order = tree.order(rt, my_r, my_c, row);
#ifdef KVD_ABL_ROW_LANE
            const uint32_t lane = lane_of(row, lanes);
#else
            // one link a planner (not a row hash): its class-0 rows fill 4 ports instead of 8, so its packets are ~2x
            // fuller (L14 190 -> 186, L62 496 -> 476); the links stay balanced over the planners
            const uint32_t lane = (planner / lanes3) % lanes;  // (independent of the reader lane, planner % lanes3)
#endif
            const uint32_t tie = hbit(row, kTieBit);
            const uint64_t src = input.get_noc_addr(slot_base + lb * stripe + in_block);  // (once a row)
            const uint32_t tt1 = stats_clock();
            cy_tree += tt1 - tt0;
            for (uint32_t d = 0; d < kDirections; ++d) {
                const uint32_t q = d * lanes + lane;
                if (!(ports_present >> q & 1)) {
                    continue;
                }
                const uint32_t hops = tree.own_hops(rt, my_r, my_c, order, d, tie);
                if (hops == 0) {
                    continue;
                }
                add_row(q, entry_word0(row, hops, 0, order), m, src);
            }
            cy_add += stats_clock() - tt1;
        }
    };
    auto group_words = [&](uint32_t g, uint32_t& aw0, uint32_t& aw1) {
        aw0 = g * kGW > a_lo ? g * kGW : a_lo;
        aw1 = g * kGW + kGW < a_hi ? g * kGW + kGW : a_hi;
    };
    uint32_t j = 0;
#ifdef KVD_ABL_EMPTY_OWN
    for (uint32_t v = n_units; v < n_units; v += planners, ++j) {  // no own rows: the bare protocol
#else
    for (uint32_t v = planner; v < n_units; v += planners, ++j) {
#endif
        const uint32_t g = unit_group(v);
        const uint32_t part_mask = unit_rows(v);
#ifdef KVD_ABL_PFLUSH
        // every KVD_ABL_PFLUSH groups, let the open (partial) slots go: the first packets of sparse ports leave early
        if (j != 0 && j % KVD_ABL_PFLUSH == 0) {
            noc_async_read_barrier();  // (their rows have landed)
            for (uint32_t q = 0; q < kQ; ++q) {
                if ((ports_present >> q & 1) && openq[q] != 0) {
                    publish(q);
                }
            }
        }
#endif
        uint32_t aw0, aw1;
        group_words(g, aw0, aw1);
        for (uint32_t aw = aw0; aw < aw1; ++aw) {
            const uint32_t smask =
                use_sum ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wm_l1)[j * kGW + (aw - aw0)] : ~0u;
            if (smask != 0) {
                route_word(inbox_l1 + j * N * 64, aw, smask, part_mask);
            }
        }
    }
    flush_copies();
    noc_async_write_barrier();
#ifdef KVD_ABL_P_DELAY
    {  // critical-path probe: hold the class-0 partial slots back by KVD_ABL_P_DELAY cycles
        const uint32_t t = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
        while (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L) - t < KVD_ABL_P_DELAY) {
        }
    }
#endif
#ifdef KVD_ABL_PDEFER
    for (uint32_t q = 0; q < kQ; ++q) {  // the deferred rows, waiting for their ports now
        while (dh[q] != dt[q]) {
            auto* e = dslot(q, dh[q]++);
            place_row(q, e[0], e[1], (static_cast<uint64_t>(e[3]) << 32) | e[2]);
        }
    }
#endif
    const uint32_t t_compute = stats_clock();
    // The partial slots, then done(0) to every TX of this chip.
    noc_async_read_barrier();
    for (uint32_t q = 0; q < kQ; ++q) {
        if ((ports_present >> q & 1) && openq[q] != 0) {
            publish(q);
        }
    }
    noc_async_atomic_barrier();  // every publish is in before done(0)
    for (uint32_t q = 0; q < kQ; ++q) {
        if (ports_present >> q & 1) {
            noc_semaphore_inc(get_noc_addr(rd_x[q], rd_y[q], ctrl + tx_done(lane3, 0) * 4), 1);
        }
    }
    noc_async_atomic_barrier();
#ifdef KVD_ABL_COPY_LATE
    for (uint32_t i = 0; i < n_late; ++i) {
        copy_rows[copies++] = late[i];
        if (copies == kCopyBatch) {
            flush_copies();
        }
    }
    flush_copies();
    noc_async_write_barrier();
#endif
    // stay until every published slot is pulled: TX reads this core's staging and frees it in its control block
    for (uint32_t q = 0; q < kQ; ++q) {
        if (ports_present >> q & 1) {
            while (cw[p_freed(q)] != pubq[q]) {
            }
        }
    }
#ifdef KVD_STATS
    DPRINT(
        "KVDP planner={} go={} read={} compute={} publish={} wait={} rdbar={} rows={} at_go={} first_pub={} done={}\n",
        planner,
        t_go - t_p0,
        t_read - t_go,
        t_compute - t_read,
        stats_clock() - t_compute,
        cy_wait,
        cy_rdbar,
        n_rows,
        (t_go - t_k0) / 1350,
        (t_first_pub - t_k0) / 1350,
        (t_compute - t_k0) / 1350);
    DPRINT("KVDQ planner={} words={} bits={} copies={} copy={}\n", planner, n_words, n_bits, n_copies, cy_copy);
    DPRINT("KVDZ planner={} load={} tree={} add={}\n", planner, cy_load, cy_tree, cy_add);
#ifdef KVD_ABL_PDEFER
    DPRINT("KVDD planner={} deferred={}\n", planner, n_deferred);
#endif
    for (uint32_t q = 0; q < kQ; ++q) {
        if (n_rowsq[q] != 0) {
            DPRINT(
                "KVDW planner={} q={} wait={} nwait={} rows={}\n",
                planner,
                q,
                cy_waitq[q] / 1350,
                n_waitq[q],
                n_rowsq[q]);
        }
    }
    DPRINT("KVDR hdr={} issue={} wait={}\n", t_hdr - t_go, t_issued - t_hdr, t_read - t_issued);
    if (is_sync) {
        DPRINT(
            "KVDS0 bar={} own={} inbox={} go={} k0={} (0.1 us from kernel start)\n",
            (t_bar - t_k0) / 135,
            (t_own - t_k0) / 135,
            (t_inbox - t_k0) / 135,
            (t_go - t_k0) / 135,
            t_k0);
    }
#endif
    (void)t_k0;
    (void)t_first_pub;
    (void)cy_wait;
    (void)cy_waitq;
    (void)n_waitq;
    (void)n_rowsq;
    (void)cy_rdbar;
    (void)n_rows;
    (void)n_words;
    (void)n_bits;
    (void)n_copies;
    (void)cy_copy;
    (void)cy_load;
    (void)cy_tree;
    (void)cy_add;
    (void)t_hdr;
    (void)t_issued;
    (void)t_bar;
    (void)t_own;
    (void)t_inbox;
}
