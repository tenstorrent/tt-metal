// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE on-device planner for the multicast-tree push (see sparse_push_tree_plan_device_operation.hpp; host
// reference _tree_plan_records in tests/nightly/blackhole/sdpa/test_sparse_sdpa_gather_overlap.py, which this
// reproduces word for word). Each core owns a slice of rows (bitmap words [w0, w1)):
//   1. count: for every row any requester needs, build its tree from its owner (fewer edges, row parity on ties) and
//      count this chip's rows per group: outgoing (dir, level, keep), incoming (dir, level, keep), own, needed.
//   2. exchange: every core reads every core's counts -> group totals and this core's rank offsets.
//   3. emit: rescan; every row gets its rank in each of its groups, so its record slot and words are known. A
//      packet (or 16-slot line) is written whole, as one 128 B (64 B) write, by the core holding its first row;
//      rows of a packet begun on an earlier core go to that core's carry buffer and it writes them after a barrier.
//   4. rewrite: the requester's top-k indices become compact slots through the per-rank slot table.

#include "api/dataflow/dataflow_api.h"

#include <cstdint>
#if defined(DEBUG_PRINT_ENABLED)
#include "api/debug/dprint.h"
#define PLAN_T(i) stamp[i] = get_timestamp_32b()
#else
#define PLAN_T(i)
#endif

namespace {
constexpr uint32_t kRecordWords = 32;
constexpr uint32_t kHeaderWords = 32;
constexpr uint32_t kLineSlots = 16;  // slot-table entries per 64 B line
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
constexpr uint32_t kE = 0, kW = 1, kN = 2, kS = 3;
enum : uint32_t { kEnd = 0, kSendInput = 1, kSendOutput = 2, kLocal = 3, kLevel = 6 };
enum Cb : uint32_t {
    kBits = 0,
    kCnt = 1,
    kAllCnt = 2,
    kOpen = 3,
    kCarry = 4,
    kScratch = 5,
    kRw = 6,
    kIdx = 7,
    kVars = 8,
    kDens = 9,
    kCache = 10,
    kHandoff = 11
};
constexpr uint32_t kWin = 256;        // bitmap words per requester streamed into L1 at a time
constexpr uint32_t kChunkWords = 16;  // rows are balanced over cores in chunks of 512

inline uint32_t popcount(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}
inline uint32_t ctz(uint32_t x) { return __builtin_ctz(x); }
inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}  // namespace

void kernel_main() {
    constexpr uint32_t n_dev = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t sc = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t links = get_compile_time_arg_val(6);
    constexpr uint32_t k_per = get_compile_time_arg_val(7);
    constexpr uint32_t LV = get_compile_time_arg_val(8);
    constexpr uint32_t stage_pages = get_compile_time_arg_val(9);
    constexpr uint32_t table_pages = get_compile_time_arg_val(10);
    constexpr uint32_t bitmap_off = get_compile_time_arg_val(11);
    constexpr uint32_t counts_off = get_compile_time_arg_val(12);
    constexpr uint32_t owner_off = get_compile_time_arg_val(13);
    constexpr uint32_t W = get_compile_time_arg_val(14);
    constexpr uint32_t wpc = get_compile_time_arg_val(15);
    constexpr uint32_t G = get_compile_time_arg_val(16);
    constexpr uint32_t rows = get_compile_time_arg_val(17);
    constexpr uint32_t topk = get_compile_time_arg_val(18);
    constexpr uint32_t stream_page_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(20);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(21);
    constexpr uint32_t slot_cap = get_compile_time_arg_val(22);
    constexpr uint32_t cnt_words = get_compile_time_arg_val(23);
    constexpr uint32_t max_blocks = get_compile_time_arg_val(24);
    constexpr uint32_t cache_rows = get_compile_time_arg_val(25);
    constexpr uint32_t handoff_sem_id = get_compile_time_arg_val(26);
    constexpr auto tables_args = TensorAccessorArgs<27>();
    constexpr auto idx_args = TensorAccessorArgs<tables_args.next_compile_time_args_offset()>();
    constexpr auto rank_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto streams_args = TensorAccessorArgs<rank_args.next_compile_time_args_offset()>();
    constexpr auto compact_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();
    constexpr auto slots_args = TensorAccessorArgs<compact_args.next_compile_time_args_offset()>();
    constexpr auto uni_args = TensorAccessorArgs<slots_args.next_compile_time_args_offset()>();
    constexpr auto paths_args = TensorAccessorArgs<uni_args.next_compile_time_args_offset()>();

    constexpr uint32_t per_dir = links * k_per;
    constexpr uint32_t senders = 4 * per_dir;
    constexpr uint32_t unit = P * B;
    constexpr uint32_t words_per_block = sc / 32;
    constexpr uint32_t kOut = 0, kIn = 4 * LV * 2, kOwn = 2 * kIn, kNeed = kOwn + 1, kGroups = kNeed + 1;
    static_assert(R * C == n_dev && n_dev <= 32, "rank masks are 32 bit");
    static_assert(cnt_words >= kGroups, "count block too small");
    static_assert(8 + 2 * P <= kRecordWords, "record too small");

    const uint32_t me = get_arg_val<uint32_t>(0);
    [[maybe_unused]] uint32_t stamp[16] = {};
    PLAN_T(0);
    const auto tables = TensorAccessor(tables_args, get_arg_val<uint32_t>(1), 4096);
    const auto indices = TensorAccessor(idx_args, get_arg_val<uint32_t>(2), topk * 4);
    const auto rank_t = TensorAccessor(rank_args, get_arg_val<uint32_t>(3), 64);
    const auto streams = TensorAccessor(streams_args, get_arg_val<uint32_t>(4), stream_page_bytes);
    const auto compact = TensorAccessor(compact_args, get_arg_val<uint32_t>(5), topk * 4);
    const auto slots_t = TensorAccessor(slots_args, get_arg_val<uint32_t>(6), slot_cap * 4);
    const auto uni_t = TensorAccessor(uni_args, get_arg_val<uint32_t>(7), W * 4);
    const auto paths_t = TensorAccessor(paths_args, get_arg_val<uint32_t>(8), 1024);

    // Handoff to NCRISC, which rewrites the indices while BRISC emits records:
    // [0] rank | [1] valid words of this requester | [2] its needed rows | [16, 48) in_of | [48, 48 + 4 LV) lanes
    // before each (arrival dir, hop) group of kept rows.
    auto* handoff = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kHandoff));
    constexpr uint32_t sc_div = sc;  // keeps the helper's division by a constant
    auto lane_rel_h = [](uint32_t j) { return (j % B) + (j / B) * unit; };
#if defined(REWRITE_HELPER)
    {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(handoff_sem_id)), 1);
        invalidate_l1_cache();
        const uint32_t a = handoff[0], vw = handoff[1], need_a = handoff[2];
        uint32_t in_of[32], lnb_in[4 * LV];
        for (uint32_t o = 0; o < n_dev; ++o) {
            in_of[o] = handoff[16 + o];
        }
        for (uint32_t i = 0; i < 4 * LV; ++i) {
            lnb_in[i] = handoff[48 + i];
        }
        const uint32_t region_a = ceil_div(ceil_div(need_a, P) + LV, B) * unit;
        constexpr uint32_t segs = topk % 512 == 0 ? topk / 512 : 1, seg_len = topk / segs;
        const uint32_t units = rows * segs, per_core = ceil_div(units, G);
        const uint32_t u0 = me * per_core < units ? me * per_core : units;
        const uint32_t u1 = u0 + per_core < units ? u0 + per_core : units;
        if (u0 >= u1) {
            return;
        }
        // A needed row's slot follows from its owner: owned here -> the own region; otherwise it arrived from
        // in_of[owner] at that hop, so it sits at its rank among this chip's needed rows of owners with the same
        // (direction, hop) -- a per-block prefix over the requester's bitmap per such class.
        const uint32_t rw_l1 = get_write_ptr(kRw);
        auto* bpre = reinterpret_cast<uint32_t*>(rw_l1);  // [max_blocks] | this requester's bitmap [W]
        auto* abits = bpre + max_blocks;
        const uint32_t blocks = vw / words_per_block;
        for (uint32_t w = 0; w < vw;) {
            const uint32_t word = bitmap_off + w;
            const uint32_t in_page = word % 1024;
            const uint32_t n = vw - w < 1024 - in_page ? vw - w : 1024 - in_page;
            noc_async_read(
                tables.get_noc_addr(a * table_pages + word / 1024) + in_page * 4,
                reinterpret_cast<uint32_t>(abits + w),
                n * 4);
            w += n;
        }
        for (uint32_t b = 0; b < blocks;) {
            const uint32_t word = counts_off + b;
            const uint32_t in_page = word % 1024;
            const uint32_t n = blocks - b < 1024 - in_page ? blocks - b : 1024 - in_page;
            noc_async_read(
                tables.get_noc_addr(a * table_pages + word / 1024) + in_page * 4,
                reinterpret_cast<uint32_t>(bpre + b),
                n * 4);
            b += n;
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        uint32_t cls_run[1 + 4 * LV];  // class 0 = owned here, else 1 + arrival dir * LV + hop
        for (uint32_t c = 0; c < 1 + 4 * LV; ++c) {
            cls_run[c] = 0;
        }
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint32_t o = b % n_dev;
            const uint32_t c = o == a ? 0u : 1u + (in_of[o] & 0xFF) % 4 * LV + (in_of[o] >> 8);
            const uint32_t v = bpre[b];
            bpre[b] = cls_run[c];
            cls_run[c] += v;
        }
        const uint32_t idx_l1 = get_write_ptr(kIdx);
        auto* idx = reinterpret_cast<uint32_t*>(idx_l1);
        for (uint32_t u = u0; u < u1; ++u) {
            const uint64_t seg_off = (u % segs) * seg_len * 4;
            noc_async_write_barrier();
            noc_async_read(indices.get_noc_addr(u / segs) + seg_off, idx_l1, seg_len * 4);
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t k = 0; k < seg_len; ++k) {
                const uint32_t v = idx[k];
                if (v == kSentinel) {
                    continue;
                }
                const uint32_t blk = v / sc_div, w = v >> 5, o = blk % n_dev;
                uint32_t rank = bpre[blk];
                for (uint32_t w2 = blk * words_per_block; w2 < w; ++w2) {
                    rank += popcount(abits[w2]);
                }
                rank += popcount(abits[w] & ((1u << (v & 31)) - 1u));
                if (o == a) {
                    idx[k] = lane_rel_h(rank / P) + (rank % P) * B;
                } else {
                    const uint32_t din = in_of[o] & 0xFF, h = in_of[o] >> 8;
                    idx[k] = (1 + din) * region_a + lane_rel_h(lnb_in[din * LV + h] + rank / P) + (rank % P) * B;
                }
            }
            noc_async_write(idx_l1, compact.get_noc_addr(u / segs) + seg_off, seg_len * 4);
        }
        noc_async_write_barrier();
        return;
    }
#endif

    const uint32_t bits_l1 = get_write_ptr(kBits);
    const uint32_t cnt_l1 = get_write_ptr(kCnt);
    const uint32_t all_l1 = get_write_ptr(kAllCnt);
    const uint32_t open_l1 = get_write_ptr(kOpen);
    const uint32_t carry_l1 = get_write_ptr(kCarry);
    const uint32_t scratch_l1 = get_write_ptr(kScratch);
    // Core-private L1 arrays are plain memory; barrier() is a compiler fence before other cores read them.
    auto* bits = reinterpret_cast<uint32_t*>(bits_l1);
    auto* cnt = reinterpret_cast<uint32_t*>(cnt_l1);
    auto* all = reinterpret_cast<uint32_t*>(all_l1);
    auto* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);

    auto core_noc = [&](uint32_t c, uint32_t addr) {
        return get_noc_addr(get_common_arg_val<uint32_t>(2 * c), get_common_arg_val<uint32_t>(2 * c + 1), addr);
    };
    // Working arrays live in L1 (the RISC-V stack is small).
    uint32_t vars_l1 = get_write_ptr(kVars);
    auto carve = [&](uint32_t words) {
        auto* p = reinterpret_cast<uint32_t*>(vars_l1);
        vars_l1 += (words + 3) / 4 * 16;
        return p;
    };
    auto* arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_sem_id));
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id));
    uint32_t phase = 0;
    auto barrier = [&]() {
        asm volatile("" ::: "memory");
        ++phase;
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        noc_semaphore_inc(core_noc(0, get_semaphore(arrive_sem_id)), 1);
        noc_async_atomic_barrier();
        if (me == 0) {
            noc_semaphore_wait_min(arrive, G * phase);
            for (uint32_t c = 0; c < G; ++c) {
                noc_semaphore_inc(core_noc(c, get_semaphore(go_sem_id)), 1);
            }
            noc_async_atomic_barrier();
        }
        noc_semaphore_wait_min(go, phase);
        invalidate_l1_cache();
    };

    // This chip's rank, and every requester's needed-row count and valid bitmap extent.
    noc_async_read(rank_t.get_noc_addr(0), scratch_l1, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t a = scratch[0];
    const uint32_t ar = a / C, ac = a % C;
    auto* needed_rows = carve(n_dev);
    auto* valid_words = carve(n_dev);
    constexpr uint32_t head_words = (owner_off + n_dev + 15) / 16 * 16;
    for (uint32_t d = 0; d < n_dev; ++d) {
        noc_async_read(tables.get_noc_addr(d * table_pages), all_l1 + d * head_words * 4, head_words * 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t d = 0; d < n_dev; ++d) {
        const uint32_t* head = all + d * head_words;
        valid_words[d] = head[3] * words_per_block;  // header [3]: active blocks; later words may be stale
        uint32_t n = 0;
        for (uint32_t o = 0; o < n_dev; ++o) {
            n += head[owner_off + o];
        }
        needed_rows[d] = n;
    }
    auto region = [&](uint32_t b) { return ceil_div(ceil_div(needed_rows[b], P) + LV, B) * unit; };
    auto base = [&](uint32_t b, uint32_t dr, uint32_t keep) {
        return keep == 0 ? (1 + dr) * region(b) : 5 * region(b) + dr * stage_pages;
    };
    auto lane_rel = [](uint32_t j) { return (j % B) + (j / B) * unit; };
    auto neighbour = [&](uint32_t c, uint32_t dr) {
        const uint32_t r = c / C, k = c % C;
        const uint32_t nr = (r + (dr == kS ? 1 : 0) + (dr == kN ? R - 1 : 0)) % R;
        const uint32_t nk = (k + (dr == kE ? 1 : 0) + (dr == kW ? C - 1 : 0)) % C;
        return nr * C + nk;
    };
    auto stream_of = [](uint32_t dr, uint32_t n) {
        const uint32_t pick = n % per_dir;
        return ((pick / k_per) * 4 + dr) * k_per + pick % k_per;
    };

    // Dimension-ordered paths: owner (ro, co) -> chip d goes along column co to d's row, then along that row, each the
    // short way round; half-way ties go S / E when ro + co is even. Per owner o, from this chip's point of view:
    //   via[o]        requesters whose path passes through this chip (a row of o arrives here if one wants it)
    //   in_of[o]      arrival direction | hop index << 8 (4 = this chip is the owner)
    //   beyond[o][dr] requesters whose path continues to this chip's neighbour in direction dr
    //   depth[o]      hops from the owner to this chip (the level of edges leaving it)
    // Host-built (see _dim_path_tables in the test): [via x32 | in_of x32 | depth x32 | beyond x128] for this chip.
    auto* via = carve(256);
    auto* in_of = via + n_dev;
    auto* depth = in_of + n_dev;
    auto* beyond = depth + n_dev;
    static_assert(n_dev == 32, "path tables hold 32 owners");
    noc_async_read(paths_t.get_noc_addr(a), reinterpret_cast<uint32_t>(via), 1024);
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t nbr[4];
    for (uint32_t dr = 0; dr < 4; ++dr) {
        nbr[dr] = (dr == kE || dr == kW) ? ar * C + (ac + (dr == kE ? 1 : C - 1)) % C
                                         : ((ar + (dr == kS ? 1 : R - 1)) % R) * C + ac;
    }

    // Streams requester bitmaps [ws, we) through L1 window by window (zero past each requester's valid extent).
    auto* col_v = carve(32);
    uint32_t* uni_win = bits + n_dev * kWin;
    auto for_each_word = [&](uint32_t ws, uint32_t we, auto&& fn, bool with_union = false) {
        for (uint32_t base_w = ws; base_w < we; base_w += kWin) {
            const uint32_t end_w = base_w + kWin < we ? base_w + kWin : we;
            if (with_union) {
                noc_async_read(
                    uni_t.get_noc_addr(0) + base_w * 4, reinterpret_cast<uint32_t>(uni_win), (end_w - base_w) * 4);
                noc_async_read_barrier();
                invalidate_l1_cache();
                bool empty = true;
                for (uint32_t w = 0; w < end_w - base_w && empty; ++w) {
                    empty = uni_win[w] == 0;
                }
                if (empty) {
                    continue;
                }
            }
            for (uint32_t d = 0; d < n_dev; ++d) {
                const uint32_t hi = end_w < valid_words[d] ? end_w : valid_words[d];
                for (uint32_t w = hi > base_w ? hi : base_w; w < end_w; ++w) {
                    bits[d * kWin + (w - base_w)] = 0;
                }
                for (uint32_t w = base_w; w < hi;) {
                    const uint32_t word = bitmap_off + w;
                    const uint32_t in_page = word % 1024;
                    const uint32_t n = hi - w < 1024 - in_page ? hi - w : 1024 - in_page;
                    noc_async_read(
                        tables.get_noc_addr(d * table_pages + word / 1024) + in_page * 4,
                        bits_l1 + (d * kWin + w - base_w) * 4,
                        n * 4);
                    w += n;
                }
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t w = base_w; w < end_w; ++w) {
                fn(w, w - base_w);
            }
        }
    };
    // Rows of bitmap word i (owner o) that arrive here, leave in each direction, or that this chip needs.
    auto classify = [&](uint32_t i, uint32_t o, uint32_t& arrive, uint32_t* out, uint32_t& go, uint32_t& need) {
        uint32_t* v = col_v;
        for (uint32_t d = 0; d < n_dev; ++d) {
            v[d] = bits[d * kWin + i];
        }
        auto any_of = [&](uint32_t set) {
            uint32_t r = 0;
            for (; set; set &= set - 1) {
                r |= v[ctz(set)];
            }
            return r;
        };
        arrive = o == a ? 0u : any_of(via[o]);
        go = 0;
        for (uint32_t dr = 0; dr < 4; ++dr) {
            out[dr] = beyond[o * 4 + dr] ? any_of(beyond[o * 4 + dr]) : 0u;
            go |= out[dr];
        }
        need = v[a];
    };
    // Visits every row of [ws, we) that involves this chip with its cache entry (see the count pass). The owner is
    // fixed per bitmap word, so arrival and each forwarding direction are ORs of a few requesters' words.
    auto for_each_entry = [&](uint32_t ws, uint32_t we, auto&& fn) {
        for_each_word(
            ws,
            we,
            [&](uint32_t w, uint32_t i) {
                if (!uni_win[i]) {
                    return;
                }
                const uint32_t blk = (w * 32) / sc, o = blk % n_dev;
                uint32_t arrive, out[4], go, need;
                classify(i, o, arrive, out, go, need);
                const uint32_t own = o == a ? 1u : 0u;
                const uint32_t* v = col_v;
                const uint32_t in_dir = in_of[o] & 0xFF, in_depth = in_of[o] >> 8;
                for (uint32_t u = arrive | go | need; u; u &= u - 1) {
                    const uint32_t j = ctz(u), x = w * 32 + j;
                    const uint32_t local = (blk / n_dev) * sc + x % sc;
                    const bool arrives = arrive >> j & 1;
                    uint32_t outs = 0, n_out = 0;
                    for (uint32_t dr = 0; dr < 4; ++dr) {
                        if (out[dr] >> j & 1) {
                            const uint32_t keep = (v[nbr[dr]] >> j & 1) ? 0u : 1u;
                            outs |= (dr | depth[o] << 2 | keep << 6) << (7 * n_out++);
                        }
                    }
                    fn(local | (need >> j & 1) << 20 | own << 21 | (arrives ? in_dir : 4u) << 22 |
                           (arrives ? in_depth : 0u) << 25 | n_out << 29,
                       outs);
                }
            },
            true);
    };

    PLAN_T(10);
    // 0. Balance: needed rows per 512-row chunk over an even split, then contiguous chunk ranges of equal rows.
    constexpr uint32_t chunks = W / kChunkWords;
    constexpr uint32_t chunks_per_core = wpc / kChunkWords;
    constexpr uint32_t dens_per_core = (chunks_per_core + 15) / 16 * 16;
    static_assert(W % kChunkWords == 0 && wpc % kChunkWords == 0, "chunks of 16 bitmap words");
    const uint32_t dens_l1 = get_write_ptr(kDens);
    auto* dens = reinterpret_cast<uint32_t*>(dens_l1);
    auto* my_dens = dens + G * dens_per_core;
    {
        const uint32_t e0 = me * wpc < W ? me * wpc : W;
        const uint32_t e1 = e0 + wpc < W ? e0 + wpc : W;
        for (uint32_t k = 0; k < dens_per_core; ++k) {
            my_dens[k] = 0;
        }
        for_each_word(e0, e1, [&](uint32_t w, uint32_t i) {
            uint32_t any = 0;
            for (uint32_t d = 0; d < n_dev; ++d) {
                any |= bits[d * kWin + i];
            }
            uni_win[i] = any;
            if (any) {
                uint32_t arrive, out[4], go, need;
                classify(i, ((w * 32) / sc) % n_dev, arrive, out, go, need);
                my_dens[(w - e0) / kChunkWords] += 5 * popcount(arrive | go | need) + 2;  // a row ~ 2.5 word scans
            }
            if (i + 1 == kWin || w + 1 == e1) {
                const uint32_t first = w - i;
                noc_async_write(reinterpret_cast<uint32_t>(uni_win), uni_t.get_noc_addr(0) + first * 4, (i + 1) * 4);
                noc_async_write_barrier();
            }
        });
    }
    PLAN_T(11);
    barrier();
    PLAN_T(12);
    for (uint32_t c = 0; c < G; ++c) {
        noc_async_read(
            core_noc(c, dens_l1 + G * dens_per_core * 4), dens_l1 + c * dens_per_core * 4, dens_per_core * 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    auto chunk_rows = [&](uint32_t k) { return dens[(k / chunks_per_core) * dens_per_core + k % chunks_per_core]; };
    uint32_t total_rows = 0;
    for (uint32_t k = 0; k < chunks; ++k) {
        total_rows += chunk_rows(k);
    }
    // Core c starts at the first chunk with at least c / G of the rows before it.
    uint32_t w0 = W, w1 = W;
    {
        uint32_t before = 0, c = 0;
        for (uint32_t k = 0; k <= chunks && c <= me + 1; ++k) {
            while (c <= me + 1 &&
                   (k == chunks || static_cast<uint64_t>(before) * G >= static_cast<uint64_t>(total_rows) * c)) {
                if (c == me) {
                    w0 = k * kChunkWords;
                }
                if (c == me + 1) {
                    w1 = k * kChunkWords;
                }
                ++c;
            }
            if (k < chunks) {
                before += chunk_rows(k);
            }
        }
        if (me + 1 == G) {
            w1 = W;
        }
    }

    PLAN_T(1);
    // 1. Count. Rows that involve this chip are also cached (2 words each) so the emit pass skips the rest:
    //   [0] local | need << 20 | own << 21 | in_dir << 22 | in_depth << 25 | n_out << 29
    //   [1] per outgoing edge e, 7 bits at 7e: dir | depth << 2 | keep << 6
    [[maybe_unused]] uint32_t visited = 0;
    auto* cache = reinterpret_cast<uint32_t*>(get_write_ptr(kCache));
    uint32_t cached = 0;
    bool cache_full = false;
    for (uint32_t g = 0; g < cnt_words; ++g) {
        cnt[g] = 0;
    }
    for_each_entry(w0, w1, [&](uint32_t e0, uint32_t outs) {
        const uint32_t need = e0 >> 20 & 1, own = e0 >> 21 & 1, in_dir = e0 >> 22 & 7, in_depth = e0 >> 25 & 15;
        const uint32_t n_out = e0 >> 29;
        ++visited;
        cnt[kNeed] += need;
        if (own && need) {
            cnt[kOwn] += 1;
        }
        if (in_dir < 4) {
            cnt[kIn + (in_dir * LV + in_depth) * 2 + (need ? 0 : 1)] += 1;
        }
        for (uint32_t e = 0; e < n_out; ++e) {
            const uint32_t o = outs >> (7 * e);
            cnt[kOut + ((o & 3) * LV + ((o >> 2) & 15)) * 2 + ((o >> 6) & 1)] += 1;
        }
        if (cached == cache_rows) {
            cache_full = true;
            return;
        }
        cache[2 * cached] = e0;
        cache[2 * cached + 1] = outs;
        ++cached;
    });

    PLAN_T(2);
    // 2. Exchange: all[c][g] becomes core c's exclusive prefix; tot[g] the group total.
    barrier();
    PLAN_T(3);
    for (uint32_t c = 0; c < G; ++c) {
        noc_async_read(core_noc(c, cnt_l1), all_l1 + c * cnt_words * 4, cnt_words * 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    auto* tot = carve(kGroups);
    for (uint32_t g = 0; g < kGroups; ++g) {
        uint32_t s = 0;
        for (uint32_t c = 0; c < G; ++c) {
            const uint32_t v = all[c * cnt_words + g];
            all[c * cnt_words + g] = s;
            s += v;
        }
        tot[g] = s;
    }
    auto pre = [&](uint32_t c, uint32_t g) { return all[c * cnt_words + g]; };
    auto owner_core = [&](uint32_t g, uint32_t rank) {  // core holding rank `rank` of group g
        uint32_t lo = 0, hi = G - 1;
        while (lo < hi) {
            const uint32_t mid = (lo + hi + 1) / 2;
            if (pre(mid, g) <= rank) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    };
    // Per edge: packets of earlier (level, keep) groups, and lanes of earlier levels in the same keep region.
    auto* pkb = carve(2 * 4 * LV * 2);
    auto* lnb = carve(2 * 4 * LV * 2);
    auto* edge_pk = carve(2 * 4);
    auto at = [](uint32_t io, uint32_t dr, uint32_t h, uint32_t keep) { return ((io * 4 + dr) * LV + h) * 2 + keep; };
    for (uint32_t io = 0; io < 2; ++io) {
        for (uint32_t dr = 0; dr < 4; ++dr) {
            uint32_t pk = 0, lanes[2] = {0, 0};
            for (uint32_t h = 0; h < LV; ++h) {
                for (uint32_t keep = 0; keep < 2; ++keep) {
                    const uint32_t n = ceil_div(tot[(io ? kIn : kOut) + (dr * LV + h) * 2 + keep], P);
                    pkb[at(io, dr, h, keep)] = pk;
                    lnb[at(io, dr, h, keep)] = lanes[keep];
                    pk += n;
                    lanes[keep] += n;
                }
            }
            edge_pk[io * 4 + dr] = pk;
        }
    }
    handoff[0] = a;
    handoff[1] = valid_words[a];
    handoff[2] = needed_rows[a];
    for (uint32_t o = 0; o < n_dev; ++o) {
        handoff[16 + o] = in_of[o];
    }
    for (uint32_t dr = 0; dr < 4; ++dr) {
        for (uint32_t h = 0; h < LV; ++h) {
            handoff[48 + dr * LV + h] = lnb[at(1, dr, h, 0)];
        }
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(handoff_sem_id)), 1);
    const uint32_t own_packets = ceil_div(tot[kOwn], P);
    auto locals_of = [&](uint32_t w) { return own_packets > w ? ceil_div(own_packets - w, senders) : 0u; };
    auto record_noc = [&](uint32_t w, uint32_t r) {
        return streams.get_noc_addr(w) + (kHeaderWords + r * kRecordWords) * 4;
    };

    PLAN_T(4);
    // 3. Emit. Open packets / lines live in L1 in their final layout, one per group.
    constexpr uint32_t open_bytes = kRecordWords * 4;
    auto open_rec = [&](uint32_t g) { return reinterpret_cast<uint32_t*>(open_l1 + g * open_bytes); };
    for (uint32_t i = 0; i < kGroups * kRecordWords; ++i) {
        reinterpret_cast<uint32_t*>(open_l1)[i] = 0;
    }
    auto* run = carve(kGroups);  // this core's next rank per group
    for (uint32_t g = 0; g < kGroups; ++g) {
        run[g] = pre(me, g);
    }
    uint32_t carry_slot = 0;
    // Entry `i` of the packet (line) starting at `start` of group g. Words land at their final offsets: a record
    // entry at word 8 + 2i (two words), a line entry at word i.
    auto put = [&](uint32_t g, uint32_t start, uint32_t i, uint32_t v0, uint32_t v1, bool line) {
        const uint32_t word = line ? i : 8 + 2 * i;
        if (start >= pre(me, g)) {
            auto* rec = open_rec(g);
            rec[word] = v0;
            if (!line) {
                rec[word + 1] = v1;
            }
            return;
        }
        // Begun on an earlier core: write into its carry buffer (same 16 B offset at both ends).
        const uint32_t dst = carry_l1 + g * open_bytes + word * 4;
        const uint32_t src = scratch_l1 + 64 + (carry_slot++ % 32) * 16 + (dst & 15);
        noc_async_writes_flushed();  // the ring slot's previous write has left L1
        auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        s[0] = v0;
        s[1] = v1;
        noc_async_write(src, core_noc(owner_core(g, start), dst), line ? 4 : 8);
    };
    // Writes group g's open packet (line) once complete, then clears it.
    auto flush = [&](uint32_t g, uint32_t start) {
        auto* rec = open_rec(g);
        uint64_t dst;
        uint32_t bytes;
        if (g == kNeed) {
            dst = slots_t.get_noc_addr(0) + start * 4;
            bytes = kLineSlots * 4;
        } else if (g == kOwn) {
            const uint32_t q = start / P;
            rec[0] = kLocal;
            rec[1] = lane_rel(q);
            rec[2] = tot[g] - start < P ? tot[g] - start : P;
            dst = record_noc(q % senders, q / senders);
            bytes = open_bytes;
        } else {
            const uint32_t dr = (g - kOut) / (2 * LV), h = ((g - kOut) / 2) % LV, keep = (g - kOut) % 2;
            const uint32_t k = start;
            const uint32_t n = pkb[at(0, dr, h, keep)] + k / P;
            const uint32_t w = stream_of(dr, n);
            rec[0] = h == 0 ? kSendInput : kSendOutput;
            rec[1] = base(neighbour(a, dr), dr, keep) + lane_rel(lnb[at(0, dr, h, keep)] + k / P);
            rec[2] = tot[g] - start < P ? tot[g] - start : P;
            dst = record_noc(w, locals_of(w) + h + n / per_dir);
            bytes = open_bytes;
        }
        const uint32_t staged = scratch_l1 + 1024 + (carry_slot++ % 8) * open_bytes;
        noc_async_writes_flushed();
        for (uint32_t i = 0; i < kRecordWords; ++i) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staged)[i] = rec[i];
            rec[i] = 0;
        }
        noc_async_write(staged, dst, bytes);
    };
    auto add = [&](uint32_t g, uint32_t v0, uint32_t v1) {
        const bool line = g == kNeed;
        const uint32_t size = line ? kLineSlots : P;
        const uint32_t k = run[g]++;
        const uint32_t start = k - k % size;
        put(g, start, k % size, v0, v1, line);
        const bool last = k % size == size - 1 || k + 1 == tot[g];
        if (start >= pre(me, g) && last) {
            flush(g, start);
        }
    };
    auto emit_row = [&](uint32_t e0, uint32_t outs) {
        const uint32_t local = e0 & 0xFFFFFu, need = e0 >> 20 & 1, own = e0 >> 21 & 1;
        const uint32_t in_dir = e0 >> 22 & 7, in_depth = e0 >> 25 & 15, n_out = e0 >> 29;
        uint32_t here = 0, arrival = 0;  // where the row sits on this chip, and its arrival
        if (own) {
            if (need) {
                const uint32_t q = run[kOwn];
                here = lane_rel(q / P) + (q % P) * B;
                add(kOwn, local, 0);
            }
        } else if (in_dir < 4) {
            const uint32_t keep = need ? 0 : 1;
            const uint32_t g = kIn + (in_dir * LV + in_depth) * 2 + keep;
            const uint32_t k = run[g]++;
            const uint32_t n = pkb[at(1, in_dir, in_depth, keep)] + k / P;
            here = base(a, in_dir, keep) + lane_rel(lnb[at(1, in_dir, in_depth, keep)] + k / P) + (k % P) * B;
            arrival = (stream_of(in_dir, n) << 24) | (n / per_dir + 1);
        }
        for (uint32_t e = 0; e < n_out; ++e) {
            const uint32_t o = outs >> (7 * e);
            const uint32_t dr = o & 3, h = (o >> 2) & 15, keep = (o >> 6) & 1;
            if (h == 0) {
                add(kOut + (dr * LV + h) * 2 + keep, local, 0);
            } else {
                add(kOut + (dr * LV + h) * 2 + keep, here, arrival);
            }
        }
    };
    if (!cache_full) {
        for (uint32_t i = 0; i < cached; ++i) {
            emit_row(cache[2 * i], cache[2 * i + 1]);
        }
    } else {
        for_each_entry(w0, w1, [&](uint32_t e0, uint32_t outs) { emit_row(e0, outs); });
    }

    PLAN_T(5);
    // Packets begun here and finished on later cores: their tails are in the carry buffer now.
    barrier();
    PLAN_T(6);
    for (uint32_t g = 0; g < kGroups; ++g) {
        if (g >= kIn && g < kOwn) {
            continue;  // incoming groups only rank rows; they form no packets here
        }
        const uint32_t size = g == kNeed ? kLineSlots : P;
        const uint32_t mine = run[g] - pre(me, g);
        if (!mine || run[g] == tot[g] || run[g] % size == 0) {
            continue;  // nothing here, or the last packet closed on this core
        }
        const uint32_t start = (run[g] - 1) - (run[g] - 1) % size;
        if (start < pre(me, g)) {
            continue;  // this core held only tail rows of an earlier core's packet
        }
        const uint32_t end = start + size < tot[g] ? start + size : tot[g];
        auto* rec = open_rec(g);
        auto* carry = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(carry_l1 + g * open_bytes);
        for (uint32_t k = run[g]; k < end; ++k) {
            const uint32_t i = k % size;
            if (g == kNeed) {
                rec[i] = carry[i];
            } else {
                rec[8 + 2 * i] = carry[8 + 2 * i];
                rec[9 + 2 * i] = carry[9 + 2 * i];
            }
        }
        flush(g, start);
    }

    // Headers, level markers and end records: sender w's on core w.
    if (me < senders) {
        const uint32_t w = me;
        const uint32_t dr = (w / k_per) % 4, pick = (w / (4 * k_per)) * k_per + w % k_per;
        uint32_t slot_i = 0;
        auto stage = [&]() {
            auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1 + 1024 + (slot_i++ % 8) * open_bytes);
            noc_async_writes_flushed();
            for (uint32_t i = 0; i < kRecordWords; ++i) {
                rec[i] = 0;
            }
            return rec;
        };
        auto* head = stage();
        for (uint32_t s2 = 0; s2 < senders; ++s2) {
            const uint32_t sd = (s2 / k_per) % 4, sp = (s2 / (4 * k_per)) * k_per + s2 % k_per;
            const uint32_t e = edge_pk[4 + sd];
            head[1 + s2] = e > sp ? ceil_div(e - sp, per_dir) : 0;
        }
        noc_async_write(reinterpret_cast<uint32_t>(head), streams.get_noc_addr(w), open_bytes);
        uint32_t through = 0;
        for (uint32_t h = 0; h < LV; ++h) {
            through += pkb[at(0, dr, h, 1)] - pkb[at(0, dr, h, 0)] + ceil_div(tot[kOut + (dr * LV + h) * 2 + 1], P);
            const uint32_t done = through > pick ? ceil_div(through - pick, per_dir) : 0;
            auto* rec = stage();
            rec[0] = h + 1 < LV ? kLevel : kEnd;
            noc_async_write(reinterpret_cast<uint32_t>(rec), record_noc(w, locals_of(w) + h + done), open_bytes);
        }
    }

    PLAN_T(7);
    PLAN_T(8);
    // 4. The index rewrite runs on NCRISC (REWRITE_HELPER) from the handoff written above.
#if defined(DEBUG_PRINT_ENABLED)
    PLAN_T(9);
    DPRINT(
        "TPLAN core {} pre {} dens {} b0 {} bnd {} w0 {} w1 {} rows {} setup {} count {} b1 {} xch {} emit {} b2 {} "
        "tail {} b3 {} rewrite {}\n",
        me,
        stamp[10] - stamp[0],
        stamp[11] - stamp[10],
        stamp[12] - stamp[11],
        stamp[1] - stamp[12],
        w0,
        w1,
        visited,
        stamp[1] - stamp[0],
        stamp[2] - stamp[1],
        stamp[3] - stamp[2],
        stamp[4] - stamp[3],
        stamp[5] - stamp[4],
        stamp[6] - stamp[5],
        stamp[7] - stamp[6],
        stamp[8] - stamp[7],
        stamp[9] - stamp[8]);
#endif
}
