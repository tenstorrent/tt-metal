// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm narrow-flat reader (NCRISC / NoC0) — ROW_MAJOR WIDTH / BLOCK_SHARDED shards whose width w_s divides a
// 16-datum face row (the golden harness's 16 B RM granule: 8 bf16 / 4 fp32 datums), see
// rms_norm_narrow_flat_program_descriptor.py.
//
// The shard (N sticks x w_s datums, sticks back to back) is read by compute as F_full "flat tiles" of 1024
// consecutive datums straight from the input shard (cb_x is backed on it, tile-sized pages): the reader only
// publishes those pages. Then, in the order compute needs them:
//   1. cb_mms (rd_const): the NB = min(R, F) pass-1 0/1 matrices S_j: S_j[i][c] = 1 iff c == j * G + i / w_s and
//      i % w_s < v (G = 32 / w_s stick groups per logical row, v = this member's valid columns: the W tail mask is
//      folded into S). Zeros from the NoC zero engine, the ones from a compile-time store table.
//   2. the last partial flat tile (rd_tail): the shard bytes past F_full tiles copied into cb_xtail (one local NoC
//      read; the caller's shard is never written) and zero-filled to the tile end.
//   3. gamma (rd_gamma): ONE 64 B-aligned DRAM read of the window holding this member's w_s gamma datums (BH DRAM
//      reads are 64 B aligned; the slice is w_s * g_elem <= 64 B and aligned to its own size, so it never straddles a
//      window), pad columns zeroed, folded into the pass-2 expand matrices cb_mme: E_j[c][i] = gamma[i % w_s] at
//      c == j * G + i / w_s (1.0 without gamma) — out = (inv_u @ E_j) * x needs no gamma multiply.
//   4. owner duties (rd_wait_peers / rd_inv_return / rd_lead_mcast): see the block comment below.
//
// Raw dataflow API (documented helper bypasses): no kernel_lib helper builds a 0/1 selection matrix; the inv return
// multicasts the leader's cb_inv tile to <= MAX_RECTS member rectangles with a flag per unit while the k members
// are simultaneously senders of other units — kernel_lib Mcast fixes ONE sender per group (or rotates it in a fixed
// order), so the per-unit leader (owner of the unit's first sub-unit) cannot be expressed. Capability gap.

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "rms_norm_narrow_flat_common.hpp"

using namespace rms_norm_narrow_flat_cb;

constexpr uint32_t F_full = get_compile_time_arg_val(0);
constexpr uint32_t tail_bytes = get_compile_time_arg_val(1);
constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t w_s = get_compile_time_arg_val(3);
constexpr uint32_t R = get_compile_time_arg_val(4);  // flat tiles per unit (= w_s)
constexpr bool acc_fp32 = get_compile_time_arg_val(5) != 0;
constexpr uint32_t g_elem_bytes = get_compile_time_arg_val(6);
constexpr uint32_t g_page_bytes = get_compile_time_arg_val(7);
constexpr uint32_t NB = get_compile_time_arg_val(8);  // constant matrices built: min(R, F)
constexpr bool mme_fp32 = get_compile_time_arg_val(9) != 0;
constexpr uint32_t U = get_compile_time_arg_val(10);
constexpr uint32_t K = get_compile_time_arg_val(11);
constexpr uint32_t acc_tile_bytes = get_compile_time_arg_val(12);
constexpr uint32_t sem_part_id = get_compile_time_arg_val(13);
constexpr uint32_t flag_bytes = get_compile_time_arg_val(14);
constexpr uint32_t NOC_XY_ARG_BASE = get_compile_time_arg_val(15);
constexpr uint32_t Q = get_compile_time_arg_val(16);   // exchange sub-blocks per unit tile
constexpr uint32_t KT = get_compile_time_arg_val(17);  // gathered tiles per owned sub-unit
constexpr uint32_t RECT_ARG_BASE = get_compile_time_arg_val(18);
[[maybe_unused]] constexpr auto g_args = TensorAccessorArgs<19>();

constexpr uint32_t G = 32 / w_s;  // stick groups per logical tile row
static_assert(16 % w_s == 0, "narrow-flat regime: w_s must divide a 16-datum face row");

// Datum index of logical (r, c) inside a 32x32 tile (faces row-major, 16x16 each).
constexpr uint32_t tile_pos(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// Compile-time store tables (with the index math in the loop the build cost ~20 RISC-V instructions per datum —
// 3.5 us on the critical path; tables: 1.0 us): word offset + word of every S_j 1.0 (one per tile row; bf16: the
// other datum of the word is 0 — a row never holds two ones), and the word offset of every E_j gamma segment.
constexpr uint32_t S_ONES = NB * 32;
struct WordStore {
    uint32_t off;
    uint32_t val;
};
template <bool fp32>
constexpr std::array<WordStore, S_ONES> make_s_table() {
    std::array<WordStore, S_ONES> t{};
    uint32_t k = 0;
    for (uint32_t j = 0; j < NB; ++j) {
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t p = j * 1024 + tile_pos(i, j * G + i / w_s);
            t[k++] = fp32 ? WordStore{p, 0x3F800000u} : WordStore{p >> 1, (p & 1) ? 0x3F800000u : 0x00003F80u};
        }
    }
    return t;
}
constexpr uint32_t E_SEGS = NB * G;
template <uint32_t me>
constexpr std::array<uint32_t, E_SEGS> make_e_table() {
    std::array<uint32_t, E_SEGS> t{};
    for (uint32_t j = 0; j < NB; ++j) {
        for (uint32_t q = 0; q < G; ++q) {
            t[j * G + q] = (j * 1024 + tile_pos(j * G + q, q * w_s)) * me / 4;
        }
    }
    return t;
}

FORCE_INLINE uint64_t peer_addr(uint32_t peer, uint32_t l1_addr) {
    return get_noc_addr(
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer),
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer + 1),
        l1_addr);
}

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t valid_cols = get_arg_val<uint32_t>(1);  // valid columns of this member's slice (<= w_s)
    [[maybe_unused]] const uint32_t col0 = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t g_addr = get_arg_val<uint32_t>(3);
    const uint32_t member_idx = get_arg_val<uint32_t>(4);
    const uint32_t owned = get_arg_val<uint32_t>(5);
    constexpr uint32_t ONE = 0x3F800000u;

    // ---- full flat tiles: cb_x IS the shard ----
    if constexpr (F_full > 0) {
        cb_reserve_back(cb_x, F_full);
        cb_push_back(cb_x, F_full);
    }

    Noc noc;
    // ---- pass-1 0/1 matrices S_j ----
    {
        MaybeDeviceZoneScope("rd_const");
        constexpr uint32_t t_bytes = 1024 * (acc_fp32 ? 4 : 2);
        cb_reserve_back(cb_mms, NB);
        CircularBuffer mms(cb_mms);
        noc.async_write_zeros(mms, NB * t_bytes);
        noc.write_zeros_l1_barrier();
        static constexpr auto s_tab = make_s_table<acc_fp32>();
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_mms));
#pragma GCC unroll 8
        for (uint32_t k = 0; k < S_ONES; ++k) {
            w[s_tab[k].off] = s_tab[k].val;
        }
        if (valid_cols < w_s) {  // W tail member: padding columns never enter sum x^2
            for (uint32_t k = 0; k < S_ONES; ++k) {
                if ((k % 32) % w_s >= valid_cols) {
                    w[s_tab[k].off] = 0;
                }
            }
        }
        cb_push_back(cb_mms, NB);
    }

    // ---- last partial flat tile: local copy + zero fill ----
    if constexpr (tail_bytes > 0) {
        MaybeDeviceZoneScope("rd_tail");
        cb_reserve_back(cb_xtail, 1);
        const uint32_t l1 = get_write_ptr(cb_xtail);
        noc_async_read(get_noc_addr(x_addr + F_full * x_tile_bytes), l1, tail_bytes);
        CircularBuffer xt(cb_xtail);
        noc.async_write_zeros(xt, x_tile_bytes - tail_bytes, {.offset_bytes = tail_bytes});
        noc.write_zeros_l1_barrier();
        noc_async_read_barrier();
        cb_push_back(cb_xtail, 1);
    }

    // ---- gamma slice (this member's w_s datums; pad columns 0) folded into the expand matrices E_j ----
    {
        MaybeDeviceZoneScope("rd_gamma");
        uint32_t g_bits[w_s];  // gamma datums as fp32 bit patterns
        constexpr uint32_t t_bytes = 1024 * (mme_fp32 ? 4 : 2);
        cb_reserve_back(cb_mme, NB);
        const uint32_t base = get_write_ptr(cb_mme);
#ifdef RMS_HAS_GAMMA
        {
            const auto g_acc = TensorAccessor(g_args, g_addr, g_page_bytes);
            const uint32_t byte0 = col0 * g_elem_bytes;
            const uint32_t lead = byte0 & 63;
            noc_async_read(g_acc.get_noc_addr(0, byte0 - lead), base, 64);  // cb_mme as scratch (zeroed below)
            noc_async_read_barrier();
            for (uint32_t e = 0; e < w_s; ++e) {
                uint32_t bits = 0;
                if (e < valid_cols) {
                    if constexpr (g_elem_bytes == 4) {
                        bits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + lead)[e];
                    } else {
                        bits = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + lead)[e])
                               << 16;
                    }
                }
                g_bits[e] = bits;
            }
        }
#else
        for (uint32_t e = 0; e < w_s; ++e) {
            g_bits[e] = ONE;
        }
#endif
        CircularBuffer mme(cb_mme);
        noc.async_write_zeros(mme, NB * t_bytes);
        noc.write_zeros_l1_barrier();
        // E_j row r = j * G + q holds gamma[0 .. w_s) at columns q * w_s .. q * w_s + w_s - 1: w_s consecutive
        // datums inside one face row -> the packed slice as whole words
        constexpr uint32_t me = mme_fp32 ? 4 : 2;
        constexpr uint32_t seg_words = w_s * me / 4;
        uint32_t seg[seg_words];
        for (uint32_t w = 0; w < seg_words; ++w) {
            if constexpr (mme_fp32) {
                seg[w] = g_bits[w];
            } else {
                seg[w] = (g_bits[2 * w] >> 16) | (g_bits[2 * w + 1] & 0xFFFF0000u);
            }
        }
        static constexpr auto e_tab = make_e_table<me>();
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
#pragma GCC unroll 4
        for (uint32_t k = 0; k < E_SEGS; ++k) {
#pragma GCC unroll 8
            for (uint32_t w = 0; w < seg_words; ++w) {
                d[e_tab[k] + w] = seg[w];
            }
        }
        cb_push_back(cb_mme, NB);
    }

    // ---- owner duties: owned sub-units v = member_idx + j * K (unit u = v / Q, sub-block s = v % Q) ----
    // gather (the k contributions of v, counted on the partials semaphore) -> compute sums the KT gathered tiles,
    // folds the Q sub-blocks and finalizes -> sub-block 0 of the result is unit u's sub-block s of inv: one unicast
    // into the cb_inv tile u of the unit's LEADER (owner of sub-unit u * Q) + increment of its collect[u]. The leader
    // waits the Q sub-blocks, then sends the whole tile to every other member with ONE multicast per member rectangle
    // (data linked to the flag set that follows it, as kernel_lib Mcast's Flag signal does) and sets its own flag.
    // Per-sub-unit multicasts (V concurrent senders into the same rectangles) measured 3-8x slower: they serialize;
    // per-sub-unit unicasts cost k writes + k increments per owner. Groups that are not <= MAX_RECTS rectangles fall
    // back to k - 1 unicasts from the leader. On the reader, so that the writer's partial sends never wait behind a
    // return. cb_gathered / cb_inv / cb_flags bases are the CBs' initial write pointers (identical on all members).
    if (owned > 0) {
        constexpr uint32_t T = acc_tile_bytes;
        constexpr uint32_t SB = T / Q;
        constexpr uint32_t V = U * Q;
        const uint32_t sem_part = get_semaphore(sem_part_id);
        const uint32_t inv_base = get_write_ptr(cb_inv);
        const uint32_t flags_base = get_write_ptr(cb_flags);
        const uint32_t g_base = get_write_ptr(cb_gathered);
        const uint32_t collect_base = flags_base + V * flag_bytes;   // per-unit collect counters (leaders)
        const uint32_t one_src = flags_base + (V + U) * flag_bytes;  // multicast flag source
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one_src) = 1;
        const uint32_t n_rect = get_arg_val<uint32_t>(RECT_ARG_BASE);
        const uint32_t self_mask = get_arg_val<uint32_t>(RECT_ARG_BASE + 1);  // bit r: rectangle r holds this core

        // K % Q != 0: the last gathered tile of each owned slot has sub-blocks no member writes -> zero them (sum = 0)
        if constexpr (K % Q != 0) {
            for (uint32_t j = 0; j < owned; ++j) {
                const uint32_t a0 = g_base + (j * KT + KT - 1) * T + (K % Q) * SB;
                noc.async_write_zeros(CircularBuffer(cb_gathered), (Q - K % Q) * SB, {.offset_bytes = a0 - g_base});
            }
            noc.write_zeros_l1_barrier();
        }
        for (uint32_t j = 0; j < owned; ++j) {
            const uint32_t v = member_idx + j * K;
            const uint32_t u = v / Q;
            {
                MaybeDeviceZoneScope("rd_wait_peers");
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_part), (j + 1) * K);
            }
            cb_reserve_back(cb_gathered, KT);
            cb_push_back(cb_gathered, KT);
            {
                MaybeDeviceZoneScope("rd_inv_return");  // incl. the wait on compute's combine
                cb_wait_front(cb_inv_owned, 1);
                const uint32_t leader = (u * Q) % K;
                noc_async_write(get_read_ptr(cb_inv_owned), peer_addr(leader, inv_base + u * T + (v % Q) * SB), SB);
                noc_async_writes_flushed();
                noc_semaphore_inc(peer_addr(leader, collect_base + u * flag_bytes), 1);
                noc_async_write_barrier();  // cb_inv_owned page reused after the pop
                cb_pop_front(cb_inv_owned, 1);
            }
            if (v % Q == 0) {  // this member leads unit u
                MaybeDeviceZoneScope("rd_lead_mcast");
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(collect_base + u * flag_bytes), Q);
                const uint32_t tile_l1 = inv_base + u * T;
                const uint32_t flag_l1 = flags_base + u * Q * flag_bytes;
                if (n_rect > 0) {
                    for (uint32_t r = 0; r < n_rect; ++r) {
                        const uint32_t a = RECT_ARG_BASE + 2 + 5 * r;
                        uint32_t x0 = get_arg_val<uint32_t>(a), y0 = get_arg_val<uint32_t>(a + 1);
                        uint32_t x1 = get_arg_val<uint32_t>(a + 2), y1 = get_arg_val<uint32_t>(a + 3);
                        const uint32_t n = get_arg_val<uint32_t>(a + 4);
                        const bool self = (self_mask >> r) & 1;
                        const uint32_t nd = self ? n - 1 : n;  // non-loopback: the leader already holds the tile
                        if (nd == 0) {
                            continue;
                        }
                        if (noc_index == 1) {  // NoC1 multicasts run from the high corner
                            const uint32_t tx = x0, ty = y0;
                            x0 = x1;
                            y0 = y1;
                            x1 = tx;
                            y1 = ty;
                        }
                        noc_async_write_multicast(
                            tile_l1, get_noc_multicast_addr(x0, y0, x1, y1, tile_l1), T, nd, true /*linked*/);
                        noc_semaphore_set_multicast(one_src, get_noc_multicast_addr(x0, y0, x1, y1, flag_l1), nd);
                    }
                } else {
                    for (uint32_t p = 0; p < K; ++p) {
                        if (p != member_idx) {
                            noc_async_write(tile_l1, peer_addr(p, tile_l1), T);
                        }
                    }
                    noc_async_writes_flushed();
                    for (uint32_t p = 0; p < K; ++p) {
                        if (p != member_idx) {
                            noc_semaphore_inc(peer_addr(p, flag_l1), 1);
                        }
                    }
                }
                noc_async_write_barrier();
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_l1) = 1;
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
}
