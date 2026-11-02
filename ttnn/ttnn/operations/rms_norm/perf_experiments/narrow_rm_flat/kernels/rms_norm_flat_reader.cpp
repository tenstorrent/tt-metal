// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm flat reader (NCRISC / NoC0) — ROW_MAJOR WIDTH / BLOCK_SHARDED shards of width w_s | 16 datums.
//
// The shard (N sticks x w_s datums, sticks back to back) is read by compute as F_full "flat tiles" of 1024
// consecutive datums straight from the input shard (cb_x is backed on it, tile-sized pages): the reader only
// publishes those pages. Then, in the order compute needs them:
//   1. cb_mms pass-1 0/1 constants (rd_const): FLAT_CMP — S_j (j < R): S_j[i][c] = 1 iff c == j * G + i / w_s and
//      i % w_s < v (G = 32 / w_s stick groups per logical row, v = this member's valid columns: the W tail mask is
//      folded into S); broadcast form — B[i][c] = 1 iff i / w_s == c / w_s and i % w_s < v.
//   2. the last partial flat tile (rd_tail): the shard bytes past F_full tiles copied into cb_xtail (one local
//      NoC read; the caller's shard is never written) and zero-filled to the tile end.
//   3. gamma (rd_gamma): ONE 64 B-aligned DRAM read of the window holding this member's w_s gamma datums (BH DRAM
//      reads are 64 B aligned; the slice is w_s * g_elem <= 64 B and aligned to its own size, so it never straddles a
//      window), pad columns zeroed. FLAT_CMP: folded into the pass-2 expand matrices, cb_mme E_j[c][i] =
//      gamma[i % w_s] at c == j * G + i / w_s (1.0 without gamma) — out = (inv_u @ E_j) * x needs no gamma multiply.
//      Broadcast form: gamma_flat = the slice repeated 1024 / w_s times (cb_gamma).
// Zeros come from the NoC zero engine; the non-zeros are RISC-V stores (32 per constant tile).
//
// Raw dataflow API (documented helper bypass): no kernel_lib helper builds a 0/1 selection matrix or repeats a
// sub-row pattern over a tile; the per-datum stores are the whole job.

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "rms_norm_flat_common.hpp"

using namespace rms_norm_flat_cb;

constexpr uint32_t F_full = get_compile_time_arg_val(0);
constexpr uint32_t tail_bytes = get_compile_time_arg_val(1);
constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t w_s = get_compile_time_arg_val(3);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
constexpr uint32_t R = get_compile_time_arg_val(5);
constexpr bool acc_fp32 = get_compile_time_arg_val(6) != 0;
constexpr uint32_t g_elem_bytes = get_compile_time_arg_val(7);
constexpr uint32_t g_page_bytes = get_compile_time_arg_val(8);
constexpr uint32_t n_mms = get_compile_time_arg_val(9);
constexpr bool mme_fp32 = get_compile_time_arg_val(10) != 0;
constexpr uint32_t U = get_compile_time_arg_val(11);
constexpr uint32_t K = get_compile_time_arg_val(12);
constexpr uint32_t acc_tile_bytes = get_compile_time_arg_val(13);
constexpr uint32_t sem_part_id = get_compile_time_arg_val(14);
constexpr uint32_t flag_bytes = get_compile_time_arg_val(15);
constexpr uint32_t NOC_XY_ARG_BASE = get_compile_time_arg_val(16);
constexpr uint32_t Q = get_compile_time_arg_val(17);
constexpr uint32_t KT = get_compile_time_arg_val(18);
constexpr uint32_t RECT_ARG_BASE = get_compile_time_arg_val(19);
// inv return: 0 = per sub-unit unicast to every member, 1 = per sub-unit multicast, 2 = sub-unit owners forward to the
// unit's leader (owner of sub-unit u * Q), which multicasts the whole inv tile once.
constexpr uint32_t RET_MODE = get_compile_time_arg_val(20);
[[maybe_unused]] constexpr auto g_args = TensorAccessorArgs<21>();

constexpr uint32_t G = 32 / w_s;  // stick groups per logical tile row
static_assert(16 % w_s == 0, "flat regime: w_s must divide a 16-datum face row");

// Datum index of logical (r, c) inside a 32x32 tile (faces row-major, 16x16 each).
constexpr uint32_t tile_pos(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// 1.0 at datum index p of a tile, as ONE 32-bit store (bf16: the word holds datums p & ~1 and p | 1; the other half
// is zero in a zeroed tile — sub-word L1 stores measured ~3x slower per datum on BH).
template <bool fp32>
FORCE_INLINE void set_one_word(uint32_t tile_base, uint32_t p) {
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_base);
    if constexpr (fp32) {
        w[p] = 0x3F800000u;
    } else {
        w[p >> 1] = (p & 1) ? 0x3F800000u : 0x00003F80u;
    }
}

// Compile-time store tables (the RISC-V loop cost, ~20 instructions per datum with the index math, dominated the
// constant build): word offsets (from the cb_mms base) + words of the R ones-matrices S_j, and the word offsets of
// the E_j gamma segments (from the cb_mme base).
constexpr uint32_t NB = n_mms;        // FLAT_CMP: constant matrices built (min(R, F))
constexpr uint32_t S_ONES = NB * 32;  // one 1.0 per tile row
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
            const uint32_t p = j * 1024 + tile_pos(i, (R > 1 ? j * G : 0) + i / w_s);
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

template <bool fp32>
FORCE_INLINE void set_val(uint32_t tile_base, uint32_t r, uint32_t c, uint32_t bits32) {
    const uint32_t p = tile_pos(r, c);
    if constexpr (fp32) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_base)[p] = bits32;
    } else {
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_base)[p] = bits32 >> 16;
    }
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
    // ---- pass-1 0/1 constants ----
    {
        MaybeDeviceZoneScope("rd_const");
        constexpr uint32_t t_bytes = 1024 * (acc_fp32 ? 4 : 2);
        cb_reserve_back(cb_mms, n_mms);
        CircularBuffer mms(cb_mms);
        {
            MaybeDeviceZoneScope("rd_const_zero");
            noc.async_write_zeros(mms, n_mms * t_bytes);
            noc.write_zeros_l1_barrier();
        }
        const uint32_t base = get_write_ptr(cb_mms);
        {
            MaybeDeviceZoneScope("rd_const_ones");
#ifdef FLAT_CMP
            // S_j: row i's single 1.0 at column j * G + i / w_s (one word store each; a row never holds two ones,
            // so the word's other datum stays 0). W tail member: rows i % w_s >= valid_cols re-zeroed after.
            static constexpr auto s_tab = make_s_table<acc_fp32>();
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
#pragma GCC unroll 8
            for (uint32_t k = 0; k < S_ONES; ++k) {
                w[s_tab[k].off] = s_tab[k].val;
            }
            if (valid_cols < w_s) {
                for (uint32_t k = 0; k < S_ONES; ++k) {
                    if ((k % 32) % w_s >= valid_cols) {
                        w[s_tab[k].off] = 0;
                    }
                }
            }
#endif
        }
        for (uint32_t i = 0; i < 32; ++i) {
            if (i % w_s >= valid_cols) {
                continue;  // W tail: padding columns never enter sum x^2
            }
#ifdef FLAT_CMP
#else
            const uint32_t c0 = (i / w_s) * w_s;
            for (uint32_t c = c0; c < c0 + w_s; ++c) {
                set_val<acc_fp32>(base, i, c, ONE);
            }
#endif
        }
        cb_push_back(cb_mms, n_mms);
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

    // ---- gamma slice (this member's w_s datums; pad columns 0) ----
    {  // gamma slice + gamma consumers (cmp: expand matrices; broadcast: gamma_flat)
        MaybeDeviceZoneScope("rd_gamma");
#ifdef FLAT_CMP
        constexpr uint32_t scratch_cb = cb_mme;
#else
        constexpr uint32_t scratch_cb = cb_gamma;
#endif
        [[maybe_unused]] uint32_t g_bits[w_s];  // gamma datums as fp32 bit patterns
#ifdef RMS_HAS_GAMMA
        {
            const auto g_acc = TensorAccessor(g_args, g_addr, g_page_bytes);
            cb_reserve_back(scratch_cb, 1);
            const uint32_t base = get_write_ptr(scratch_cb);
            const uint32_t byte0 = col0 * g_elem_bytes;
            const uint32_t lead = byte0 & 63;
            noc_async_read(g_acc.get_noc_addr(0, byte0 - lead), base, 64);
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

#ifdef FLAT_CMP
        {
            constexpr uint32_t t_bytes = 1024 * (mme_fp32 ? 4 : 2);
            cb_reserve_back(cb_mme, NB);
            CircularBuffer mme(cb_mme);
            noc.async_write_zeros(mme, NB * t_bytes);
            noc.write_zeros_l1_barrier();
            const uint32_t base = get_write_ptr(cb_mme);
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
#elif defined(RMS_HAS_GAMMA)
        {
            const uint32_t base = get_write_ptr(cb_gamma);
            constexpr uint32_t slice_bytes = w_s * g_elem_bytes;
            constexpr uint32_t slice_words = slice_bytes / 4;
            uint32_t tmp[slice_words];
            uint8_t* b = reinterpret_cast<uint8_t*>(tmp);
            for (uint32_t e = 0; e < w_s; ++e) {
                if constexpr (g_elem_bytes == 4) {
                    reinterpret_cast<uint32_t*>(b)[e] = g_bits[e];
                } else {
                    reinterpret_cast<uint16_t*>(b)[e] = g_bits[e] >> 16;
                }
            }
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
            constexpr uint32_t reps = 1024 / w_s;
            for (uint32_t rep = 0; rep < reps; ++rep) {
#pragma GCC unroll 16
                for (uint32_t w = 0; w < slice_words; ++w) {
                    dst[rep * slice_words + w] = tmp[w];
                }
            }
            cb_push_back(cb_gamma, 1);
        }
#endif
    }

    // ---- owner duties: owned sub-units v = member_idx + j * K (unit u = v / Q, sub-block s = v % Q) ----
    // gather -> compute sums the KT gathered tiles, folds the Q sub-blocks and finalizes -> sub-block 0 of the result
    // is unit u's sub-block s of inv. On the reader, so that the writer's partial sends never wait behind a return.
    //   RET_MODE 0: unicast into tile u, sub-block s of every member's cb_inv + increment of each member's flag[v].
    //   RET_MODE 1: the same as one multicast per member rectangle (data linked to the flag set that follows it, as
    //               kernel_lib Mcast's Flag signal does).
    //   RET_MODE 2: unicast to the unit's leader L(u) (owner of sub-unit u * Q) into its cb_inv tile u + increment of
    //               its collect[u]; the leader waits the Q sub-blocks, then multicasts the whole tile + flag[u * Q]
    //               (one multicast per unit — concurrent per-sub-unit multicasts collide).
    // Groups that are not <= MAX_RECTS rectangles fall back to unicast. cb_gathered / cb_inv / cb_flags bases are the
    // CBs' initial write pointers (identical on all members).
    if (owned > 0) {
        constexpr uint32_t T = acc_tile_bytes;
        constexpr uint32_t SB = T / Q;
        constexpr uint32_t V = U * Q;
        const uint32_t sem_part = get_semaphore(sem_part_id);
        const uint32_t inv_base = get_write_ptr(cb_inv);
        const uint32_t flags_base = get_write_ptr(cb_flags);
        const uint32_t g_base = get_write_ptr(cb_gathered);
        const uint32_t collect_base = flags_base + V * flag_bytes;   // RET_MODE 2: per-unit collect counters
        const uint32_t one_src = flags_base + (V + U) * flag_bytes;  // multicast flag source
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one_src) = 1;
        const uint32_t n_rect = get_arg_val<uint32_t>(RECT_ARG_BASE);
        const uint32_t self_mask = get_arg_val<uint32_t>(RECT_ARG_BASE + 1);  // bit r: rectangle r holds this core

        // `bytes` from local `src` to `dst_l1` on every member (+ flag at flag_l1 = 1); include_self: also this core.
        auto broadcast = [&](uint32_t src, uint32_t dst_l1, uint32_t bytes, uint32_t flag_l1, bool include_self) {
            if (n_rect > 0) {
                for (uint32_t r = 0; r < n_rect; ++r) {
                    const uint32_t base = RECT_ARG_BASE + 2 + 5 * r;
                    uint32_t x0 = get_arg_val<uint32_t>(base), y0 = get_arg_val<uint32_t>(base + 1);
                    uint32_t x1 = get_arg_val<uint32_t>(base + 2), y1 = get_arg_val<uint32_t>(base + 3);
                    const uint32_t n = get_arg_val<uint32_t>(base + 4);
                    const bool self = (self_mask >> r) & 1;
                    if (self && !include_self && n == 1) {
                        continue;  // the rectangle is just this core
                    }
                    if (noc_index == 1) {
                        const uint32_t tx = x0, ty = y0;
                        x0 = x1;
                        y0 = y1;
                        x1 = tx;
                        y1 = ty;
                    }
                    const uint64_t d_data = get_noc_multicast_addr(x0, y0, x1, y1, dst_l1);
                    const uint64_t d_flag = get_noc_multicast_addr(x0, y0, x1, y1, flag_l1);
                    if (self && include_self) {
                        noc_async_write_multicast_loopback_src(src, d_data, bytes, n, true);
                        noc_semaphore_set_multicast_loopback_src(one_src, d_flag, n, false);
                    } else {
                        const uint32_t nd = self ? n - 1 : n;
                        noc_async_write_multicast(src, d_data, bytes, nd, true);
                        noc_semaphore_set_multicast(one_src, d_flag, nd, false);
                    }
                }
            } else {
                for (uint32_t p = 0; p < K; ++p) {
                    if (include_self || p != member_idx) {
                        noc_async_write(src, peer_addr(p, dst_l1), bytes);
                    }
                }
                noc_async_writes_flushed();
                for (uint32_t p = 0; p < K; ++p) {
                    if (include_self || p != member_idx) {
                        noc_semaphore_inc(peer_addr(p, flag_l1), 1);
                    }
                }
            }
            noc_async_write_barrier();  // the source is reused (cb_inv_owned page) / published after this
        };

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
            const uint32_t dst_l1 = inv_base + u * T + (v % Q) * SB;
            {
                MaybeDeviceZoneScope("rd_wait_peers");
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_part), (j + 1) * K);
            }
            cb_reserve_back(cb_gathered, KT);
            cb_push_back(cb_gathered, KT);
            MaybeDeviceZoneScope("rd_inv_return");  // incl. the wait on compute's combine
            cb_wait_front(cb_inv_owned, 1);
            const uint32_t src = get_read_ptr(cb_inv_owned);
            if constexpr (RET_MODE == 2) {
                const uint32_t leader = (u * Q) % K;
                noc_async_write(src, peer_addr(leader, dst_l1), SB);
                noc_async_writes_flushed();
                noc_semaphore_inc(peer_addr(leader, collect_base + u * flag_bytes), 1);
                noc_async_write_barrier();  // cb_inv_owned page reused after the pop
                cb_pop_front(cb_inv_owned, 1);
                if (v % Q == 0) {  // this member leads unit u
                    MaybeDeviceZoneScope("rd_lead_mcast");
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(collect_base + u * flag_bytes), Q);
                    const uint32_t flag_l1 = flags_base + u * Q * flag_bytes;
                    broadcast(inv_base + u * T, inv_base + u * T, T, flag_l1, false);
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_l1) = 1;
                }
            } else {
                const uint32_t flag_l1 = flags_base + v * flag_bytes;
                if constexpr (RET_MODE == 1) {
                    broadcast(src, dst_l1, SB, flag_l1, true);
                } else {
                    const uint32_t keep = n_rect;
                    (void)keep;
                    for (uint32_t p = 0; p < K; ++p) {
                        noc_async_write(src, peer_addr(p, dst_l1), SB);
                    }
                    noc_async_writes_flushed();
                    for (uint32_t p = 0; p < K; ++p) {
                        noc_semaphore_inc(peer_addr(p, flag_l1), 1);
                    }
                    noc_async_write_barrier();
                }
                cb_pop_front(cb_inv_owned, 1);
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
}
