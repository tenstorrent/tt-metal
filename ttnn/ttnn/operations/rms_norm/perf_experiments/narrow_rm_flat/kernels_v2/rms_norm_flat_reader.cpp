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
[[maybe_unused]] constexpr auto g_args = TensorAccessorArgs<17>();

constexpr uint32_t G = 32 / w_s;  // stick groups per logical tile row
static_assert(16 % w_s == 0, "flat regime: w_s must divide a 16-datum face row");

// Datum index of logical (r, c) inside a 32x32 tile (faces row-major, 16x16 each).
FORCE_INLINE uint32_t tile_pos(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
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
        noc.async_write_zeros(mms, n_mms * t_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t base = get_write_ptr(cb_mms);
        for (uint32_t i = 0; i < 32; ++i) {
            if (i % w_s >= valid_cols) {
                continue;  // W tail: padding columns never enter sum x^2
            }
#ifdef FLAT_CMP
            for (uint32_t j = 0; j < R; ++j) {
                set_val<acc_fp32>(base + j * t_bytes, i, j * G + i / w_s, ONE);
            }
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
            cb_reserve_back(cb_mme, R);
            CircularBuffer mme(cb_mme);
            noc.async_write_zeros(mme, R * t_bytes);
            noc.write_zeros_l1_barrier();
            const uint32_t base = get_write_ptr(cb_mme);
            for (uint32_t j = 0; j < R; ++j) {
                for (uint32_t i = 0; i < 32; ++i) {
                    set_val<mme_fp32>(base + j * t_bytes, j * G + i / w_s, i, g_bits[i % w_s]);
                }
            }
            cb_push_back(cb_mme, R);
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

    // ---- owner duties (units o = member_idx + j * K): gather -> compute combines -> return to every member ----
    // On the reader so that the writer's partial sends (and with them every peer's combine) never wait behind a
    // k-way return. cb_gathered / cb_inv / cb_flags bases are the CBs' initial write pointers (identical on all
    // members); the per-unit flag is incremented after the inv tile's writes are flushed (same NoC, in order).
    if (owned > 0) {
        constexpr uint32_t T = acc_tile_bytes;
        const uint32_t sem_part = get_semaphore(sem_part_id);
        const uint32_t inv_base = get_write_ptr(cb_inv);
        const uint32_t flags_base = get_write_ptr(cb_flags);
        for (uint32_t j = 0; j < owned; ++j) {
            const uint32_t o = member_idx + j * K;
            {
                MaybeDeviceZoneScope("rd_wait_peers");
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_part), (j + 1) * K);
            }
            cb_reserve_back(cb_gathered, K);
            cb_push_back(cb_gathered, K);
            MaybeDeviceZoneScope("rd_inv_return");  // incl. the wait on compute's combine
            cb_wait_front(cb_inv_owned, 1);
            const uint32_t src = get_read_ptr(cb_inv_owned);
            for (uint32_t p = 0; p < K; ++p) {
                noc_async_write(src, peer_addr(p, inv_base + o * T), T);
            }
            noc_async_writes_flushed();
            for (uint32_t p = 0; p < K; ++p) {
                noc_semaphore_inc(peer_addr(p, flags_base + o * flag_bytes), 1);
            }
            cb_pop_front(cb_inv_owned, 1);
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
}
