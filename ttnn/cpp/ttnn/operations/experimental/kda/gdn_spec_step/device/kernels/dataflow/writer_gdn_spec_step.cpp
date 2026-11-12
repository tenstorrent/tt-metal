// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer: one core = (user u, value head h). Reads the ctrl page (parity, HOLD). Window: rows 0..Lw-1 (rounded up to
// an even count; the extra row is an exact zero) of the compute's rebuilt window tiles go to win[1-par] (q/k chunks
// only from heads with h % rf == 0, v chunks from every head); a held user's tiles are copied through
// win[par] -> win[1-par] via a bounce buffer and the compute's tiles are drained. Ring: per token the post-token state
// (KV fp32 tiles) from `hnew` to block (t*B*Nv + bh), skipped for held heads. Then the user's T output rows (whole
// face-row spans when T is even, 32 B single rows otherwise). The output tensor is not zero-filled at creation, so the
// cores of the last user (u = B-1) also write exact zeros into rows [B*T % 32, 32) of the last tile row; with
// R <= round_up(B*T, 32) enforced by the host, every output row outside [u*T, u*T + T) is then exactly 0.
// HOLD is decided per (u,h) ctrl word; q/k window chunk ownership is per key-head group (h % rf == 0), so a caller
// must hold ALL Nv heads of a user (the op cannot check ctrl contents).
// GDN_RING_PAGE_IDX: the state after token t goes to ring page ring_page_idx[u][t*Nv + h] instead of block t*B*Nv + bh.
// GDN_RING_POOL2 (= ring's page count): pages from there on are ring_pool2's (page - count).
// VB > 1 (value blocks, GDN_VSPLIT): core (u, h, s) owns the state / output column tiles [s*Vs, (s+1)*Vs). Block 0
// alone writes (or copies through) the window; each block writes its KVs state tiles (i, s*Vs + j) per token and only
// its Vs output tiles, and before that swaps the last token's pre-norm output tiles with the head's other blocks:
// acc_mine (Vs fp32 tiles, exact bytes) -> acc_x[s*Vs ..] on every block (itself included) by unicast NoC writes.
// Handshake (both semaphores start at 0 on every launch): each block raises `ready` on every block once its acc_x is
// reserved, waits for VB before writing, write-barriers, then raises `valid` on every block and waits for VB before it
// hands acc_x to the compute.
// GDN_QKSHARE (kernel_opts bit 4, VB > 1): each block writes the window rows of its own chunks only, and swaps its Qk
// q/k conv tiles into every block's qc / kc the same way (semaphore `qkv`).
// GDN_GATE_SCALARS (kernel_opts bit 1): this writer, idle until the compute's phase A hands it the window tiles,
// computes the T beta / decay gate pairs of (u, h) instead of the reader (the a|b tile and the head's dt_bias /
// -exp(A) read here, then the software-fp32 gate math of gdn_step_gate_math.hpp) and hands the compute the 2*T fp32
// bit patterns in one small `gsc` entry (beta[t] at word t, decay[t] at word T + t); the compute fills its DST tiles
// from the same bits. The qkv / dt_bias / -exp(A) geometry arrives as defines (GDN_AB_PAGE, GDN_W_TILES,
// GDN_DTB_FP32, GDN_NEA_FP32). GDN_GSC_SPLIT (bit 6, VB > 1): the head's blocks split the T gate computations and swap
// one 64 B slot per token (semaphore `gsv`). GDN_NARROW_GR (bit 10): the gate reads fetch only the face rows they use.
// GDN_HIST_PAGED: instead of the window, the convolution history after every token goes to gdn_decode_step's packed
// history pages (slots 1..3 at ring_page_idx[u][T*Nv + t*Nv + h] + 1..3), from the compute's window tiles: packed tile
// P_j (rows 2c and 2c + 1 = window row j of chunk c, other rows 0) is built once per window row j = 1..Lw-1 in the
// bounce buffer, and slot i after token t is P_{t+i}. Held heads write no history. With value blocks, every block
// reads its start history from the start block's pages, which a destination may alias, so no block writes bytes that
// another block of the head may still read: with GDN_QKSHARE each block reads and writes only its own chunks' rows
// (block 0 also the zero rows past the head's chunks); without it block 0 writes the whole packed tiles once the
// `valid` handshake shows that every block of the head has finished its token loop (and so its reads).
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"

#ifndef GDN_HIST_PAGED
#define GDN_HIST_PAGED 0
#endif
#ifndef GDN_VSPLIT
#define GDN_VSPLIT 0
#endif
#ifndef GDN_GATE_SCALARS
#define GDN_GATE_SCALARS 0
#endif
#ifndef GDN_QKSHARE
#define GDN_QKSHARE 0
#endif
#ifndef GDN_GSC_SPLIT
#define GDN_GSC_SPLIT 0
#endif
#ifndef GDN_NARROW_GR
#define GDN_NARROW_GR 0
#endif
#if GDN_GATE_SCALARS
#define GDN_GATE_NOINLINE 1  // code size; see gdn_step_gate_math.hpp
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_gate_math.hpp"
#endif
#if GDN_VSPLIT
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#endif

using namespace gdn_step_df;

namespace {
// Rows [r0, r0 + nr) of `count` L1 tiles to the same rows of the destination tiles, ONE face-row (32 B for bf16,
// 64 B for fp32) per NoC write; source and destination offsets are equal, so both sit in the same alignment class
// whatever the row parity.
template <typename Accessor>
inline void write_rows_single(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, uint32_t r0, uint32_t nr) {
    dfb.wait_front(count);
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t esz = entry / 1024;
    const uint32_t seg = entry / 64;
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t base = t * entry;
        for (uint32_t r = r0; r < r0 + nr; ++r) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = tile_elem_index(r, half * 16) * esz;
                noc.async_write(
                    dfb, acc, seg, {.offset_bytes = base + off}, {.page_id = first_page + t, .offset_bytes = off});
            }
        }
    }
    noc.async_write_barrier();
    dfb.pop_front(count);
}

// Packed history tiles P_j, j = 1 .. n_rows, into dst tiles j - 1 (zeroed by the caller): rows 2g and 2g + 1 of P_j
// take row j of window tile c, g = head_chunk(c), for every chunk c < n_chunks. bf16: one face row is 16 values, 8
// words.
// Plain loads and stores, 8 loads then 16 stores per face row (volatile accesses would serialize them, and the copy
// delays the writer's first ring write); the barrier keeps the stores ahead of the NoC writes that read them.
template <typename ChunkMap>
inline void pack_window_rows(
    DataflowBuffer& win, DataflowBuffer& dst, uint32_t n_chunks, uint32_t n_rows, ChunkMap head_chunk) {
    auto src_lock = win.scoped_read_lock(n_chunks);
    const uint32_t* src = &src_lock.template get_ptr<uint32_t>()[0];
    auto dst_lock = dst.scoped_write_lock(n_rows);
    uint32_t* out = &dst_lock.template get_ptr<uint32_t>()[0];
    const uint32_t words = win.get_entry_size() / 4;
    for (uint32_t j = 1; j <= n_rows; ++j) {
        for (uint32_t c = 0; c < n_chunks; ++c) {
            const uint32_t g = head_chunk(c);
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t* from = src + c * words + tile_elem_index(j, 16 * half) / 2;
                uint32_t* to0 = out + (j - 1) * words + tile_elem_index(2 * g, 16 * half) / 2;
                uint32_t* to1 = to0 + 8;  // row 2g + 1: the next face row
                uint32_t v[8];
#pragma GCC unroll 8
                for (uint32_t w = 0; w < 8; ++w) {
                    v[w] = from[w];
                }
#pragma GCC unroll 8
                for (uint32_t w = 0; w < 8; ++w) {
                    to0[w] = v[w];
                    to1[w] = v[w];
                }
            }
        }
    }
    asm volatile("" ::: "memory");
}

#if GDN_HIST_PAGED && GDN_QKSHARE
// Rows [r0, r1) (both even) of slot `src_tile` into the same rows of tile `page`, one face-row span per tile half and
// face pair: every destination address is 64 B aligned and the head's blocks write disjoint 64 B units.
template <typename Accessor>
inline void write_row_span(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t src_tile, uint32_t r0, uint32_t r1) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t esz = entry / 1024;
    const uint32_t seg = entry / 64;  // one face row
    for (uint32_t lo = r0; lo < r1;) {
        const uint32_t hi = (lo < 16) ? (r1 < 16 ? r1 : 16) : r1;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t off = tile_elem_index(lo, half * 16) * esz;
            noc.async_write(
                dfb,
                acc,
                seg * (hi - lo),
                {.offset_bytes = src_tile * entry + off},
                {.page_id = page, .offset_bytes = off});
        }
        lo = hi;
    }
}
#endif

template <typename Accessor>
inline void write_whole_tile(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t src_tile) {
    const uint32_t entry = dfb.get_entry_size();
    noc.async_write(dfb, acc, entry, {.offset_bytes = src_tile * entry}, {.page_id = page});
}

#if GDN_NARROW_GR
// Rows [r0, r0 + n) of tile `page` into the same rows of slot `dst_tile`, one face-row segment (16 elements) per row
// and tile half (left: columns 0..15, right: 16..31). Every core's gate reads hit the same few DRAM pages (the users'
// a|b tile row, tile 0 of dt_bias / -exp(A)), and whole 2-4 KB pages for a few elements serialize on one bank.
// Source and destination offsets are equal; esz is the source's element size.
template <typename Accessor>
inline void read_rows_at(
    const Accessor& acc,
    DataflowBuffer& dfb,
    Noc& noc,
    uint32_t page,
    uint32_t dst_tile,
    uint32_t r0,
    uint32_t n,
    bool left,
    bool right,
    uint32_t esz) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = 16 * esz;
    for (uint32_t r = r0; r < r0 + n; ++r) {
        for (uint32_t half = 0; half < 2; ++half) {
            if ((half == 0 && !left) || (half == 1 && !right)) {
                continue;
            }
            const uint32_t off = tile_elem_index(r, half * 16) * esz;
            noc.async_read(
                acc, dfb, seg, {.page_id = page, .offset_bytes = off}, {.offset_bytes = dst_tile * entry + off});
        }
    }
}
#endif
}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t T,
    uint32_t B,
    uint32_t Lw,
    uint32_t Ct,
    uint32_t out_w_tiles,
    uint32_t ctrl_bytes,
    uint32_t hold_sentinel,
    uint32_t VB>
TT_KERNEL void writer(
    uint32_t u,
    uint32_t h,
    uint32_t s,
    uint32_t peer_x0,
    uint32_t peer_y0,
    uint32_t peer_x1,
    uint32_t peer_y1,
    uint32_t peer_x2,
    uint32_t peer_y2,
    uint32_t peer_x3,
    uint32_t peer_y3) {
    const auto ring_acc = TensorAccessor(tensor::ring_out);
#ifdef GDN_RING_POOL2
    const auto ring2_acc = same_layout_accessor(ring_acc, tensor::ring2_out);
#endif
    const auto out_acc = TensorAccessor(tensor::out);
#if GDN_HIST_PAGED
    const auto hist_acc = TensorAccessor(tensor::hist_w);
#else
    const auto wa_acc = TensorAccessor(tensor::win_a_w);
    const auto wb_acc = TensorAccessor(tensor::win_b_w);
#endif
    const auto ctrl_acc = TensorAccessor(tensor::ctrl_w);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    DataflowBuffer wout(dfb::wout);
    DataflowBuffer ctrl_w(dfb::ctrl_w);
    DataflowBuffer bounce(dfb::bounce);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ch = 2 * Kt + Vt;
    static_assert(VB >= 1 && VB <= 4 && Vt % VB == 0, "value blocks must divide Vt (at most 4 peers)");
    static_assert((VB > 1) == (GDN_VSPLIT != 0), "GDN_VSPLIT must be defined exactly when VB > 1");
    constexpr uint32_t Vs = Vt / VB;  // this block's state / output column tiles
    constexpr uint32_t KVs = Kt * Vs;
    const uint32_t v0 = s * Vs;  // first column tile of this block
    // QKSHARE: every block writes the window rows of its own ChA chunks (q/k share [s*Qk, (s+1)*Qk), then its v
    // chunks); otherwise block 0 writes the head's window rows (ChA = Ch, block order = head order)
    constexpr uint32_t Qk = GDN_QKSHARE ? (2 * Kt) / VB : 0u;
    constexpr uint32_t ChA = GDN_QKSHARE ? Qk + Vs : Ch;
    const bool window_writer = GDN_QKSHARE ? true : (s == 0);
    auto blk_chunk = [&](uint32_t ci) -> uint32_t {
        if constexpr (GDN_QKSHARE) {
            return ci < Qk ? s * Qk + ci : 2 * Kt + v0 + (ci - Qk);
        } else {
            return ci;
        }
    };
    constexpr uint32_t rf = Nv / Nk;
    constexpr uint32_t BH = B * Nv;
    constexpr uint32_t Lw_w = (Lw + 1) & ~1u;  // even row count (row Lw is an exact zero when Lw is odd)
    const uint32_t hk = h / rf;
    const uint32_t bh = u * Nv + h;
    const uint32_t tr = (u * T) / 32;  // output tile row of the user's rows
    const uint32_t r0 = (u * T) % 32;
#if !GDN_HIST_PAGED
    const bool write_qk = (h % rf) == 0;  // the rf value heads of a key head share its q/k chunks: one writer
    auto chunk_col = [&](uint32_t c) -> uint32_t {
        if (c < Kt) {
            return hk * Kt + c;
        }
        if (c < 2 * Kt) {
            return Nk * Kt + hk * Kt + (c - Kt);
        }
        return 2 * Nk * Kt + h * Vt + (c - 2 * Kt);
    };
    auto owns = [&](uint32_t c) -> bool { return c >= 2 * Kt || write_qk; };
#endif

    // ---- ctrl page: parity and HOLD
    ctrl_w.reserve_back(1);
    noc.async_read(ctrl_acc, ctrl_w, ctrl_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
#ifdef GDN_RING_PAGE_IDX
    // row u of ring_page_idx: word t*Nv + h = first ring page of token t's state of head h (reserved, never pushed)
    const auto ridx_acc = TensorAccessor(tensor::ridx);
    DataflowBuffer ridx(dfb::ridx_w);
    ridx.reserve_back(1);
    noc.async_read(ridx_acc, ridx, ridx.get_entry_size(), {.page_id = u, .offset_bytes = 0}, {.offset_bytes = 0});
#endif
    noc.async_read_barrier();
    uint32_t par, blk;
    {
        auto lock = ctrl_w.scoped_write_lock(1);
        auto p = lock.template get_ptr<volatile uint32_t>();
        par = p[0] & 1u;
        blk = p[1 + B + bh];
    }
    ctrl_w.push_back(1);
    ctrl_w.wait_front(1);
    ctrl_w.pop_front(1);
    const bool hold = (blk == hold_sentinel);

#if GDN_VSPLIT
    // acc_x is this program's buffer from launch on: reserve it now (same L1 address on every block: same spec, first
    // use) and tell every block of the head it may write into it
    const uint32_t peer_x[4] = {peer_x0, peer_x1, peer_x2, peer_x3};
    const uint32_t peer_y[4] = {peer_y0, peer_y1, peer_y2, peer_y3};
    DataflowBuffer acc_mine(dfb::acc_mine);
    DataflowBuffer acc_x(dfb::acc_x);
    Semaphore ready(sem::ready);
    Semaphore valid(sem::valid);
    acc_x.reserve_back(Vt);
    const uint32_t acc_x_base = acc_x.get_write_ptr();
#if GDN_GSC_SPLIT
    // gsc is filled by the head's blocks (each writes its tokens' 64 B slots into every block): reserved before `ready`
    DataflowBuffer gsc_x(dfb::gsc);
    gsc_x.reserve_back(1);
    const uint32_t gsc_base = gsc_x.get_write_ptr();
#endif
#if GDN_QKSHARE
    // qc / kc are filled by the head's blocks (each writes its Qk conv tiles into every block): reserved before
    // `ready` goes up, at the same L1 address on every block (same spec, first use), like acc_x
    DataflowBuffer qc_x(dfb::qc);
    DataflowBuffer kc_x(dfb::kc);
    qc_x.reserve_back(Kt);
    kc_x.reserve_back(Kt);
    const uint32_t qc_base = qc_x.get_write_ptr();
    const uint32_t kc_base = kc_x.get_write_ptr();
#endif
    for (uint32_t p = 0; p < VB; ++p) {
        ready.up(noc, peer_x[p], peer_y[p], 1);
    }
#else
    (void)peer_x0;
    (void)peer_y0;
    (void)peer_x1;
    (void)peer_y1;
    (void)peer_x2;
    (void)peer_y2;
    (void)peer_x3;
    (void)peer_y3;
#endif

#if GDN_GATE_SCALARS
    // ---- the T gate pairs of (u, h) -> gsc (the reader's fast_mode gate reads and math, moved here)
    {
        const auto qkv_acc = TensorAccessor(tensor::qkv_w);
        const auto dtb_acc = TensorAccessor(tensor::dtb_w);
        const auto nea_acc = TensorAccessor(tensor::nea_w);
        DataflowBuffer ab_w(dfb::ab_w);
        DataflowBuffer dtb_ws(dfb::dtb_ws);
        DataflowBuffer nea_ws(dfb::nea_ws);
#if GDN_GSC_SPLIT
        // this block computes tokens [t0, t1) only, a contiguous share of ceil(T / VB), into its send buffer gss (one
        // 16-word slot per token: beta at word 16*t, dec at 16*t + 1); then every block's gsc receives them
        constexpr uint32_t Tb = (T + VB - 1) / VB;
        const uint32_t t0 = (s * Tb < T) ? s * Tb : T;
        const uint32_t t1 = (t0 + Tb < T) ? t0 + Tb : T;
        DataflowBuffer gsc(dfb::gss);
#else
        constexpr uint32_t t0 = 0;
        constexpr uint32_t t1 = T;
        DataflowBuffer gsc(dfb::gsc);
#endif
        const uint32_t pg = tr * GDN_W_TILES;
        // staging only (never pushed): the a|b tile and the head scalars' tile 0, all issued under one barrier
        ab_w.reserve_back(1);
        dtb_ws.reserve_back(1);
        nea_ws.reserve_back(1);
#if GDN_NARROW_GR
        // only the user's T rows of the a|b tile (both halves: a at column h, b at column Nv + h) and tile 0's face row
        // of dt_bias / -exp(A) holding column h
        read_rows_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE, 0, r0, T, true, true, 2);
        read_rows_at(dtb_acc, dtb_ws, noc, 0, 0, 0, 1, h < 16, h >= 16, GDN_DTB_FP32 ? 4 : 2);
        read_rows_at(nea_acc, nea_ws, noc, 0, 0, 0, 1, h < 16, h >= 16, GDN_NEA_FP32 ? 4 : 2);
#else
        read_tiles_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE, 1, 0);
        noc.async_read(dtb_acc, dtb_ws, GDN_DTB_FP32 ? 4096u : 2048u, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read(nea_acc, nea_ws, GDN_NEA_FP32 ? 4096u : 2048u, {.page_id = 0}, {.offset_bytes = 0});
#endif
        noc.async_read_barrier();
        uint32_t dtb_bits, nea_bits;
        {
            auto lock = dtb_ws.scoped_write_lock(1);
            if constexpr (GDN_DTB_FP32 != 0) {
                dtb_bits = lock.template get_ptr<volatile uint32_t>()[tile_elem_index(0, h)];
            } else {
                const uint32_t b16 = lock.template get_ptr<volatile uint16_t>()[tile_elem_index(0, h)];
                dtb_bits = b16 << 16;
            }
        }
        {
            auto lock = nea_ws.scoped_write_lock(1);
            if constexpr (GDN_NEA_FP32 != 0) {
                nea_bits = lock.template get_ptr<volatile uint32_t>()[tile_elem_index(0, h)];
            } else {
                const uint32_t b16 = lock.template get_ptr<volatile uint16_t>()[tile_elem_index(0, h)];
                nea_bits = b16 << 16;
            }
        }
        gsc.reserve_back(1);
        {
            auto ab_lock = ab_w.scoped_write_lock(1);
            auto p16 = ab_lock.template get_ptr<volatile uint16_t>();
            auto g_lock = gsc.scoped_write_lock(1);
            auto g32 = g_lock.template get_ptr<volatile uint32_t>();
            for (uint32_t t = t0; t < t1; ++t) {
                const uint32_t a_bits = static_cast<uint32_t>(p16[tile_elem_index(r0 + t, h)]) << 16;
                const uint32_t b_bits = static_cast<uint32_t>(p16[tile_elem_index(r0 + t, Nv + h)]) << 16;
                uint32_t beta_bits, dec_bits;
                gate_beta_decay_bits(a_bits, b_bits, dtb_bits, nea_bits, beta_bits, dec_bits);
#if GDN_GSC_SPLIT
                g32[16 * t] = beta_bits;
                g32[16 * t + 1] = dec_bits;
#else
                g32[t] = beta_bits;
                g32[T + t] = dec_bits;
#endif
            }
        }
        gsc.push_back(1);
#if GDN_GSC_SPLIT
        // slots [t0, t1) (64 B aligned) into every block's gsc (itself included); `gsv` up on every block; once all VB
        // blocks' slots are in, gsc goes to the compute
        {
            Semaphore gsv(sem::gsv);
            gsc.wait_front(1);
            ready.wait(VB);  // every block of the head runs this program and has reserved its gsc
            if (t1 > t0) {
                for (uint32_t p = 0; p < VB; ++p) {
                    noc.async_write(
                        gsc,
                        UnicastEndpoint{},
                        (t1 - t0) * 64,
                        {.offset_bytes = t0 * 64},
                        {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = gsc_base + t0 * 64});
                }
                noc.async_write_barrier();  // the slots landed before `gsv` is raised
            }
            gsc.pop_front(1);
            for (uint32_t p = 0; p < VB; ++p) {
                gsv.up(noc, peer_x[p], peer_y[p], 1);
            }
            gsv.wait(VB);
            gsc_x.push_back(1);
        }
#endif
    }
#endif

#if GDN_HIST_PAGED
    // ---- history: slots 1..3 after every token into the pages ring_page_idx names (held heads: nothing). Called
    //      where every other block of the head is done reading the start block's history (see the header).
    (void)par;
    auto write_history = [&]() {
        constexpr uint32_t NP = Lw - 1;  // packed tiles P_1 .. P_{Lw-1}
        wout.wait_front(ChA);
        if (!hold && window_writer) {
            bounce.reserve_back(NP);
            zero_reserved(bounce, noc, NP);
            pack_window_rows(wout, bounce, ChA, NP, blk_chunk);
            bounce.push_back(NP);
            bounce.wait_front(NP);
            for (uint32_t t = 0; t < T; ++t) {
                uint32_t page0;
                {
                    auto lock = ridx.scoped_write_lock(1);
                    page0 = lock.template get_ptr<volatile uint32_t>()[T * Nv + t * Nv + h];
                }
                for (uint32_t slot = 1; slot < 4; ++slot) {
#if GDN_QKSHARE
                    const uint32_t src = t + slot - 1;
                    write_row_span(hist_acc, bounce, noc, page0 + slot, src, 2 * s * Qk, 2 * (s + 1) * Qk);
                    write_row_span(hist_acc, bounce, noc, page0 + slot, src, 2 * (2 * Kt + v0), 2 * (2 * Kt + v0 + Vs));
                    if (s == 0 && 2 * Ch < 32) {
                        write_row_span(hist_acc, bounce, noc, page0 + slot, src, 2 * Ch, 32);
                    }
#else
                    write_whole_tile(hist_acc, bounce, noc, page0 + slot, t + slot - 1);
#endif
                }
            }
            noc.async_write_barrier();
            bounce.pop_front(NP);
        }
        wout.pop_front(ChA);
    };
#else
    // ---- window: win[1 - par] <- rebuilt rows (or the held user's rows copied through)
    if (!window_writer) {
        wout.wait_front(ChA);  // value blocks other than block 0: block 0 writes the head's window rows
        wout.pop_front(ChA);
    } else if (hold) {
        bounce.reserve_back(Ch);  // the whole ring (only the first ChA slots are used): keeps its pointer aligned
        for (uint32_t c = 0; c < ChA; ++c) {
            if (!owns(blk_chunk(c))) {
                continue;
            }
            const uint32_t page = u * Ct + chunk_col(blk_chunk(c));
            if (par) {
                read_tiles_at(wb_acc, bounce, noc, page, 1, c);
            } else {
                read_tiles_at(wa_acc, bounce, noc, page, 1, c);
            }
        }
        noc.async_read_barrier();
        bounce.push_back(Ch);
        bounce.wait_front(Ch);
        for (uint32_t c = 0; c < ChA; ++c) {
            if (!owns(blk_chunk(c))) {
                continue;
            }
            const uint32_t page = u * Ct + chunk_col(blk_chunk(c));
            if (par) {
                write_whole_tile(wa_acc, bounce, noc, page, c);
            } else {
                write_whole_tile(wb_acc, bounce, noc, page, c);
            }
        }
        noc.async_write_barrier();
        bounce.pop_front(Ch);
        wout.wait_front(ChA);  // drain the compute's window tiles without writing them
        wout.pop_front(ChA);
    } else {
        for (uint32_t c = 0; c < ChA; ++c) {
            if (!owns(blk_chunk(c))) {
                wout.wait_front(1);
                wout.pop_front(1);
                continue;
            }
            const uint32_t page = u * Ct + chunk_col(blk_chunk(c));
            if (par) {
                write_rows(wa_acc, wout, noc, page, 1, 0, Lw_w);
            } else {
                write_rows(wb_acc, wout, noc, page, 1, 0, Lw_w);
            }
        }
    }
#endif

#if GDN_QKSHARE
    // ---- the q/k conv tiles: this block's Qk tiles (head q/k chunks [s*Qk, (s+1)*Qk), all in q or all in k) go to the
    //      same tile offset of qc / kc on every block of the head (itself included), exact fp32 bytes, unicast; then
    //      `qkv` goes up on every block and, once all VB blocks' tiles are in, qc / kc are handed to the compute
    {
        DataflowBuffer qkm(dfb::qkm);
        Semaphore qkv(sem::qkv);
        qkm.wait_front(Qk);
        const uint32_t entry = qkm.get_entry_size();
        const uint32_t first = s * Qk;
        const uint32_t dst = (first < Kt) ? qc_base + first * entry : kc_base + (first - Kt) * entry;
        ready.wait(VB);  // every block of the head runs this program and has reserved its qc / kc
        for (uint32_t p = 0; p < VB; ++p) {
            noc.async_write(
                qkm,
                UnicastEndpoint{},
                Qk * entry,
                {.offset_bytes = 0},
                {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = dst});
        }
        noc.async_write_barrier();  // the tiles landed before `qkv` is raised
        qkm.pop_front(Qk);
        for (uint32_t p = 0; p < VB; ++p) {
            qkv.up(noc, peer_x[p], peer_y[p], 1);
        }
        qkv.wait(VB);  // every block's q/k tiles are in this block's qc / kc
        qc_x.push_back(Kt);
        kc_x.push_back(Kt);
    }
#endif

#if GDN_HIST_PAGED && !(GDN_VSPLIT && !GDN_QKSHARE)
    write_history();
#endif

    // ---- output padding rows: the last user's cores zero rows [r0 + T, 32) of the head's Vt output tiles from a
    //      zeroed bounce region (4 KiB covers every in-tile offset of an fp32 tile). Equal source / destination
    //      offsets keep both addresses in the same alignment class, as write_rows / write_rows_single do.
    if (u == B - 1 && r0 + T < 32) {
        bounce.reserve_back(2);
        zero_reserved(bounce, noc, 2);
        bounce.push_back(2);
        bounce.wait_front(2);
        const uint32_t entry = out.get_entry_size();
        const uint32_t esz = entry / 1024;
        const uint32_t seg = entry / 64;
        for (uint32_t t = 0; t < Vs; ++t) {
            const uint32_t page = tr * out_w_tiles + h * Vt + v0 + t;
            for (uint32_t lo = r0 + T; lo < 32;) {
                const uint32_t hi = (lo < 16) ? 16u : 32u;
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t off = tile_elem_index(lo, half * 16) * esz;
                    noc.async_write(
                        bounce,
                        out_acc,
                        seg * (hi - lo),
                        {.offset_bytes = off},
                        {.page_id = page, .offset_bytes = off});
                }
                lo = hi;
            }
        }
        noc.async_write_barrier();
        bounce.pop_front(2);
    }

    // ---- ring: the post-token state of every token, skipped for held heads. This block's tile (i, j) of hnew (index
    //      i*Vs + j) is tile (i, v0 + j) of the [Kt, Vt] state
    {
        const uint32_t entry = hnew.get_entry_size();
        for (uint32_t t = 0; t < T; ++t) {
            hnew.wait_front(KVs);
            if (!hold) {
#ifdef GDN_RING_PAGE_IDX
                uint32_t page0;
                {
                    auto lock = ridx.scoped_write_lock(1);
                    page0 = lock.template get_ptr<volatile uint32_t>()[t * Nv + h] + v0;
                }
#else
                const uint32_t page0 = (t * BH + bh) * KV + v0;
#endif
                auto write_state = [&](const auto& acc, uint32_t first) {
                    for (uint32_t i = 0; i < Kt; ++i) {
                        for (uint32_t j = 0; j < Vs; ++j) {
                            noc.async_write(
                                hnew,
                                acc,
                                entry,
                                {.offset_bytes = (i * Vs + j) * entry},
                                {.page_id = first + i * Vt + j});
                        }
                    }
                };
#ifdef GDN_RING_POOL2
                const bool in2 = page0 >= GDN_RING_POOL2;
                write_state(in2 ? ring2_acc : ring_acc, in2 ? page0 - GDN_RING_POOL2 : page0);
#else
                write_state(ring_acc, page0);
#endif
                noc.async_write_barrier();
            }
            hnew.pop_front(KVs);
        }
    }

#if GDN_VSPLIT
    // ---- swap the last token's pre-norm output tiles within the head: acc_mine -> acc_x[v0 ..] of every block
    {
        acc_mine.wait_front(Vs);
        const uint32_t entry = acc_mine.get_entry_size();
        ready.wait(VB);  // every block of the head runs this program (its acc_x is ours to write)
        for (uint32_t p = 0; p < VB; ++p) {
            noc.async_write(
                acc_mine,
                UnicastEndpoint{},
                Vs * entry,
                {.offset_bytes = 0},
                {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = acc_x_base + v0 * entry});
        }
        noc.async_write_barrier();  // the tiles landed before `valid` is raised
        acc_mine.pop_front(Vs);
        for (uint32_t p = 0; p < VB; ++p) {
            valid.up(noc, peer_x[p], peer_y[p], 1);
        }
        valid.wait(VB);  // every block's tiles are in this block's acc_x
        acc_x.push_back(Vt);
    }
#endif

#if GDN_HIST_PAGED && GDN_VSPLIT && !GDN_QKSHARE
    write_history();  // after `valid`: every block of the head has read its start history
#endif

    // ---- the user's T output rows of this block's Vs output tiles
    if constexpr ((T % 2) == 0) {
        write_rows(out_acc, out, noc, tr * out_w_tiles + h * Vt + v0, Vs, r0, T);
    } else {
        write_rows_single(out_acc, out, noc, tr * out_w_tiles + h * Vt + v0, Vs, r0, T);
    }
#if GDN_VSPLIT
    noc.async_atomic_barrier();  // retire the ready / valid increments before exit
#endif
}
