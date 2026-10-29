// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer: one core = (user u, value head h). Reads the ctrl page (parity, HOLD). Window: rows 0..Lw-1 (rounded up
// to an even count; the extra row is an exact zero) of the compute's rebuilt window tiles go to win[1-par] (q/k
// chunks only from heads with h % rf == 0, v chunks from every head); a held user's tiles are copied through
// win[par] -> win[1-par] via a bounce buffer and the compute's tiles are drained. Ring: per token the post-token
// state (16 fp32 tiles) from the buffered `hnew` to block (t*BH + bh), skipped for held heads. Then the user's T
// output rows (whole face-row spans when T is even, 32 B single rows at T = 1). The output tensor is not zero-filled
// at creation, so the cores of the last user (u = B-1) also write exact zeros into rows [B*T % 32, 32) of the last
// tile row (before the ring stream, while the writer is idle) -- with R <= round_up(B*T, 32) enforced by the host,
// every output row outside [u*T, u*T + T) is then exactly 0. HOLD is decided per (u,h) ctrl word; q/k window chunk
// ownership is per key-head group (h % rf == 0), so a caller must hold ALL Nv heads of a user (the op cannot check
// ctrl contents).
// VB > 1 (value blocks, GDN_VSPLIT): core (u, h, s) owns state / output column tiles [s*Vs, (s+1)*Vs). Block 0 alone
// writes (or copies through) the window; each block writes its KVs ring tiles (i, s*Vs + j) per token, zeroes / writes
// only its Vs output tiles, and before that swaps the last token's pre-norm output tiles with its head's other blocks:
// acc_mine (Vs fp32 tiles, exact bytes) -> acc_x[s*Vs ..] on every block (itself included) by unicast NoC writes.
// Handshake (both semaphores start at 0 on every launch): each block raises `ready` on every block once it runs (its
// acc_x is reserved), waits for VB before writing, write-barriers, then raises `valid` on every block and waits for VB
// before handing acc_x to the compute. No multicast; at most 2 * VB atomics and VB writes per core.
// GDN_GATE_SCALARS (kernel_opts bit 1, fast_mode + fp32_state): this writer, idle until the compute's phase A1 hands it
// the window tiles, computes the T beta / decay gate pairs of (u, h) instead of the reader -- the a|b tile(s) and the
// head's dt_bias / -exp(A) read here, then reader_gdn_decode_step_conv_fast.cpp's software-fp32 gate math
// (gdn_step_gate_math.hpp, bit for bit) -- and hands the compute the 2*T fp32 bit patterns in ONE small `gsc` entry
// (beta[t] at word t, decay[t] at word T + t) instead of 2*T broadcast tiles: the compute fills its DST tiles from the
// same bits (fill_tile_bitcast). The extra qkv / dt_bias / -exp(A) geometry arrives as defines (GDN_AB_PAGE,
// GDN_W_TILES, GDN_DTB_FP32, GDN_NEA_FP32), so the named-argument signature is the same in every mode.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "api/debug/kernel_profiler.hpp"  // zones compile to nothing unless the build defines PROFILE_KERNEL

#ifndef GDN_VSPLIT
#define GDN_VSPLIT 0
#endif
#ifndef GDN_GATE_SCALARS
#define GDN_GATE_SCALARS 0
#endif
#ifndef GDN_QKSHARE
#define GDN_QKSHARE 0  // kernel_opts bit 4 (VB > 1): per-block window ownership + the q/k conv tile exchange
#endif
#ifndef GDN_GSC_SPLIT
#define GDN_GSC_SPLIT 0  // kernel_opts bit 6 (GATE_SCALARS, VB > 1): the head's blocks split the T gate computations
#endif
#if GDN_GATE_SCALARS
#define GDN_GATE_NOINLINE 1  // code size; see gdn_step_gate_math.hpp
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_spec_step/device/kernels/dataflow/gdn_step_gate_math.hpp"
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
// whatever the row parity (validated in the M0a scratch for fp32 outputs; bf16 is tested here).
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

template <typename Accessor>
inline void write_whole_tile(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t src_tile) {
    const uint32_t entry = dfb.get_entry_size();
    noc.async_write(dfb, acc, entry, {.offset_bytes = src_tile * entry}, {.page_id = page});
}
}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t T,
    uint32_t B,
    uint32_t M,  // ring/ctrl layout width (>= B)
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
    const auto out_acc = TensorAccessor(tensor::out);
    const auto wa_acc = TensorAccessor(tensor::win_a_w);
    const auto wb_acc = TensorAccessor(tensor::win_b_w);
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
    constexpr uint32_t BH = M * Nv;
    constexpr uint32_t Lw_w = (Lw + 1) & ~1u;  // even row count (row Lw is an exact zero when Lw is odd)
    const uint32_t hk = h / rf;
    const uint32_t bh = u * Nv + h;
    const uint32_t tr = (u * T) / 32;  // output tile row of the user's rows
    const uint32_t r0 = (u * T) % 32;
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

    // ---- ctrl page: parity and HOLD
    ctrl_w.reserve_back(1);
    noc.async_read(ctrl_acc, ctrl_w, ctrl_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    uint32_t par, blk;
    {
        auto lock = ctrl_w.scoped_write_lock(1);
        auto p = lock.template get_ptr<volatile uint32_t>();
        par = p[0] & 1u;
        blk = p[1 + M + bh];
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
    // ---- the T gate pairs of (u, h) -> gsc (the reader's FAST-mode gate block, moved here: same reads, same math)
    {
        DeviceZoneScopedN("wr_gates");
        const auto qkv_acc = TensorAccessor(tensor::qkv_w);
        const auto dtb_acc = TensorAccessor(tensor::dtb_w);
        const auto nea_acc = TensorAccessor(tensor::nea_w);
        DataflowBuffer ab_w(dfb::ab_w);
        DataflowBuffer dtb_ws(dfb::dtb_ws);
        DataflowBuffer nea_ws(dfb::nea_ws);
#if GDN_GSC_SPLIT
        // GSC_SPLIT: this block computes tokens [t0, t1) only (contiguous share of ceil(T / VB)) into its send buffer
        // gss (slot t = 16 words at word 16*t: beta, dec), then every block's gsc receives them
        constexpr uint32_t Tb = (T + VB - 1) / VB;
        const uint32_t t0 = (s * Tb < T) ? s * Tb : T;
        const uint32_t t1 = (t0 + Tb < T) ? t0 + Tb : T;
        DataflowBuffer gsc(dfb::gss);
#else
        constexpr uint32_t t0 = 0;
        constexpr uint32_t t1 = T;
        DataflowBuffer gsc(dfb::gsc);
#endif
        constexpr uint32_t AB2 = (2 * Nv > 32) ? 1u : 0u;  // the factory's rule: the a|b block spans two tiles
        // b_idx: head h's b column lies in a different tile than its a column (the factory's per-core rule)
        const uint32_t b_tile = (AB2 != 0 && ((Nv + h) >> 5) != (h >> 5)) ? 1u : 0u;
        const uint32_t pg = tr * GDN_W_TILES;
        // staging only (never pushed): the a|b tile(s) and the head scalars' tile 0, all issued under one barrier
        ab_w.reserve_back(1 + AB2);
        read_tiles_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + (h >> 5), 1, 0);
        if (b_tile != 0) {
            read_tiles_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + ((Nv + h) >> 5), 1, 1);
        }
        dtb_ws.reserve_back(1);
        nea_ws.reserve_back(1);
        noc.async_read(dtb_acc, dtb_ws, GDN_DTB_FP32 ? 4096u : 2048u, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read(nea_acc, nea_ws, GDN_NEA_FP32 ? 4096u : 2048u, {.page_id = 0}, {.offset_bytes = 0});
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
            auto ab_lock = ab_w.scoped_write_lock(1 + AB2);
            auto p16 = ab_lock.template get_ptr<volatile uint16_t>();
            auto g_lock = gsc.scoped_write_lock(1);
            auto g32 = g_lock.template get_ptr<volatile uint32_t>();
            for (uint32_t t = t0; t < t1; ++t) {
                const uint32_t a_bits = static_cast<uint32_t>(p16[tile_elem_index(r0 + t, h & 31u)]) << 16;
                const uint32_t b_bits =
                    static_cast<uint32_t>(p16[b_tile * 1024 + tile_elem_index(r0 + t, (Nv + h) & 31u)]) << 16;
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

    // ---- window: win[1 - par] <- rebuilt rows (or the held user's rows copied through)
    {
        DeviceZoneScopedN("wr_win");
        if (!window_writer) {
            wout.wait_front(ChA);  // value blocks > 0: block 0 writes the head's window rows
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
    }

#if GDN_QKSHARE
    // ---- the q/k conv tiles: this block's Qk tiles (head q/k chunks [s*Qk, (s+1)*Qk), all in q or all in k) go to the
    //      same tile offset of qc / kc on every block of the head (itself included), exact fp32 bytes, unicast; then
    //      `qkv` goes up on every block and, once all VB blocks' tiles are in, qc / kc are handed to the compute
    {
        DeviceZoneScopedN("wr_qkx");
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

    // ---- ring: the post-token state of every token, token-major block (t*BH + bh); skipped for held heads. This
    //      block's tile (i, j) of hnew (index i*Vs + j) is tile (i, v0 + j) of the [Kt, Vt] block (VB = 1: tile i*Vt +
    //      j)
    {
        const uint32_t entry = hnew.get_entry_size();
        for (uint32_t t = 0; t < T; ++t) {
            {
                DeviceZoneScopedN("wr_wait_hnew");
                hnew.wait_front(KVs);
            }
            DeviceZoneScopedN("wr_ring");
            if (!hold) {
                const uint32_t page0 = (t * BH + bh) * KV + v0;
                for (uint32_t i = 0; i < Kt; ++i) {
                    for (uint32_t j = 0; j < Vs; ++j) {
                        noc.async_write(
                            hnew,
                            ring_acc,
                            entry,
                            {.offset_bytes = (i * Vs + j) * entry},
                            {.page_id = page0 + i * Vt + j});
                    }
                }
                noc.async_write_barrier();
            }
            hnew.pop_front(KVs);
        }
    }

#if GDN_VSPLIT
    // ---- swap the last token's pre-norm output tiles within the head: acc_mine -> acc_x[v0 ..] of every block
    {
        DeviceZoneScopedN("wr_xchg");
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

    // ---- the user's T output rows of the head's Vt output tiles
    DeviceZoneScopedN("wr_out");
    if constexpr ((T % 2) == 0) {
        write_rows(out_acc, out, noc, tr * out_w_tiles + h * Vt + v0, Vs, r0, T);
    } else {
        write_rows_single(out_acc, out, noc, tr * out_w_tiles + h * Vt + v0, Vs, r0, T);
    }
#if GDN_VSPLIT
    noc.async_atomic_barrier();  // retire the ready / valid increments before exit
#endif
}
