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
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"

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
    uint32_t Lw,
    uint32_t Ct,
    uint32_t out_w_tiles,
    uint32_t ctrl_bytes,
    uint32_t hold_sentinel>
TT_KERNEL void writer(uint32_t u, uint32_t h) {
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
    constexpr uint32_t rf = Nv / Nk;
    constexpr uint32_t BH = B * Nv;
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
        blk = p[1 + B + bh];
    }
    ctrl_w.push_back(1);
    ctrl_w.wait_front(1);
    ctrl_w.pop_front(1);
    const bool hold = (blk == hold_sentinel);

    // ---- window: win[1 - par] <- rebuilt rows (or the held user's rows copied through)
    if (hold) {
        bounce.reserve_back(Ch);
        for (uint32_t c = 0; c < Ch; ++c) {
            if (!owns(c)) {
                continue;
            }
            const uint32_t page = u * Ct + chunk_col(c);
            if (par) {
                read_tiles_at(wb_acc, bounce, noc, page, 1, c);
            } else {
                read_tiles_at(wa_acc, bounce, noc, page, 1, c);
            }
        }
        noc.async_read_barrier();
        bounce.push_back(Ch);
        bounce.wait_front(Ch);
        for (uint32_t c = 0; c < Ch; ++c) {
            if (!owns(c)) {
                continue;
            }
            const uint32_t page = u * Ct + chunk_col(c);
            if (par) {
                write_whole_tile(wa_acc, bounce, noc, page, c);
            } else {
                write_whole_tile(wb_acc, bounce, noc, page, c);
            }
        }
        noc.async_write_barrier();
        bounce.pop_front(Ch);
        wout.wait_front(Ch);  // drain the compute's window tiles without writing them
        wout.pop_front(Ch);
    } else {
        for (uint32_t c = 0; c < Ch; ++c) {
            if (!owns(c)) {
                wout.wait_front(1);
                wout.pop_front(1);
                continue;
            }
            const uint32_t page = u * Ct + chunk_col(c);
            if (par) {
                write_rows(wa_acc, wout, noc, page, 1, 0, Lw_w);
            } else {
                write_rows(wb_acc, wout, noc, page, 1, 0, Lw_w);
            }
        }
    }

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
        for (uint32_t t = 0; t < Vt; ++t) {
            const uint32_t page = tr * out_w_tiles + h * Vt + t;
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

    // ---- ring: the post-token state of every token, token-major block (t*BH + bh); skipped for held heads
    {
        const uint32_t entry = hnew.get_entry_size();
        for (uint32_t t = 0; t < T; ++t) {
            hnew.wait_front(KV);
            if (!hold) {
                const uint32_t page0 = (t * BH + bh) * KV;
                for (uint32_t i = 0; i < KV; ++i) {
                    noc.async_write(hnew, ring_acc, entry, {.offset_bytes = i * entry}, {.page_id = page0 + i});
                }
                noc.async_write_barrier();
            }
            hnew.pop_front(KV);
        }
    }

    // ---- the user's T output rows of the head's Vt output tiles
    if constexpr ((T % 2) == 0) {
        write_rows(out_acc, out, noc, tr * out_w_tiles + h * Vt, Vt, r0, T);
    } else {
        write_rows_single(out_acc, out, noc, tr * out_w_tiles + h * Vt, Vt, r0, T);
    }
}
