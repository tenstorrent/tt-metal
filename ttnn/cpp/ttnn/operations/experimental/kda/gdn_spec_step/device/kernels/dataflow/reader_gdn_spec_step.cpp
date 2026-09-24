// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader: one core = (user u, value head h). Reads the ctrl page (parity, mi[u], initial ring block of (u,h); HOLD
// sentinel -> block bh, mi 0), the constants, the head's 12 whole window tiles (win[par], user u) + 12 whole raw
// projection tiles + the 12 tap tiles (rows 0..K-1 only), z, the a|b tile(s), the norm weight row, builds the bf16
// one-hot selector tiles, then reads the 64 KiB initial state last (compute's conv/pre overlap it). No per-token reads.
// a|b: a[h] is column h and b[h] column Nv + h past tile ab_page. With 2*Nv <= 32 both sit in tile ab_page (one read,
// ab_in[0]); with Nv = 24 (TP = 2) the a|b block spans two tiles and b's tile is read into ab_in[1] when it differs
// from a's (b_idx = 1, decided by the host per core), so the compute gathers b's row from ab_in[b_idx].
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

using namespace gdn_step_df;

namespace {
template <typename P>
FORCE_INLINE void set_one(P& p, uint32_t tile, uint32_t row, uint32_t col) {
    p[tile * 1024 + tile_elem_index(row, col)] = 0x3F80;  // bf16 1.0 (p: the lock's CoreLocalMem<volatile uint16_t>)
}
}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t T,
    uint32_t B,
    uint32_t K,
    uint32_t Lw,
    uint32_t Ct,
    uint32_t z_tile0,
    uint32_t ab_page,
    uint32_t w_tiles,
    uint32_t dtb_fp32,
    uint32_t nea_fp32,
    uint32_t l2_eps_bits,
    uint32_t norm_eps_bits,
    uint32_t ctrl_bytes,
    uint32_t hold_sentinel>
TT_KERNEL void reader(uint32_t u, uint32_t h, uint32_t b_idx) {
    const auto qkv_acc = TensorAccessor(tensor::qkv);
    const auto wa_acc = TensorAccessor(tensor::win_a);
    const auto wb_acc = TensorAccessor(tensor::win_b);
    const auto ring_acc = TensorAccessor(tensor::ring);
    const auto ctrl_acc = TensorAccessor(tensor::ctrl);
    const auto taps_acc = TensorAccessor(tensor::taps);
    const auto dtb_acc = TensorAccessor(tensor::dtb);
    const auto nea_acc = TensorAccessor(tensor::nea);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer src_in(dfb::src_in);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer ab_in(dfb::ab_in);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer mask_T(dfb::mask_T);
    DataflowBuffer e_t(dfb::e_t);
    DataflowBuffer rsel(dfb::rsel);
    DataflowBuffer csel(dfb::csel);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer ctrl_r(dfb::ctrl_r);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ch = 2 * Kt + Vt;
    constexpr uint32_t rf = Nv / Nk;
    const uint32_t hk = h / rf;
    const uint32_t bh = u * Nv + h;
    const uint32_t tr = (u * T) / 32;  // the user's rows live in qkv tile row tr ...
    const uint32_t r0 = (u * T) % 32;  // ... at rows r0 .. r0+T-1 of that tile row
    const uint32_t pg = tr * w_tiles;  // page of tile column 0 of that tile row
    // tile column of this head's chunk c within a [.., C]-wide row: q(hk) | k(hk) | v(h)
    auto chunk_col = [&](uint32_t c) -> uint32_t {
        if (c < Kt) {
            return hk * Kt + c;
        }
        if (c < 2 * Kt) {
            return Nk * Kt + hk * Kt + (c - Kt);
        }
        return 2 * Nk * Kt + h * Vt + (c - 2 * Kt);
    };

    // ---- ctrl page: parity, mi[u], initial ring block of (u,h)
    ctrl_r.reserve_back(1);
    noc.async_read(ctrl_acc, ctrl_r, ctrl_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    uint32_t par, mi, blk;
    {
        auto lock = ctrl_r.scoped_write_lock(1);
        auto p = lock.template get_ptr<volatile uint32_t>();
        par = p[0] & 1u;
        mi = p[1 + u];
        blk = p[1 + B + bh];
    }
    ctrl_r.push_back(1);
    ctrl_r.wait_front(1);
    ctrl_r.pop_front(1);
    if (blk == hold_sentinel) {
        // HOLD: the writer skips this head's ring writes and copies the window through, so any valid block and any
        // mi serve the (discarded) recurrence -- this head's own slot-0 block, mi = 0.
        blk = bh;
        mi = 0;
    }
    if (mi > T - 1) {
        mi = T - 1;  // ctrl contents are data (not validated by the host): an out-of-range mi is clamped, not reported
    }

    // ---- constants and per-head scalars (each scalar is a small read with its own barrier)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
    load_head_scalar<dtb_fp32 != 0>(dtb_acc, dtb_s, noc, h);
    load_head_scalar<nea_fp32 != 0>(nea_acc, nea_s, noc, h);

    // ---- phase 1 (small): z, the a|b tile(s), the gate selectors -> compute starts silu(z) and the T gate pairs
    z_in.reserve_back(Vt);
    read_tiles_at(qkv_acc, z_in, noc, pg + z_tile0 + h * Vt, Vt, 0);
    // ab_in[0] = a's tile (ab_page + h/32); ab_in[1] = b's tile (ab_page + (Nv+h)/32) only when it differs (b_idx = 1;
    // AB2 = 0 -> one tile, one entry, exactly the 2*Nv <= 32 read)
    constexpr uint32_t AB2 = (2 * Nv > 32) ? 1u : 0u;
    ab_in.reserve_back(1 + AB2);
    read_tiles_at(qkv_acc, ab_in, noc, pg + ab_page + (h >> 5), 1, 0);
    if constexpr (AB2 != 0) {
        if (b_idx != 0) {
            read_tiles_at(qkv_acc, ab_in, noc, pg + ab_page + ((Nv + h) >> 5), 1, 1);
        }
    } else {
        (void)b_idx;
    }
    // csel[0]: row h & 31 all ones (column extractor of a within its tile), csel[1]: row (Nv + h) & 31 (of b within
    // its)
    csel.reserve_back(2);
    zero_reserved(csel, noc, 2);
    {
        auto lock = csel.scoped_write_lock(2);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t j = 0; j < 32; ++j) {
            set_one(p16, 0, h & 31u, j);
            set_one(p16, 1, (Nv + h) & 31u, j);
        }
    }
    csel.push_back(2);
    // rsel[t]: 1.0 at column r0 + t in every row (rsel[t] @ X = every row equal to row r0 + t of X)
    rsel.reserve_back(T);
    zero_reserved(rsel, noc, T);
    {
        auto lock = rsel.scoped_write_lock(T);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t i = 0; i < 32; ++i) {
                set_one(p16, t, i, r0 + t);
            }
        }
    }
    rsel.push_back(T);
    noc.async_read_barrier();
    z_in.push_back(Vt);
    ab_in.push_back(1 + AB2);

    // ---- phase 2 (bulk): window + raw projection tiles, taps, weight row; the conv selectors built meanwhile
    // src_in[c] = whole window tile of user u, chunk c (win[par]); src_in[Ch + c] = whole raw projection tile
    src_in.reserve_back(2 * Ch);
    for (uint32_t c = 0; c < Ch; ++c) {
        const uint32_t col = chunk_col(c);
        if (par) {
            read_tiles_at(wb_acc, src_in, noc, u * Ct + col, 1, c);
        } else {
            read_tiles_at(wa_acc, src_in, noc, u * Ct + col, 1, c);
        }
        read_tiles_at(qkv_acc, src_in, noc, pg + col, 1, Ch + c);
    }
    // taps[c]: rows 0..K-1 (tap j in row j) of tile column chunk_col(c); the rest of the tile is zero
    taps.reserve_back(Ch);
    zero_reserved(taps, noc, Ch);
    {
        const uint32_t entry = taps.get_entry_size();
        const uint32_t nbytes = K * 32;  // K face rows of bf16
        for (uint32_t c = 0; c < Ch; ++c) {
            const uint32_t page = chunk_col(c);
            const uint32_t base = c * entry;
            noc.async_read(taps_acc, taps, nbytes, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = base});
            noc.async_read(
                taps_acc,
                taps,
                nbytes,
                {.page_id = page, .offset_bytes = entry / 4},
                {.offset_bytes = base + entry / 4});
        }
    }
    w_in.reserve_back(Vt);
    zero_reserved(w_in, noc, Vt);
    for (uint32_t t = 0; t < Vt; ++t) {
        read_row0_at(w_acc, w_in, noc, t, t);  // the norm weight lives in row 0 (row-broadcast operand)
    }
    // sel[0] = WO_W: row i (i < K-1) picks window row mi+1+i of E_prev; sel[1] = WO_N: row i (K-1 <= i < Lw) picks
    // raw row r0 + i - (K-1); sel[2+j] = SH_j: row r0+t picks W row t+j (the tap-j operand of output token t);
    // sel[2+K] = P: diagonal one-hot on rows r0..r0+T-1 (the conv tile passes through a matmul as the plain op's does)
    sel.reserve_back(3 + K);
    zero_reserved(sel, noc, 3 + K);
    {
        auto lock = sel.scoped_write_lock(3 + K);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t i = 0; i < K - 1; ++i) {
            set_one(p16, 0, i, mi + 1 + i);
        }
        for (uint32_t i = K - 1; i < Lw; ++i) {
            set_one(p16, 1, i, r0 + i - (K - 1));
        }
        for (uint32_t j = 0; j < K; ++j) {
            for (uint32_t t = 0; t < T; ++t) {
                set_one(p16, 2 + j, r0 + t, t + j);
            }
        }
        for (uint32_t t = 0; t < T; ++t) {
            set_one(p16, 2 + K, r0 + t, r0 + t);
        }
    }
    sel.push_back(3 + K);
    // mask_T: 1.0 at (r0 + t, 0) for t < T; e_t[t]: 1.0 at (r0 + t, 0)
    mask_T.reserve_back(1);
    zero_reserved(mask_T, noc, 1);
    {
        auto lock = mask_T.scoped_write_lock(1);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t t = 0; t < T; ++t) {
            set_one(p16, 0, r0 + t, 0);
        }
    }
    mask_T.push_back(1);
    e_t.reserve_back(T);
    zero_reserved(e_t, noc, T);
    {
        auto lock = e_t.scoped_write_lock(T);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t t = 0; t < T; ++t) {
            set_one(p16, t, r0 + t, 0);
        }
    }
    e_t.push_back(T);
    noc.async_read_barrier();
    src_in.push_back(2 * Ch);
    taps.push_back(Ch);
    w_in.push_back(Vt);

    // ---- the initial state, last: 16 tiles from ring block blk (the writer's block-0 write of this core is ordered
    // after this read completes, since compute cannot produce a state before it is pushed)
    state_in.reserve_back(KV);
    read_tiles_at(ring_acc, state_in, noc, blk * KV, KV, 0);
    noc.async_read_barrier();
    state_in.push_back(KV);
}
