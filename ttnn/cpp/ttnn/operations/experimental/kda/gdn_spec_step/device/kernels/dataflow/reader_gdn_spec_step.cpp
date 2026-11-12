// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader: one core = (user u, value head h). Reads the ctrl page (parity, mi[u], start state of (u,h); the HOLD
// sentinel reads a placeholder state with mi 0), the constants, the head's whole window tiles (win[par], user u), the
// whole raw projection tiles and the tap tiles (rows 0..K-1 only), z, the a|b tile, the norm weight row, builds the
// bf16 one-hot selector tiles, and reads the start state last (the compute's conv and row-batched steps overlap it).
// a[h] is column h and b[h] column Nv + h of the a|b tile (2*Nv <= 32).
// Plain mode: the compute gathers the T gate pairs from the a|b tile with the csel / rsel one-hot tiles. GDN_FAST
// (fast_mode): the a|b tile is staging only; the reader computes beta = sigmoid(b) and
// decay = exp(-exp(A) * softplus(a + dt_bias)) for the T tokens in software fp32 (gdn_step_gate_math.hpp, the
// arithmetic of reader_gdn_decode_step_conv_fast.cpp) while the start state read is in flight, and pushes them as T
// all-equal fp32 scalar tiles beta_t[t] / dec[t].
// GDN_GATE_SCALARS (kernel_opts bit 1): the writer computes the gates and the compute fills the scalar tiles in DST,
// so this reader skips the a|b and dt_bias / -exp(A) reads, the gate math and the 2*T full-tile fills, and pushes the
// start state as soon as its read lands.
// VB > 1 (value blocks): core (u, h, s) reads only the state column tiles [s*Vs, (s+1)*Vs) of each of the Kt tile rows
// of the start state (KVs = Kt*Vs tiles, (i, j) at i*Vs + j). GDN_QKSHARE (kernel_opts bit 4) also limits the window,
// projection and tap reads to the block's ChA chunks: its q/k share [s*Qk, (s+1)*Qk), Qk = 2*Kt / VB, then its own v
// chunks. Without it every block reads all of the head's chunks.
// GDN_RING_PAGE_IDX: the ctrl word of (u,h) is the first fp32 ring page of the start state, not a block index.
// GDN_RING_POOL2 (= ring's page count): pages from there on are ring_pool2's (page - count).
// GDN_HIST_PAGED: the start history comes from gdn_decode_step's packed history pages (conv_hist) instead of win[par]:
// ctrl word 1 + B + B*Nv + bh is the slot-0 page of (u, h)'s start history, slots 1..3 go to rows 1..3 of the head's
// window tiles (mi = 0), the other window rows are 0, and the ctrl mi word is not read.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

#ifndef GDN_FAST
#define GDN_FAST 0
#endif
#ifndef GDN_HIST_PAGED
#define GDN_HIST_PAGED 0
#endif
#ifndef GDN_GATE_SCALARS
#define GDN_GATE_SCALARS 0
#endif
#ifndef GDN_QKSHARE
#define GDN_QKSHARE 0
#endif
#if GDN_FAST && !GDN_GATE_SCALARS
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_gate_math.hpp"
#endif

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
    uint32_t hold_sentinel,
    uint32_t VB>
TT_KERNEL void reader(uint32_t u, uint32_t h, uint32_t s) {
    const auto qkv_acc = TensorAccessor(tensor::qkv);
#if GDN_HIST_PAGED
    const auto hist_acc = TensorAccessor(tensor::hist);
#else
    const auto wa_acc = TensorAccessor(tensor::win_a);
    const auto wb_acc = TensorAccessor(tensor::win_b);
#endif
    const auto ring_acc = TensorAccessor(tensor::ring);
#ifdef GDN_RING_POOL2
    const auto ring2_acc = same_layout_accessor(ring_acc, tensor::ring2);
#endif
    const auto ctrl_acc = TensorAccessor(tensor::ctrl);
    const auto taps_acc = TensorAccessor(tensor::taps);
    const auto dtb_acc = TensorAccessor(tensor::dtb);
    const auto nea_acc = TensorAccessor(tensor::nea);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer src_in(dfb::src_in);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer z_in(dfb::z_in);
#if !GDN_GATE_SCALARS
    DataflowBuffer ab_in(dfb::ab_in);
#endif
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer mask_T(dfb::mask_T);
    DataflowBuffer e_t(dfb::e_t);
#if GDN_GATE_SCALARS
    (void)dtb_acc;
    (void)nea_acc;
#elif GDN_FAST
    DataflowBuffer beta_t(dfb::beta_t);
    DataflowBuffer dec(dfb::dec);
#else
    DataflowBuffer rsel(dfb::rsel);
    DataflowBuffer csel(dfb::csel);
#endif
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
#if !GDN_GATE_SCALARS
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
#endif
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer ctrl_r(dfb::ctrl_r);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ch = 2 * Kt + Vt;
    static_assert(VB >= 1 && Vt % VB == 0, "value blocks must divide Vt");
    constexpr uint32_t Vs = Vt / VB;  // this block's state column tiles
    constexpr uint32_t KVs = Kt * Vs;
    constexpr uint32_t rf = Nv / Nk;
    const uint32_t hk = h / rf;
    const uint32_t bh = u * Nv + h;
    const uint32_t tr = (u * T) / 32;  // the user's rows live in qkv tile row tr ...
    const uint32_t r0 = (u * T) % 32;  // ... at rows r0 .. r0+T-1 of that tile row
    const uint32_t pg = tr * w_tiles;  // page of tile column 0 of that tile row
    constexpr uint32_t Qk = GDN_QKSHARE ? (2 * Kt) / VB : 0u;
    constexpr uint32_t ChA = GDN_QKSHARE ? Qk + Vs : Ch;  // chunks this block reads
    // head chunk of this block's chunk ci (the identity without QKSHARE)
    auto blk_chunk = [&](uint32_t ci) -> uint32_t {
        if constexpr (GDN_QKSHARE) {
            return ci < Qk ? s * Qk + ci : 2 * Kt + s * Vs + (ci - Qk);
        } else {
            return ci;
        }
    };
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

    // ---- ctrl page: parity, mi[u], start state of (u,h) (and its start history page)
    uint32_t par, mi, blk;
#if GDN_HIST_PAGED
    uint32_t hist_page0;
#endif
    ctrl_r.reserve_back(1);
    noc.async_read(ctrl_acc, ctrl_r, ctrl_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    {
        auto lock = ctrl_r.scoped_write_lock(1);
        auto p = lock.template get_ptr<volatile uint32_t>();
        par = p[0] & 1u;
#if GDN_HIST_PAGED
        mi = 0;
        hist_page0 = p[1 + B + B * Nv + bh];
#else
        mi = p[1 + u];
#endif
        blk = p[1 + B + bh];
    }
    ctrl_r.push_back(1);
    ctrl_r.wait_front(1);
    ctrl_r.pop_front(1);
    if (blk == hold_sentinel) {
        // HOLD: the writer skips this head's ring writes and copies the window through, so any valid state and any
        // mi serve the discarded recurrence: this head's own block of the default ring layout, mi = 0
#ifdef GDN_RING_PAGE_IDX
        blk = bh * KV;  // ctrl words are pages in this mode
#else
        blk = bh;
#endif
        mi = 0;
#if GDN_HIST_PAGED
        hist_page0 = 0;  // any 4 valid pages: the held head's result is discarded
#endif
    }
#if GDN_HIST_PAGED
    (void)par;  // no window pair
#endif
    if (mi > T - 1) {
        mi = T - 1;  // ctrl contents are data (not validated by the host): an out-of-range mi is clamped, not reported
    }
#ifdef GDN_RING_PAGE_IDX
    const uint32_t state_page0 = blk;
#else
    const uint32_t state_page0 = blk * KV;
#endif
    // this block's KVs tiles of the start state at page `first` (VB = 1: the whole state in one run)
    auto read_state_from = [&](const auto& acc, uint32_t first) {
        if constexpr (VB == 1) {
            (void)s;
            read_tiles_at(acc, state_in, noc, first, KV, 0);
        } else {
            for (uint32_t i = 0; i < Kt; ++i) {
                read_tiles_at(acc, state_in, noc, first + i * Vt + s * Vs, Vs, i * Vs);
            }
        }
    };
    auto read_state = [&]() {
#ifdef GDN_RING_POOL2
        const bool in2 = state_page0 >= GDN_RING_POOL2;
        read_state_from(in2 ? ring2_acc : ring_acc, in2 ? state_page0 - GDN_RING_POOL2 : state_page0);
#else
        read_state_from(ring_acc, state_page0);
#endif
    };

    // ---- constants and per-head scalars (each scalar is a small read with its own barrier)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
#if !GDN_FAST
    load_head_scalar<dtb_fp32 != 0>(dtb_acc, dtb_s, noc, h);
    load_head_scalar<nea_fp32 != 0>(nea_acc, nea_s, noc, h);
#endif

    // ---- phase 1 (small): z, the a|b tile -> the compute starts silu(z) (and, plain mode, the T gate pairs)
    z_in.reserve_back(Vt);
    read_tiles_at(qkv_acc, z_in, noc, pg + z_tile0 + h * Vt, Vt, 0);
#if !GDN_GATE_SCALARS
    ab_in.reserve_back(1);
    read_tiles_at(qkv_acc, ab_in, noc, pg + ab_page, 1, 0);
#endif
#if GDN_FAST
    noc.async_read_barrier();
    z_in.push_back(Vt);
#else
    // csel[0]: row h all ones (column extractor of a), csel[1]: row Nv + h (of b)
    csel.reserve_back(2);
    zero_reserved(csel, noc, 2);
    {
        auto lock = csel.scoped_write_lock(2);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t j = 0; j < 32; ++j) {
            set_one(p16, 0, h, j);
            set_one(p16, 1, Nv + h, j);
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
    ab_in.push_back(1);
#endif

    // ---- phase 2 (bulk): window + raw projection tiles, taps, weight row; the conv selectors built meanwhile
    // src_in[c] = whole window tile of user u, chunk c (win[par]); src_in[ChA + c] = whole raw projection tile
    src_in.reserve_back(2 * ChA);
#if GDN_HIST_PAGED
    // window tile c: row 1 + i = history slot 1 + i, read from the slot's row of the same parity (the packed pages
    // hold head chunk g in rows 2g and 2g + 1, so each 32 B face-row read keeps its 64 B alignment class)
    zero_reserved(src_in, noc, ChA);
    for (uint32_t c = 0; c < ChA; ++c) {
        const uint32_t g = blk_chunk(c);
        for (uint32_t i = 0; i < K - 1; ++i) {
            pack_row_from_row(hist_acc, src_in, noc, hist_page0 + 1 + i, 2 * g + ((1 + i) & 1u), c, 1 + i);
        }
        read_tiles_at(qkv_acc, src_in, noc, pg + chunk_col(g), 1, ChA + c);
    }
#else
    for (uint32_t c = 0; c < ChA; ++c) {
        const uint32_t col = chunk_col(blk_chunk(c));
        if (par) {
            read_tiles_at(wb_acc, src_in, noc, u * Ct + col, 1, c);
        } else {
            read_tiles_at(wa_acc, src_in, noc, u * Ct + col, 1, c);
        }
        read_tiles_at(qkv_acc, src_in, noc, pg + col, 1, ChA + c);
    }
#endif
    // taps[c]: rows 0..K-1 (tap j in row j) of tile column chunk_col(c); the rest of the tile is zero
    taps.reserve_back(ChA);
    zero_reserved(taps, noc, ChA);
    {
        const uint32_t entry = taps.get_entry_size();
        const uint32_t nbytes = K * 32;  // K face rows of bf16
        for (uint32_t c = 0; c < ChA; ++c) {
            const uint32_t page = chunk_col(blk_chunk(c));
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
    // sel[0] = WO_W: row i (i < K-1) picks window row mi+1+i of E_prev; sel[1] = WO_N: row i (K-1 <= i < Lw) picks raw
    // row r0 + i - (K-1); sel[2+j] = SH_j: row r0+t picks W row t+j (the tap-j operand of output token t);
    // sel[2+K] = P: diagonal one-hot on rows r0..r0+T-1
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
    src_in.push_back(2 * ChA);
    taps.push_back(ChA);
    w_in.push_back(Vt);

    // ---- the start state, last. The writer's first state write of this core comes after this read completes: the
    //      compute cannot produce a state before it is pushed, so a destination may alias the start pages.
#if GDN_FAST && !GDN_GATE_SCALARS
    // dt_bias / -exp(A) first (small, own barriers), then the start state read is issued and the T gate pairs of user
    // u are computed while it is in flight; the compute needs dec / beta_t only inside the token loop
    const uint32_t dtb_bits = read_head_scalar_bits < dtb_fp32 != 0 > (dtb_acc, dtb_s, noc, h);
    const uint32_t nea_bits = read_head_scalar_bits < nea_fp32 != 0 > (nea_acc, nea_s, noc, h);
    state_in.reserve_back(KVs);
    read_state();
    {
        // bf16 -> fp32 bits is exact (<< 16). The ab_in slot stays reserved (staging, never pushed).
        uint32_t beta_bits[T];
        uint32_t dec_bits[T];
        {
            auto lock = ab_in.scoped_write_lock(1);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t a_bits = static_cast<uint32_t>(p16[tile_elem_index(r0 + t, h)]) << 16;
                const uint32_t b_bits = static_cast<uint32_t>(p16[tile_elem_index(r0 + t, Nv + h)]) << 16;
                gate_beta_decay_bits(a_bits, b_bits, dtb_bits, nea_bits, beta_bits[t], dec_bits[t]);
            }
        }
        beta_t.reserve_back(T);
        fill_scalar_tiles_fast(beta_t, beta_bits, T);
        beta_t.push_back(T);
        dec.reserve_back(T);
        fill_scalar_tiles_fast(dec, dec_bits, T);
        dec.push_back(T);
    }
    noc.async_read_barrier();
    state_in.push_back(KVs);
#else
    state_in.reserve_back(KVs);
    read_state();
    noc.async_read_barrier();
    state_in.push_back(KVs);
#endif
}
