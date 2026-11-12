// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader: one core = (user u, value head h). Reads the ctrl page (parity, mi[u], initial ring block of (u,h); HOLD
// sentinel -> block bh, mi 0), the constants, the head's 12 whole window tiles (win[par], user u) + 12 whole raw
// projection tiles + the 12 tap tiles (rows 0..K-1 only), z, the a|b tile(s), the norm weight row, builds the bf16
// one-hot selector tiles, then reads the 64 KiB initial state last (compute's conv/pre overlap it). No per-token reads.
// a|b: a[h] is column h and b[h] column Nv + h past tile ab_page. With 2*Nv <= 32 both sit in tile ab_page (one read,
// ab_in[0]); with Nv = 24 (TP = 2) the a|b block spans two tiles and b's tile is read into ab_in[1] when it differs
// from a's (b_idx = 1, decided by the host per core), so the compute gathers b's row from ab_in[b_idx].
// GDN_FAST order: ctrl, constants, z + a|b, bulk reads + selectors (pushed), then dt_bias / nea, the state read
// (issued), the software gate math and beta_t / dec fills, state push (compute needs dec / beta_t only inside the token
// loop). GDN_FAST (fast_mode): the a|b tile(s) are staging only. The reader extracts a[u*T+t, h] / b[u*T+t, h] for the
// T tokens, computes beta = sigmoid(b) and decay = exp(-exp(A) * softplus(a + dt_bias)) in software fp32 (the very
// functions of reader_gdn_decode_step_conv_fast.cpp, gdn_step_gate_math.hpp) and pushes them as T all-equal fp32
// scalar tiles beta_t[t] / dec[t]; the gate selector tiles (csel, rsel) and the SFPU gate path do not exist.
// VB > 1 (value blocks): core (u, h, s) reads only state column tiles [s*Vs, (s+1)*Vs) of each of the Kt tile rows of
// the initial block (KVs = Kt*Vs tiles, row-major (i, j) at i*Vs + j); every other read is the full head's, as the
// conv / l2norm / gate prefix runs in full on every block.
// GDN_GATE_SCALARS (kernel_opts bit 1, fast_mode + fp32_state): the gate math moves to the writer RISC and the compute
// builds the scalar tiles in DST (fill_tile_bitcast of the same fp32 bits), so this reader skips the a|b staging read,
// the dt_bias / -exp(A) reads, the software gate math and the 2*T full-tile fills (1024 RISC-V stores each), and pushes
// the initial state as soon as its read lands. GDN_EARLY_SRC (kernel_opts bit 2): src_in and the selector tiles are
// pushed right after their reads / build, before the taps, the norm weight row and the row masks (which the compute
// needs only from phase A3 on), so phase A starts earlier. Both move no value: every tile holds the same bytes.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "api/debug/kernel_profiler.hpp"  // zones compile to nothing unless the build defines PROFILE_KERNEL

#ifndef GDN_FAST
#define GDN_FAST 0
#endif
#ifndef GDN_GATE_SCALARS
#define GDN_GATE_SCALARS 0
#endif
#ifndef GDN_EARLY_SRC
#define GDN_EARLY_SRC 0
#endif
#ifndef GDN_QKSHARE
#define GDN_QKSHARE 0  // kernel_opts bit 4 (VB > 1): this block reads only its q/k share + its v chunks
#endif
#if GDN_FAST && !GDN_GATE_SCALARS
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_spec_step/device/kernels/dataflow/gdn_step_gate_math.hpp"
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
    uint32_t M,  // ring/ctrl layout width (>= B)
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
    uint32_t AB2,
    uint32_t VB>
TT_KERNEL void reader(uint32_t u, uint32_t h, uint32_t b_idx, uint32_t s) {
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
#if !GDN_GATE_SCALARS
    DataflowBuffer ab_in(dfb::ab_in);
#endif
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer mask_T(dfb::mask_T);
    DataflowBuffer e_t(dfb::e_t);
#if GDN_GATE_SCALARS
    // the writer computes the gates; the compute builds beta_t / dec in DST
    (void)b_idx;
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
    // QKSHARE: this block's ChA chunks, in block order: q/k chunks [s*Qk, (s+1)*Qk) (Qk = 2*Kt / VB), then its own v
    // chunks [2*Kt + s*Vs, ..); otherwise ChA = Ch and the block order is the head order
    constexpr uint32_t Qk = GDN_QKSHARE ? (2 * Kt) / VB : 0u;
    constexpr uint32_t ChA = GDN_QKSHARE ? Qk + Vs : Ch;
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

    // ---- ctrl page: parity, mi[u], initial ring block of (u,h)
    uint32_t par, mi, blk;
    {
        DeviceZoneScopedN("rd_ctrl");
        ctrl_r.reserve_back(1);
        noc.async_read(ctrl_acc, ctrl_r, ctrl_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        {
            auto lock = ctrl_r.scoped_write_lock(1);
            auto p = lock.template get_ptr<volatile uint32_t>();
            par = p[0] & 1u;
            mi = p[1 + u];
            blk = p[1 + M + bh];
        }
        ctrl_r.push_back(1);
        ctrl_r.wait_front(1);
        ctrl_r.pop_front(1);
    }
    if (blk == hold_sentinel) {
        // HOLD: the writer skips this head's ring writes and copies the window through, so any valid block and any
        // mi serve the (discarded) recurrence -- this head's own slot-0 block, mi = 0.
        blk = bh;
        mi = 0;
    }
    if (mi > T - 1) {
        mi = T - 1;  // ctrl contents are data (not validated by the host): an out-of-range mi is clamped, not reported
    }
    // the initial state of this block: KVs tiles of ring block blk (VB = 1: the whole 16-tile block in one run)
    auto read_state = [&]() {
        if constexpr (VB == 1) {
            (void)s;
            read_tiles_at(ring_acc, state_in, noc, blk * KV, KV, 0);
        } else {
            for (uint32_t i = 0; i < Kt; ++i) {
                read_tiles_at(ring_acc, state_in, noc, blk * KV + i * Vt + s * Vs, Vs, i * Vs);
            }
        }
    };

    // ---- constants and per-head scalars (each scalar is a small read with its own barrier)
    {
        DeviceZoneScopedN("rd_const");
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
        generate_bcast_col_scalar(eps_l2, l2_eps_bits);
        generate_bcast_col_scalar(eps_norm, norm_eps_bits);
#if !GDN_FAST
        load_head_scalar<dtb_fp32 != 0>(dtb_acc, dtb_s, noc, h);
        load_head_scalar<nea_fp32 != 0>(nea_acc, nea_s, noc, h);
#endif
    }

    // ---- phase 1 (small): z, the a|b tile(s), the gates -> compute starts silu(z) (and, plain mode, the T gate pairs)
    z_in.reserve_back(Vt);
    read_tiles_at(qkv_acc, z_in, noc, pg + z_tile0 + h * Vt, Vt, 0);
#if GDN_GATE_SCALARS
    noc.async_read_barrier();
    z_in.push_back(Vt);
#else
    // ab_in[0] = a's tile (ab_page + h/32); ab_in[1] = b's tile (ab_page + (Nv+h)/32) only when it differs (b_idx = 1;
    // AB2 = 0 -> one tile, one entry, exactly the 2*Nv <= 32 read). AB2 is the factory's compile-time arg (it also
    // sizes ab_in and g1), not re-derived here.
    ab_in.reserve_back(1 + AB2);
    read_tiles_at(qkv_acc, ab_in, noc, pg + ab_page + (h >> 5), 1, 0);
    if constexpr (AB2 != 0) {
        if (b_idx != 0) {
            read_tiles_at(qkv_acc, ab_in, noc, pg + ab_page + ((Nv + h) >> 5), 1, 1);
        }
    } else {
        (void)b_idx;
    }
#if GDN_FAST
    noc.async_read_barrier();
    z_in.push_back(Vt);
#else
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
    // With AB2 = 1 and b_idx = 0 (a and b share a tile: heads h < 8 at Nv = 24) entry 1 is pushed UNWRITTEN; compute
    // gathers b from ab_in[b_idx] = entry 0, so it is never indexed.
    ab_in.push_back(1 + AB2);
#endif
#endif  // GDN_GATE_SCALARS

    // ---- phase 2 (bulk): window + raw projection tiles, taps, weight row; the conv selectors built meanwhile
    auto build_sel = [&]() {
        // sel[0] = WO_W: row i (i < K-1) picks window row mi+1+i of E_prev; sel[1] = WO_N: row i (K-1 <= i < Lw) picks
        // raw row r0 + i - (K-1); sel[2+j] = SH_j: row r0+t picks W row t+j (the tap-j operand of output token t);
        // sel[2+K] = P: diagonal one-hot on rows r0..r0+T-1 (the conv tile passes through a matmul as the plain op's
        // does)
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
    };
    {
        DeviceZoneScopedN("rd_bulk");
        // src_in[c] = whole window tile of user u, chunk c (win[par]); src_in[Ch + c] = whole raw projection tile
        src_in.reserve_back(2 * ChA);
        for (uint32_t c = 0; c < ChA; ++c) {
            const uint32_t col = chunk_col(blk_chunk(c));
            if (par) {
                read_tiles_at(wb_acc, src_in, noc, u * Ct + col, 1, c);
            } else {
                read_tiles_at(wa_acc, src_in, noc, u * Ct + col, 1, c);
            }
            read_tiles_at(qkv_acc, src_in, noc, pg + col, 1, ChA + c);
        }
#if GDN_EARLY_SRC
        // phase A1 / A2 need only src_in and the selectors: hand them over before the taps / weight row / masks
        build_sel();
        noc.async_read_barrier();
        src_in.push_back(2 * ChA);
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
#if !GDN_EARLY_SRC
        build_sel();
#endif
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
#if !GDN_EARLY_SRC
        src_in.push_back(2 * ChA);
#endif
        taps.push_back(ChA);
        w_in.push_back(Vt);
    }

#if GDN_GATE_SCALARS
    // ---- GATE_SCALARS: the gates are the writer's; the initial state goes out as soon as it lands
    {
        DeviceZoneScopedN("rd_state");
        state_in.reserve_back(KVs);
        read_state();
        noc.async_read_barrier();
        state_in.push_back(KVs);
    }
#elif GDN_FAST
    // ---- FAST tail: the compute is busy with phase A / B (about 25 us), so the reader's serial software gate math
    // runs now instead of ahead of the bulk reads. dt_bias / -exp(A) first (small, own barriers), then the 64 KiB
    // initial state read is issued, then the T gate pairs of user u are computed while that read is in flight; the
    // compute needs dec / beta_t only inside the token loop.
    uint32_t dtb_bits, nea_bits;
    {
        DeviceZoneScopedN("rd_scalars");
        dtb_bits = read_head_scalar_bits < dtb_fp32 != 0 > (dtb_acc, dtb_s, noc, h);
        nea_bits = read_head_scalar_bits < nea_fp32 != 0 > (nea_acc, nea_s, noc, h);
    }
    state_in.reserve_back(KVs);
    read_state();
    // The T gate pairs of user u from rows r0 .. r0+T-1 of the a|b tile(s): a[h] at column h & 31 of entry 0,
    // b[h] at column (Nv + h) & 31 of entry b_idx (AB2) / entry 0. bf16 -> fp32 bits is exact (<< 16); the gate math
    // is reader_gdn_decode_step_conv_fast.cpp's, bit for bit. The ab_in slot stays reserved (staging, never pushed).
    {
        DeviceZoneScopedN("rd_gates");
        uint32_t beta_bits[T];
        uint32_t dec_bits[T];
        {
            auto lock = ab_in.scoped_write_lock(1 + AB2);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            const uint32_t b_tile = (AB2 != 0) ? b_idx : 0u;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t a_bits = static_cast<uint32_t>(p16[tile_elem_index(r0 + t, h & 31u)]) << 16;
                const uint32_t b_bits =
                    static_cast<uint32_t>(p16[b_tile * 1024 + tile_elem_index(r0 + t, (Nv + h) & 31u)]) << 16;
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
    {
        DeviceZoneScopedN("rd_state");
        noc.async_read_barrier();
        state_in.push_back(KVs);
    }
#else
    // ---- the initial state, last: 16 tiles from ring block blk (the writer's block-0 write of this core is ordered
    // after this read completes, since compute cannot produce a state before it is pushed)
    {
        DeviceZoneScopedN("rd_state");
        state_in.reserve_back(KVs);
        read_state();
        noc.async_read_barrier();
        state_in.push_back(KVs);
    }
#endif
}
