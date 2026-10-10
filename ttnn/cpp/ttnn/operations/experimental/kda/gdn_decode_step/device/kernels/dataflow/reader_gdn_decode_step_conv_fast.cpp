// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for the batched fused-conv variant: one core = value head `head`, users [u0, u0 + nu). Per core once: taps,
// norm weight, z tiles (rows = users), dt_bias / -exp(A) scalars, constants. Per user: 3 packed history slots, the new
// token packed from row b of the projection tile, a/b scalars from row b, per-user selector + mask tiles, the state.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

using namespace gdn_step_df;

namespace {
// Broadcast one fp32 value over a whole fp32 tile (slot reserved): plain unrolled stores (the lock flushes on release).
inline void fill_scalar_tile_fast(DataflowBuffer& dfb, uint32_t value) {
    auto lock = dfb.scoped_write_lock(1);
    uint32_t* p = reinterpret_cast<uint32_t*>(lock.template get_ptr<uint32_t>().get_address());
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 1024; ++i) {
        p[i] = value;
    }
}

// FAST2: fp32 scalar gate math on the reader (the compute gets ready-made beta / decay scalar tiles).
inline float f_from_bits(uint32_t b) {
    union {
        uint32_t u;
        float f;
    } v{b};
    return v.f;
}
inline uint32_t bits_from_f(float f) {
    union {
        float f;
        uint32_t u;
    } v{f};
    return v.u;
}
// exp(x): Cody-Waite range reduction + degree-7 Taylor on |r| <= ln2/2 (rel. err ~1e-7)
inline float exp_acc(float x) {
    if (x < -87.0f) {
        return 0.0f;
    }
    if (x > 88.0f) {
        x = 88.0f;
    }
    const float fn = x * 1.4426950408889634f;
    const int32_t n = static_cast<int32_t>(fn < 0.0f ? fn - 0.5f : fn + 0.5f);
    const float r = (x - static_cast<float>(n) * 0.693359375f) - static_cast<float>(n) * (-2.12194440e-4f);
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return f_from_bits(bits_from_f(p) + (static_cast<uint32_t>(n) << 23));
}
// log(1 + y), y >= 0
inline float log1p_acc(float y) {
    if (y < 1e-4f) {
        return y * (1.0f - y * (0.5f - y * (1.0f / 3.0f)));
    }
    const float z = 1.0f + y;
    uint32_t zb = bits_from_f(z);
    int32_t e = static_cast<int32_t>((zb >> 23) & 0xFF) - 127;
    float m = f_from_bits((zb & 0x007FFFFFu) | 0x3F800000u);  // [1, 2)
    if (m > 1.41421356f) {
        m *= 0.5f;
        e += 1;
    }
    const float t = (m - 1.0f) / (m + 1.0f);
    const float t2 = t * t;
    float s = 1.0f / 11.0f;
    s = s * t2 + 1.0f / 9.0f;
    s = s * t2 + 1.0f / 7.0f;
    s = s * t2 + 1.0f / 5.0f;
    s = s * t2 + 1.0f / 3.0f;
    s = s * t2 + 1.0f;
    return static_cast<float>(e) * 0.6931471805599453f + 2.0f * t * s;
}

template <bool src_fp32, typename Accessor>
inline uint32_t read_head_scalar_bits(const Accessor& acc, DataflowBuffer& staging, Noc& noc, uint32_t col) {
    staging.reserve_back(1);  // staging only: never pushed (the compute does not consume this DFB in FAST2)
    return load_tile_scalar<src_fp32>(acc, staging, noc, 0, col);
}

template <bool src_fp32, typename Accessor>
inline void load_head_scalar_fast(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t col) {
    dfb.reserve_back(1);
    const uint32_t value = load_tile_scalar<src_fp32>(acc, dfb, noc, 0, col);
    fill_scalar_tile_fast(dfb, value);
    dfb.push_back(1);
}
}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t z_tile0,
    uint32_t ab_page,
    uint32_t dtb_fp32,
    uint32_t nea_fp32,
    uint32_t l2_eps_bits,
    uint32_t norm_eps_bits>
TT_KERNEL void reader(uint32_t head, uint32_t u0, uint32_t nu) {
    const auto qkv_acc = TensorAccessor(tensor::qkv);
    const auto dtb_acc = TensorAccessor(tensor::dtb);
    const auto nea_acc = TensorAccessor(tensor::nea);
    const auto state_acc = TensorAccessor(tensor::state);
    const auto w_acc = TensorAccessor(tensor::weight);
    const auto hist_acc = TensorAccessor(tensor::hist);
    const auto taps_acc = TensorAccessor(tensor::taps);
    DataflowBuffer hist(dfb::hist);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer cur(dfb::cur);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer mask(dfb::mask);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ct = 2 * Kt + Vt;
    constexpr uint32_t rf = Nv / Nk;
    const uint32_t h = head;
    const uint32_t hk = h / rf;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
    // FAST: order = when the compute needs it: z first (its SiLU runs while the conv inputs are still in flight), then
    // the conv inputs (taps, selectors, history, new token), then dt_bias / -exp(A) / a / b (gates run after the L2
    // norms), then the state, then the norm weight. Scalar tiles are filled with plain (non-volatile, unrolled) stores.
    read_tiles(qkv_acc, z_in, noc, z_tile0 + h * Vt, Vt);  // all users' rows of this head's z
    read_tiles(taps_acc, taps, noc, h * 4, 4);
    uint32_t dtb_bits = 0, nea_bits = 0;
    for (uint32_t ui = 0; ui < nu; ++ui) {
        const uint32_t b = u0 + ui;
        const uint32_t bh = b * Nv + h;
        build_user_selectors(sel, mask, noc, b, Ct);
        read_tiles(hist_acc, hist, noc, bh * 4 + 1, 3);  // slots 1..3
        cur.reserve_back(1);
        zero_reserved(cur, noc, 1);
        pack_head_tile_user<Kt, Vt, Nk>(qkv_acc, cur, noc, hk, h, b, 0);
        noc.async_read_barrier();
        cur.push_back(1);
        if (ui == 0) {
            dtb_bits = read_head_scalar_bits < dtb_fp32 != 0 > (dtb_acc, dtb_s, noc, h);
            nea_bits = read_head_scalar_bits < nea_fp32 != 0 > (nea_acc, nea_s, noc, h);
        }
        // a[b,h], b[b,h]: row b of the a|b tile, columns h and Nv + h
        a_s.reserve_back(1);
        noc.async_read(qkv_acc, a_s, 2048, {.page_id = ab_page, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        uint32_t a_bits, b_bits;
        {
            auto lock = a_s.scoped_write_lock(1);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            a_bits = static_cast<uint32_t>(p16[tile_elem_index(b, h)]) << 16;
            b_bits = static_cast<uint32_t>(p16[tile_elem_index(b, Nv + h)]) << 16;
        }
        // FAST2: a_s <- beta = sigmoid(b), b_s <- decay = exp(-exp(A) * softplus(a + dt_bias)) (softplus threshold 20)
        {
            const float bv = f_from_bits(b_bits);
            const float beta = 1.0f / (1.0f + exp_acc(-bv));
            const float xv = f_from_bits(a_bits) + f_from_bits(dtb_bits);
            const float sp = xv > 20.0f ? xv : log1p_acc(exp_acc(xv));
            const float dec = exp_acc(f_from_bits(nea_bits) * sp);
            a_bits = bits_from_f(beta);
            b_bits = bits_from_f(dec);
        }
        fill_scalar_tile_fast(a_s, a_bits);
        a_s.push_back(1);
        b_s.reserve_back(1);
        fill_scalar_tile_fast(b_s, b_bits);
        b_s.push_back(1);
        read_tiles(state_acc, state_in, noc, bh * KV, KV);
        if (ui == 0) {
            read_tiles(w_acc, w_in, noc, 0, Vt);
        }
    }
}
