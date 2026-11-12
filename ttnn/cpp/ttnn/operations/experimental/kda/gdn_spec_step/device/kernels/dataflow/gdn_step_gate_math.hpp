// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Software-fp32 GDN gate math for the fast_mode readers. VERBATIM copy of the anonymous-namespace helpers of
// gdn_decode_step/device/kernels/dataflow/reader_gdn_decode_step_conv_fast.cpp (fill_scalar_tile_fast, f_from_bits,
// bits_from_f, exp_acc, log1p_acc, read_head_scalar_bits) plus gate_beta_decay_bits, the per-user gate block of that
// reader as one function. gdn_spec_step(fast_mode=True) is bit-identical to gdn_decode_step(fast_mode=True) only
// while the two stay identical: keep them in sync (tests/ttnn/unit_tests/operations/experimental/kda/
// test_gdn_spec_step.py::test_tier1_vs_gdn_decode_step_conv[fast*] guards the identity on device).
#pragma once
#include <cstdint>
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"

// GDN_GATE_NOINLINE (the GATE_SCALARS writer): exp_acc / log1p_acc out of line -- the same float operations in the
// same order (soft-float library calls on these RISC-Vs), one copy instead of one per call site (code size)
#ifdef GDN_GATE_NOINLINE
#define GDN_GATE_FN __attribute__((noinline))
#else
#define GDN_GATE_FN inline
#endif

namespace gdn_step_df {

// Broadcast one fp32 value over a whole fp32 tile (slot reserved): plain unrolled stores (the lock flushes on release).
inline void fill_scalar_tile_fast(DataflowBuffer& dfb, uint32_t value) {
    auto lock = dfb.scoped_write_lock(1);
    uint32_t* p = &lock.template get_ptr<uint32_t>()[0];
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 1024; ++i) {
        p[i] = value;
    }
}

// Broadcast values[t] over fp32 tile t of `n` reserved consecutive slots (same stores as fill_scalar_tile_fast).
inline void fill_scalar_tiles_fast(DataflowBuffer& dfb, const uint32_t* values, uint32_t n) {
    auto lock = dfb.scoped_write_lock(n);
    uint32_t* p = &lock.template get_ptr<uint32_t>()[0];
    for (uint32_t t = 0; t < n; ++t) {
        const uint32_t value = values[t];
        uint32_t* q = p + t * 1024;
#pragma GCC unroll 16
        for (uint32_t i = 0; i < 1024; ++i) {
            q[i] = value;
        }
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
GDN_GATE_FN float exp_acc(float x) {
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
GDN_GATE_FN float log1p_acc(float y) {
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

// beta = sigmoid(b), decay = exp(-exp(A) * softplus(a + dt_bias)) (softplus threshold 20), all in fp32 bits: the
// per-user gate block of reader_gdn_decode_step_conv_fast.cpp (a_bits / b_bits are the bf16 projection values
// widened to fp32, dtb_bits / nea_bits the head's dt_bias / -exp(A_log)).
inline void gate_beta_decay_bits(
    uint32_t a_bits, uint32_t b_bits, uint32_t dtb_bits, uint32_t nea_bits, uint32_t& beta_bits, uint32_t& dec_bits) {
    const float bv = f_from_bits(b_bits);
    const float beta = 1.0f / (1.0f + exp_acc(-bv));
    const float xv = f_from_bits(a_bits) + f_from_bits(dtb_bits);
    const float sp = xv > 20.0f ? xv : log1p_acc(exp_acc(xv));
    const float dec = exp_acc(f_from_bits(nea_bits) * sp);
    beta_bits = bits_from_f(beta);
    dec_bits = bits_from_f(dec);
}

}  // namespace gdn_step_df
