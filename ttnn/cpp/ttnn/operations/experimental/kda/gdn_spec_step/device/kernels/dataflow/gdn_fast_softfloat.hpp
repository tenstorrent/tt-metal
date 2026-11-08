// gov gdn-v2 (gdn_spec_step kernel_opts bit 17, GDN_FAST_SOFTFLOAT): fast bit-exact IEEE-754 binary32 multiply / add (round-to-nearest-even) for the FPU-less RISC-V data
// movement cores. Fast path: both operands normal (or zero for add / mul), result normal; everything else (subnormal
// operands or results, infinities, NaNs, overflow) falls back to the compiler's soft-float operator (libgcc), so every
// input gives exactly the IEEE result the plain `a * b` / `a + b` gives.
#pragma once
#include <cstdint>

namespace gdn_fsf {
// (host-verified against IEEE binary32 on 2e9+ random / edge-case operand pairs and exhaustively over bf16 gate inputs:
// scratchpad gv2/sf/test.cpp, gate_test.cpp)

inline float fb(uint32_t b) { union { uint32_t u; float f; } v{b}; return v.f; }
inline uint32_t bf(float f) { union { float f; uint32_t u; } v{f}; return v.u; }

// a * b
__attribute__((noinline)) float fmul(float fa, float fb_) {
    const uint32_t a = bf(fa), b = bf(fb_);
    const uint32_t ea = (a >> 23) & 0xFFu, eb = (b >> 23) & 0xFFu;
    const uint32_t sign = (a ^ b) & 0x80000000u;
    if ((ea - 1u) >= 254u || (eb - 1u) >= 254u) {  // zero / subnormal / inf / nan operand
        if (((a << 1) == 0u && eb != 0xFFu) || ((b << 1) == 0u && ea != 0xFFu)) {
            return fb(sign);  // 0 * finite = signed zero
        }
        return fa * fb_;
    }
    const uint32_t ma = (a & 0x7FFFFFu) | 0x800000u, mb = (b & 0x7FFFFFu) | 0x800000u;
    const uint64_t p = static_cast<uint64_t>(ma) * mb;  // [2^46, 2^48)
    int32_t e = static_cast<int32_t>(ea + eb) - 127;
    uint32_t hi = static_cast<uint32_t>(p >> 32), lo = static_cast<uint32_t>(p);
    uint32_t mant, rem, half;
    if (hi & 0x8000u) {  // p >= 2^47: mantissa = p >> 24
        mant = (hi << 8) | (lo >> 24);
        rem = lo & 0xFFFFFFu;
        half = 0x800000u;
        e += 1;
    } else {  // mantissa = p >> 23
        mant = (hi << 9) | (lo >> 23);
        rem = lo & 0x7FFFFFu;
        half = 0x400000u;
    }
    if (rem > half || (rem == half && (mant & 1u))) {
        mant += 1u;
        if (mant == 0x1000000u) {
            mant >>= 1;
            e += 1;
        }
    }
    if (e <= 0 || e >= 255) {
        return fa * fb_;  // subnormal / overflow result
    }
    return fb(sign | (static_cast<uint32_t>(e) << 23) | (mant & 0x7FFFFFu));
}

// a + b
__attribute__((noinline)) float fadd(float fa, float fb_) {
    uint32_t a = bf(fa), b = bf(fb_);
    uint32_t ea = (a >> 23) & 0xFFu, eb = (b >> 23) & 0xFFu;
    if ((ea - 1u) >= 254u || (eb - 1u) >= 254u) {  // zero / subnormal / inf / nan operand
        if ((b << 1) == 0u && ea != 0u && ea != 0xFFu) {
            return fa;  // normal + (+-0) = normal
        }
        if ((a << 1) == 0u && eb != 0u && eb != 0xFFu) {
            return fb_;
        }
        return fa + fb_;
    }
    // |a| >= |b|
    if ((b & 0x7FFFFFFFu) > (a & 0x7FFFFFFFu)) {
        const uint32_t t = a; a = b; b = t;
        const uint32_t te = ea; ea = eb; eb = te;
    }
    const uint32_t sign = a & 0x80000000u;
    const uint32_t d = ea - eb;
    if (d > 25u) {
        return fb(a);  // |b| < half an ulp of a (never a tie)
    }
    // 3 extra low bits: guard, round, sticky (mantissas in bits [3, 26])
    uint32_t ma = ((a & 0x7FFFFFu) | 0x800000u) << 3;
    uint32_t mb = ((b & 0x7FFFFFu) | 0x800000u) << 3;
    if (d) {
        const uint32_t sticky = (mb & ((1u << d) - 1u)) != 0u;
        mb = (mb >> d) | sticky;
    }
    int32_t e = static_cast<int32_t>(ea);
    uint32_t m;
    if ((a ^ b) & 0x80000000u) {
        m = ma - mb;
        if (m == 0u) {
            return fb(0u);  // exact cancellation: +0 under round-to-nearest
        }
        // normalize left until bit 26 is set
        while (!(m & 0x4000000u)) {
            m <<= 1;
            e -= 1;
        }
    } else {
        m = ma + mb;
        if (m & 0x8000000u) {  // carry into bit 27
            m = (m >> 1) | (m & 1u);
            e += 1;
        }
    }
    // round to nearest even on the 3 low bits
    uint32_t mant = m >> 3;
    const uint32_t low = m & 7u;
    if (low > 4u || (low == 4u && (mant & 1u))) {
        mant += 1u;
        if (mant == 0x1000000u) {
            mant >>= 1;
            e += 1;
        }
    }
    if (e <= 0 || e >= 255) {
        return fa + fb_;  // subnormal / overflow result
    }
    return fb(sign | (static_cast<uint32_t>(e) << 23) | (mant & 0x7FFFFFu));
}

inline float fsub(float a, float b) { return fadd(a, fb(bf(b) ^ 0x80000000u)); }

}  // namespace gdn_fsf
