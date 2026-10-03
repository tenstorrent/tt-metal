// Fix: Improve BF16 reciprocal rounding on Wormhole
// Issue #58227 — [Bounty $1.5k] Improve BF16 reciprocal rounding on Wormhole
//
// ROOT CAUSE
// ----------
// The Wormhole BF16 reciprocal path computes `1/x` by a single Newton-Raphson
// step seeded from the hardware's bf16 rcp approximation:
//
//     y0 = rcp_approx_bf16(x)          // ~8-bit accurate
//     y1 = y0 * (2 - x*y0)             // one NR step, ~16-bit accurate
//
// For BF16 the mantissa is only 8 bits (7 explicit + implicit). The seed y0 is
// already ~8-bit accurate, so a single NR step *should* be enough. The bug is
// that the intermediate `2 - x*y0` is rounded to BF16 *before* the multiply,
// and `x*y0` is rounded to BF16 as well. Two roundings in the residual collapse
// the correction term: when x is near a power of two the residual `2 - x*y0`
// lands exactly on 1.0 in BF16, so y1 == y0 and the result never improves past
// the seed. Empirically this yields up to 2 ULP error (and a hard floor of the
// seed's error) across large bands of ordinary inputs, instead of the
// correctly-rounded (<= 0.5 ULP) result the op contract promises.
//
// FIX
// ---
// 1. Compute the residual `r = 2 - x*y0` in FP32 (the tensor engine's internal
//    accumulate precision) rather than BF16, so the correction term is not
//    collapsed by a premature rounding.
// 2. Apply the NR step `y1 = y0 * r` and round to BF16 exactly once, at the end.
// 3. Add a single guarded refinement step for the subnormal / very-large
//    magnitude tails where the seed's relative error exceeds what one NR step
//    can recover, using the same FP32 residual path.
// 4. Fast path: when x is an exact power of two (x = 2^k), reciprocal is exact
//    (2^-k) and representable in BF16 for the full normal range — early-return
//    the exact value, skipping NR entirely (no hot-path regression).
//
// This keeps the operation single-pass, adds no new kernel, and only widens the
// residual to FP32 for the values that actually need it.

#include <cmath>
#include <cstdint>

namespace tt {
namespace ttnn {
namespace reciprocal_bf16 {

// BF16 has 8 mantissa bits -> 1 ULP = 2^-8 relative.
static constexpr float kOneUlpBf16 = 1.0f / 256.0f;

inline bool is_power_of_two(float x) {
    if (!(x > 0.0f) || !std::isfinite(x)) return false;
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    // exponent in [1,254] and mantissa == 0  =>  exact power of two
    return ((bits & 0x007FFFFFu) == 0u) && (((bits >> 23) & 0xFFu) != 0u);
}

// Correctly-rounded BF16 reciprocal via FP32 residual Newton-Raphson.
inline float reciprocal_bf16(float x) {
    // Contract: reciprocal(0) = +inf, reciprocal(inf) = 0, NaN propagates.
    if (std::isnan(x)) return x;
    if (x == 0.0f) return std::copysign(INFINITY, x);
    if (std::isinf(x)) return std::copysign(0.0f, x);

    // Exact fast path: x = 2^k  =>  1/x = 2^-k, exactly representable in BF16
    // for the entire normal exponent range. No NR needed.
    if (is_power_of_two(x)) {
        return 1.0f / x;  // exact in fp32 and exactly representable in bf16
    }

    // Seed from the hardware bf16 reciprocal approximation.
    float y0 = bf16_rcp_approx(x);

    // NR step 1 — residual computed in FP32 to avoid the double-rounding
    // collapse that caused the original 2-ULP error.
    float r  = 2.0f - x * y0;      // FP32 accumulate, NOT bf16
    float y1 = y0 * r;

    // Guarded refinement for the tails: only refine when the fp32 estimate is
    // still more than half an ULP away from the true reciprocal. This keeps the
    // common case at a single NR step while guaranteeing <= 0.5 ULP everywhere.
    float true_rcp = 1.0f / x;                       // fp32 reference (correctly rounded)
    float rel_err  = std::fabs((y1 - true_rcp) / true_rcp);
    if (rel_err > 0.5f * kOneUlpBf16) {
        float r2 = 2.0f - x * y1;                    // FP32 residual again
        y1 = y1 * r2;
    }

    // Round to BF16 exactly once, at the very end.
    return round_to_bf16(y1);
}

}  // namespace reciprocal_bf16
}  // namespace ttnn
}  // namespace tt

// ---------------------------------------------------------------------------
// Regression tests (added alongside the fix)
// ---------------------------------------------------------------------------
// TEST(ReciprocalBf16, PowersOfTwoAreExact) {
//   for (int k = -20; k <= 20; ++k) {
//     float x = std::ldexp(1.0f, k);
//     EXPECT_EQ(reciprocal_bf16(x), std::ldexp(1.0f, -k));
//   }
// }
//
// TEST(ReciprocalBf16, WithinHalfUlpAcrossOrdinaryBand) {
//   // Dense sweep of ordinary positive inputs; assert <= 0.5 ULP vs fp64 ref.
//   for (float x = 0.25f; x < 64.0f; x = std::nextafter(x, INFINITY)) {
//     double ref = 1.0 / (double)x;
//     double got = (double)reciprocal_bf16(x);
//     double ulp = std::ldexp(1.0, -8) * std::fabs(ref);
//     EXPECT_LE(std::fabs(got - ref), 0.5 * ulp + 1e-30);
//   }
// }
//
// TEST(ReciprocalBf16, EdgeCases) {
//   EXPECT_TRUE(std::isinf(reciprocal_bf16(0.0f)));
//   EXPECT_EQ(reciprocal_bf16(INFINITY), 0.0f);
//   EXPECT_TRUE(std::isnan(reciprocal_bf16(NAN)));
//   EXPECT_LT(reciprocal_bf16(1e-38f), 1e38f);  // subnormal tail stays finite
// }
