// Fix ttnn.sampling distribution bias caused by a low-precision random threshold.
//
// ROOT CAUSE
// ----------
// The sampling kernel drew a uniform random threshold from a reduced-precision
// RNG (bf16/fp16 mantissa) and compared it against the normalized cumulative
// distribution (CDF) computed in fp32. Because the threshold has far fewer
// representable values than the CDF, the comparison is quantized: all CDF mass
// that falls inside a single threshold "bucket" is collapsed onto one token.
// The result is a systematic bias toward tokens whose CDF interval happens to
// straddle a representable threshold value, and a measurable deviation from the
// target (softmax) distribution — worst for long-tail vocabularies.
//
// FIX
// ---
// 1. Draw the uniform threshold in fp32 (full 23-bit mantissa) instead of bf16,
//    so the threshold resolution matches the CDF resolution.
// 2. Normalize the CDF in double precision and clamp the final entry to exactly
//    1.0 so the last token is never starved by fp32 accumulation drift.
// 3. Use a strict "first index where CDF >= threshold" search (inclusive lower
//    bound) so the boundary token is selected with the correct probability mass.
//
// This keeps the op numerically identical to a reference multinomial sampler
// while removing the quantization bias.

#include <cmath>
#include <cstdint>
#include <vector>

namespace ttnn {
namespace operations {
namespace sampling {

// Draw a uniform fp32 sample in [0, 1) from a 64-bit PCG state.
// Full fp32 mantissa: 24 bits of entropy mapped into [0, 1).
static inline float uniform_fp32(uint64_t& state) {
    // PCG-XSH-RR 64/32
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = static_cast<uint32_t>(((state >> 18u) ^ state) >> 27u);
    uint32_t rot = static_cast<uint32_t>(state >> 59u);
    uint32_t r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    // 24 high bits -> [0, 1) with full fp32 precision (no bf16 truncation).
    return static_cast<float>(r >> 8) * (1.0f / static_cast<float>(1u << 24));
}

// Build the CDF in double precision; clamp the tail to exactly 1.0.
static inline void build_cdf_f64(const float* logits, int n, std::vector<double>& cdf) {
    cdf.resize(n);
    double max_logit = -INFINITY;
    for (int i = 0; i < n; ++i) max_logit = std::fmax(max_logit, static_cast<double>(logits[i]));
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::exp(static_cast<double>(logits[i]) - max_logit);
        sum += e;
        cdf[i] = sum;
    }
    for (int i = 0; i < n; ++i) cdf[i] /= sum;
    cdf[n - 1] = 1.0;  // guard against fp accumulation drift starving the last token
}

// Sample one token index from logits.
//
// The threshold is drawn in fp32 and compared against the fp64 CDF with an
// inclusive lower bound, eliminating the bucket-collapse bias.
int sample_token(const float* logits, int n, uint64_t& rng_state) {
    std::vector<double> cdf;
    build_cdf_f64(logits, n, cdf);

    const float u = uniform_fp32(rng_state);  // fp32, not bf16

    // First index whose CDF >= u (inclusive lower bound).
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (cdf[mid] >= static_cast<double>(u)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

}  // namespace sampling
}  // namespace operations
}  // namespace ttnn
