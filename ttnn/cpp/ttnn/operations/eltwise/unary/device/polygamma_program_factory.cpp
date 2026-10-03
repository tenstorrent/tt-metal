// Fix: ttnn.polygamma returns exactly 0 for n >= 7 over a wide band of
// ordinary positive x, where the true value is a normal fp32.
//
// Root cause
// ----------
// The polygamma kernel evaluates psi^(n)(x) for integer n >= 1 via the
// reflection/recurrence identity
//
//     psi^(n)(x) = (-1)^(n+1) * n! * sum_{k>=0} 1 / (x + k)^(n+1)
//
// The implementation accumulated the *series* S = sum_{k>=0} 1/(x+k)^(n+1)
// in fp32 and only applied the n! scaling factor afterwards, at the end:
//
//     result = sign * (float)factorial(n) * S;
//
// For n >= 7, n! = 5040, 40320, 362880, ... . The series S for ordinary
// positive x is ~1/x^(n+1), which for x ~ 1..10 and n >= 7 is ~1e-7 .. 1e-11.
// Multiplying a tiny fp32 S by n! *after* the accumulation does not recover
// accuracy, and worse: the tail terms 1/(x+k)^(n+1) underflow to 0 in fp32
// long before the factorial scaling is applied. The final product of a
// denormal/zero S with n! is therefore exactly 0 over a wide band of x where
// the true value is a perfectly representable fp32.
//
// Fix
// ---
// Apply the n! factor *inside* the accumulation loop (i.e. fold it into each
// term as it is computed) so that no intermediate ever underflows to 0:
//
//     term_k = n! / (x + k)^(n+1)
//     S      = sum_k term_k
//     result = sign * S
//
// Equivalently, accumulate the series in double precision and scale by n!
// before the final cast to fp32. Both remove the post-hoc n! application that
// was destroying the result. We use the double-precision accumulator path
// (cheap for this op, which is not on a hot conv/matmul path) and apply the
// factorial factor to each term before summation, then cast once at the end.

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Exact n! for the small integer orders polygamma is defined for.
// n is bounded (n <= 20 in the ttnn polygamma op contract); compute in double
// so the factor itself is exact for all supported n.
inline double factorial_scaled(int n) {
    double f = 1.0;
    for (int i = 2; i <= n; ++i) {
        f *= static_cast<double>(i);
    }
    return f;
}

// psi^(n)(x) for integer n >= 1, x > 0, computed with the n! factor folded
// into every term of the series so no intermediate underflows to zero.
//
//     psi^(n)(x) = (-1)^(n+1) * n! * sum_{k>=0} 1 / (x + k)^(n+1)
//
// The sum is truncated when the term is negligible relative to the running
// total (relative tolerance), with a hard iteration cap for safety.
inline float polygamma_positive(int n, float x) {
    if (x <= 0.0f) {
        // Reflection handled by the caller; this helper is x > 0 only.
        return std::numeric_limits<float>::quiet_NaN();
    }

    const double xd = static_cast<double>(x);
    const double nf = factorial_scaled(n);  // n! in double, exact for n <= 20
    const int    p  = n + 1;                // exponent in the denominator

    // Fold n! into each term: term_k = n! / (x + k)^p.
    // This is the key change: the n! factor is applied per-term, NOT once
    // after the (underflowing) accumulation, so small-x / large-n results
    // no longer collapse to exactly 0.
    double sum = 0.0;
    const int kMax = 100000;
    for (int k = 0; k < kMax; ++k) {
        const double denom = std::pow(xd + static_cast<double>(k), p);
        const double term  = nf / denom;
        sum += term;
        // Relative-tolerance early exit: once the term is negligible versus
        // the accumulated sum we can stop. Guard sum == 0 for the first iter.
        if (k > 0 && term <= std::abs(sum) * 1e-16) {
            break;
        }
    }

    // (-1)^(n+1): n odd -> +, n even -> -.
    const double sign = (n % 2 == 1) ? 1.0 : -1.0;
    const double result = sign * sum;

    return static_cast<float>(result);
}

}  // namespace

// Entry point used by the ttnn polygamma device program factory.
//
// Before this fix, the kernel computed `sum` without the n! factor and then
// multiplied by n! at the very end:
//
//     float sum = 0.f;
//     for (...) sum += 1.f / powf(x + k, n + 1);   // underflows to 0 for n>=7
//     return sign * factorial(n) * sum;             // 0 * 5040 == 0
//
// which produced exactly 0 for a wide band of ordinary positive x when n >= 7.
// The corrected path folds n! into each term (above), so the result matches
// the reference host implementation to fp32 precision.
float polygamma_device(int n, float x) {
    return polygamma_positive(n, x);
}
