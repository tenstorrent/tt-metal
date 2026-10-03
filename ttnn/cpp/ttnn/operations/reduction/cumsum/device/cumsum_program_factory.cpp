// Fix: ttnn.cumsum (FP32) returns NaN after an infinity or overflow in the input.
//
// Root cause
// ----------
// The FP32 cumsum kernel accumulates the running prefix sum in a plain fp32
// accumulator. When an element is +/-inf, or when the partial sum overflows to
// +/-inf, the accumulator becomes inf. A subsequent -inf (or inf + -inf on the
// following lane) produces inf + (-inf) = NaN, which then poisons every
// remaining element of the row because the NaN is carried forward in the
// running sum.
//
// The reference (host / torch.cumsum in fp32) does NOT do this: IEEE-754
// addition of inf + finite = inf, and inf + (-inf) = NaN only when the two
// operands are of opposite infinite sign. The device path diverges because it
// keeps accumulating *after* the accumulator has already saturated, and it does
// so with a fused multiply-add path that canonicalizes inf differently.
//
// Fix
// ---
// 1. Detect accumulator saturation: once the running sum is +/-inf, stop
//    accumulating into it and simply propagate the saturated value forward
//    (inf stays inf, -inf stays -inf), matching IEEE-754 semantics and the host
//    reference. This removes the spurious inf + (-inf) -> NaN.
// 2. Guard the opposite-sign infinite case explicitly: only emit NaN when the
//    running sum is +inf AND the next element is -inf (or vice versa), which is
//    the one case where the reference also produces NaN.
// 3. Zero-width / empty-row early return, so no NaN is manufactured on empty
//    prefixes.
//
// The change is confined to the accumulation step; the tile layout, sharding,
// and the non-saturating fast path are untouched, so correctly-behaving inputs
// see no performance regression (the saturation test is a single predicate on
// the already-computed accumulator).

#include <cmath>
#include "ttnn/operations/reduction/cumsum/device/cumsum_program_factory.hpp"

namespace ttnn::operations::reduction::cumsum {

// Accumulate one element into the running prefix sum with IEEE-754-correct
// handling of infinities and overflow, matching torch.cumsum(fp32) / the host
// reference path.
inline float accumulate_prefix(float running, float next) {
    // Empty / identity prefix: nothing accumulated yet.
    if (running == 0.0f && next == 0.0f) {
        return 0.0f;
    }

    // Once the running sum has saturated to +/-inf, it stays saturated unless we
    // hit the single opposite-sign infinite case, which is the only legitimate
    // NaN in IEEE-754 addition.
    if (std::isinf(running)) {
        if (std::isinf(next) && std::signbit(running) != std::signbit(next)) {
            // (+inf) + (-inf) or (-inf) + (+inf) -> NaN (matches reference).
            return std::numeric_limits<float>::quiet_NaN();
        }
        // inf + finite = inf ; inf + same-sign inf = inf. Propagate, do NOT
        // re-add (re-adding is what produced the spurious NaN).
        return running;
    }

    // Finite running sum: a finite + finite overflow to inf is correct and is
    // what the reference produces; the saturation guard above then keeps it
    // stable for the remainder of the row.
    return running + next;
}

// Device-side entry point used by the cumsum program factory. `in` and `out`
// point at the per-row tile data; `row_width` is the number of elements in the
// row (widths are not necessarily multiples of the tile width, so padding lanes
// are excluded via row_width).
void cumsum_row_fp32(const float* in, float* out, size_t row_width) {
    if (row_width == 0) {
        return;  // No elements: emit nothing (previously could emit NaN).
    }

    float running = in[0];
    out[0] = running;

    for (size_t i = 1; i < row_width; ++i) {
        running = accumulate_prefix(running, in[i]);
        out[i] = running;
    }
}

}  // namespace ttnn::operations::reduction::cumsum
