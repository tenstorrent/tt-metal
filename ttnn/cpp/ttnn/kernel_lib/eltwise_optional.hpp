// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_optional.hpp
 * @brief Conditional / optional chain element wrappers.
 *
 * Compile-time conditional: `OptionalChainElement<bool COND, Inner>` forwards to `Inner`
 * when COND is true and is a no-op marker (dropped from the chain) when false. Its variadic
 * ctor swallows Inner's args, so `OptionalChainElement<COND, FillScalar>{0.5f}` compiles
 * for either COND.
 *
 * Runtime conditional: template the chain-running function on a `bool` and dispatch from
 * `kernel_main`:
 *
 *     template <bool DO_MASK> inline void run_op(uint32_t n) {
 *         eltwise_chain(EltwiseShape::tiles(n), CopyTile<...>{},
 *             OptionalChainElement<DO_MASK, MaskInject<...>>{}, SfpuOp<...>{}, PackTile<...>{});
 *     }
 *     void kernel_main() {
 *         if (get_arg_val<uint32_t>(0) != 0) run_op<true>(n); else run_op<false>(n);
 *     }
 *
 * Mid-loop per-iteration runtime conditions (`if (col == Wt-1) ...`) are NOT supported —
 * they need a per-iter runtime branch inside the chain, which collides with its
 * compile-time dispatch. Use a separate chain invocation for the conditional iteration.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

/// Conditional chain element — when COND is false every hook is a no-op.
template <bool COND, class Inner>
struct OptionalChainElement;

// COND == true: forward to Inner.
template <class Inner>
struct OptionalChainElement<true, Inner> : Inner {
    using Inner::Inner;
};

// COND == false: a trivial inert marker. `eltwise_chain` strips every element carrying
// `is_disabled == true` from the pack before any stage runs, so the disabled element
// never reaches a trait scan or a runtime hook — no tag
// impersonation, no CB-id / pack-slot / lifecycle stubs, no no-op hooks are needed. This
// is self-checking: if a future stage forgets to drop it, the missing member fails to
// compile loudly rather than silently misbehaving.
template <class Inner>
struct OptionalChainElement<false, Inner> {
    static constexpr bool is_disabled = true;

    // Variadic ctor swallows any args the caller would pass to the inner element so
    // kernel-side construction `OptionalChainElement<false, Inner>{...}` compiles
    // unchanged regardless of COND. Provides a default ctor too.
    constexpr OptionalChainElement() noexcept = default;
    template <class... Ignored>
    constexpr explicit OptionalChainElement(Ignored&&...) noexcept {}
};

}  // namespace compute_kernel_lib
