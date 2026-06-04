// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_optional.hpp
 * @brief Conditional / optional chain element wrappers.
 *
 * Two patterns are supported:
 *
 *  1. **Compile-time conditional** — `OptionalChainElement<bool COND, Inner>`. When
 *     COND is true the wrapper inherits Inner's constructors AND its full type / tag,
 *     so the chain pipeline sees `Inner`. When COND is false the wrapper inherits
 *     Inner's tag (so chain traits classify it the same way) but every hook is a
 *     no-op. The wrapper's variadic constructor swallows any args the caller would
 *     pass to Inner — `OptionalChainElement<COND, FillScalar>{0.5f}` compiles for
 *     COND in {true, false}.
 *
 *  2. **Runtime conditional** — kernels that need to pick a chain shape based on a
 *     runtime arg use the standard "template the inner function on a `bool`" pattern
 *     and dispatch from `kernel_main`:
 *
 *         template <bool DO_MASK>
 *         inline void run_op(uint32_t n) {
 *             eltwise_chain(n,
 *                 CopyTile<...>{},
 *                 OptionalChainElement<DO_MASK, MaskInject<...>>{},
 *                 SfpuOp<...>{},
 *                 PackTile<...>{});
 *         }
 *         void kernel_main() {
 *             const bool do_mask = get_arg_val<uint32_t>(0) != 0;
 *             if (do_mask) run_op<true>(n); else run_op<false>(n);
 *         }
 *
 * @section optional_caveats Caveats
 *
 * Mid-loop per-iteration runtime conditions (`if (col_idx == Wt-1) mask_tile(...)`)
 * are NOT supported by these helpers — they require a per-iter runtime branch
 * inside the chain pipeline, which collides with the chain's compile-time dispatch.
 * Use a separate chain invocation for the conditional iteration if needed.
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
