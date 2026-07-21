// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

/**
 * @file eltwise_optional.hpp
 * @brief Conditional / optional chain element wrappers.
 *
 * Compile-time conditional: `OptionalChainElement<bool COND, Inner>` is `Inner`
 * when COND is true. When false, it is one shared tag-less inert marker that remains in the chain;
 * the chain describes it with neutral traits and emits no work for it. Its variadic ctor
 * swallows Inner's args, so `OptionalChainElement<COND, FillScalar>{0.5f}` compiles for
 * either COND.
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

namespace compute_kernel_lib {

/// Shared inert, tag-less chain position used by every disabled optional. Its variadic
/// constructor accepts the same runtime arguments the enabled inner element would receive.
struct DisabledChainElement {
    constexpr DisabledChainElement() noexcept = default;

    template <class... Ignored>
    constexpr explicit DisabledChainElement(Ignored&&...) noexcept {}
};

/// Enabled optionals are exactly `Inner`; disabled optionals all share one short marker type.
/// The inner type therefore does not leak into disabled worker/debug specializations.
template <bool COND, class Inner>
using OptionalChainElement = std::conditional_t<COND, Inner, DisabledChainElement>;

}  // namespace compute_kernel_lib
