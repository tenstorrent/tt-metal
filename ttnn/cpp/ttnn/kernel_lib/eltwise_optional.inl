// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace compute_kernel_lib {

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
