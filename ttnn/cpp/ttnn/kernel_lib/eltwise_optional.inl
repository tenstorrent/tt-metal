// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace compute_kernel_lib {

// COND == true: forward to Inner.
template <class Inner>
struct OptionalChainElement<true, Inner> : Inner {
    using Inner::Inner;
};

// COND == false: a trivial inert marker. It remains in the chain but inherits no operation
// tag and provides no CB, pack-slot, lifecycle, or execution members. The chain's guarded
// descriptor accessors therefore give it neutral values, and its operation-kind checks are
// all false, so it neither changes chain state nor emits work.
template <class Inner>
struct OptionalChainElement<false, Inner> {
    // Variadic ctor swallows any args the caller would pass to the inner element so
    // kernel-side construction `OptionalChainElement<false, Inner>{...}` compiles
    // unchanged regardless of COND. Provides a default ctor too.
    constexpr OptionalChainElement() noexcept = default;
    template <class... Ignored>
    constexpr explicit OptionalChainElement(Ignored&&...) noexcept {}
};

}  // namespace compute_kernel_lib
