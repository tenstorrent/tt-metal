// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

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
 * Runtime conditional: `runtime_conditional(cond0, cond1, ...)` selects the first true
 * condition, like an ordered if / else-if / else chain. Its `when<I>(element)`,
 * `when_any<Is...>(element)`, and `otherwise(element)` methods wrap individual elements
 * while keeping the chain flat, so common prefix / suffix elements are instantiated once:
 *
 *     const auto branch = runtime_conditional(use_a, use_b);
 *     eltwise_chain(EltwiseShape::single(), CopyTile<input(...)>{},
 *         branch.when<0>(OpA{}), branch.when<1>(OpB{}), branch.otherwise(DefaultOp{}),
 *         PackTile<output(...)>{});
 *
 * A condition is fixed for one `eltwise_chain` invocation, but may change between calls,
 * including calls made once per loop iteration. Runtime-gated elements are deliberately
 * limited to DEST-only operations and caller-managed CB readers. Writers and driver-managed
 * waits/pops are rejected because their lifecycle must not run on a disabled arm.
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

namespace detail {

template <class Inner>
constexpr bool runtime_conditional_element_supported();

}  // namespace detail

/// Runtime-gated view of one ordinary chain element. Inheriting from Inner preserves
/// its compile-time chain traits and resource description; init/exec are the only gated
/// operations. Consequently, only elements with no driver-owned lifecycle are accepted.
template <class Inner>
struct RuntimeConditionalChainElement : Inner, RuntimeConditionalTag {
    static_assert(
        detail::runtime_conditional_element_supported<Inner>(),
        "Runtime conditional elements must be DEST-only or use caller-managed input lifecycles");

    bool enabled;

    constexpr RuntimeConditionalChainElement(bool enabled_, Inner inner) noexcept;

    ALWI void init() const;

    template <class... Args>
    ALWI void exec(Args... args) const;
};

/// Result of an ordered runtime condition selection. Arm `I` corresponds to condition `I`;
/// the implicit default arm has index `ArmCount`.
template <std::size_t ArmCount>
struct RuntimeConditional {
    uint32_t selected_arm;

    template <std::size_t Arm, class Inner>
    ALWI auto when(Inner inner) const;

    template <std::size_t... Arms, class Inner>
    ALWI auto when_any(Inner inner) const;

    template <class Inner>
    ALWI auto otherwise(Inner inner) const;
};

/// Select the first true condition. If none is true, select the implicit default arm.
template <class... Conditions>
ALWI auto runtime_conditional(Conditions... conditions);

/// Shorthand for an if-with-empty-else runtime conditional.
template <class Inner>
ALWI auto runtime_optional(bool condition, Inner inner);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.inl"
