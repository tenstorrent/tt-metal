// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_optional.inl
 * @brief Implementation of conditional / optional eltwise chain elements.
 *
 * Included from `eltwise_optional.hpp`. Do NOT include directly.
 */

namespace compute_kernel_lib {
namespace detail {

template <class Inner>
constexpr bool runtime_conditional_element_supported() {
    if constexpr (is_runtime_conditional_op_v<Inner>) {
        return false;
    } else if constexpr (is_dest_only_op_v<Inner>) {
        return true;
    } else if constexpr (is_binary_fpu_op_v<Inner>) {
        return Inner::APolicy == InputLifecycle::CallerManaged &&
               (Inner::same_dfb || Inner::BPolicy == InputLifecycle::CallerManaged);
    } else if constexpr (is_cb_reader_op_v<Inner>) {
        return Inner::Policy == InputLifecycle::CallerManaged;
    } else {
        return false;
    }
}

}  // namespace detail

template <class Inner>
constexpr RuntimeConditionalChainElement<Inner>::RuntimeConditionalChainElement(bool enabled_, Inner inner) noexcept :
    Inner(inner), enabled(enabled_) {}

template <class Inner>
ALWI void RuntimeConditionalChainElement<Inner>::init() const {
    if (enabled) {
        static_cast<const Inner&>(*this).init();
    }
}

template <class Inner>
template <class... Args>
ALWI void RuntimeConditionalChainElement<Inner>::exec(Args... args) const {
    if (enabled) {
        static_cast<const Inner&>(*this).exec(args...);
    }
}

template <std::size_t ArmCount>
template <std::size_t Arm, class Inner>
ALWI auto RuntimeConditional<ArmCount>::when(Inner inner) const {
    static_assert(Arm < ArmCount, "Runtime conditional arm index is out of range");
    return RuntimeConditionalChainElement<Inner>{selected_arm == Arm, inner};
}

template <std::size_t ArmCount>
template <std::size_t... Arms, class Inner>
ALWI auto RuntimeConditional<ArmCount>::when_any(Inner inner) const {
    static_assert(sizeof...(Arms) > 0, "when_any requires at least one arm index");
    static_assert(((Arms < ArmCount) && ...), "Runtime conditional arm index is out of range");
    return RuntimeConditionalChainElement<Inner>{((selected_arm == Arms) || ...), inner};
}

template <std::size_t ArmCount>
template <class Inner>
ALWI auto RuntimeConditional<ArmCount>::otherwise(Inner inner) const {
    return RuntimeConditionalChainElement<Inner>{selected_arm == ArmCount, inner};
}

template <class... Conditions>
ALWI auto runtime_conditional(Conditions... conditions) {
    static_assert(sizeof...(Conditions) > 0, "runtime_conditional requires at least one condition");
    constexpr uint32_t arm_count = sizeof...(Conditions);
    const bool matches[arm_count] = {static_cast<bool>(conditions)...};
    uint32_t selected_arm = arm_count;
    for (uint32_t arm = 0; arm < arm_count; ++arm) {
        if (selected_arm == arm_count && matches[arm]) {
            selected_arm = arm;
        }
    }
    return RuntimeConditional<arm_count>{selected_arm};
}

template <class Inner>
ALWI auto runtime_optional(bool condition, Inner inner) {
    return RuntimeConditionalChainElement<Inner>{condition, inner};
}

}  // namespace compute_kernel_lib
