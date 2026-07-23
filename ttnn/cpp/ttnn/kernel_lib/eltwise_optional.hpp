// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_optional.hpp
 * @brief Compile-time optional and runtime-conditional chain elements.
 *
 * Compile-time optional:
 *
 *     OptionalChainElement<COND, Op>
 *
 * Runtime conditionals use one surface:
 *
 *     runtime_if(condition_a, OpA{}, OpB{})
 *         .else_if(condition_b, OpB{}, OpC{})
 *         .otherwise(DefaultOp{})
 *
 * `runtime_if(condition, ...)` and an if / else-if chain may be used directly;
 * their unmatched case performs no element work. `.otherwise(...)` adds an
 * explicit final else. Branch selection is ordered and exclusive: the first true
 * branch wins. Every branch is one chain position, so one runtime decision guards
 * the whole element sequence.
 *
 * Runtime-conditional sequences deliberately support only DEST-only operations
 * and caller-managed CB readers. Writers and driver-managed waits/pops are rejected:
 * their lifecycle cannot be conservatively planned across a disabled branch.
 */

namespace compute_kernel_lib {

/// Shared inert, tag-less chain position used by every disabled compile-time optional.
struct DisabledChainElement {
    constexpr DisabledChainElement() noexcept = default;

    template <class... Ignored>
    constexpr explicit DisabledChainElement(Ignored&&...) noexcept {}
};

/// Enabled optionals are exactly `Inner`; disabled optionals share one short marker type.
template <bool COND, class Inner>
using OptionalChainElement = std::conditional_t<COND, Inner, DisabledChainElement>;

namespace detail {

template <class Inner>
constexpr bool runtime_conditional_element_supported();

template <class... Elements>
struct RuntimeConditionalBranch {
    static_assert(sizeof...(Elements) > 0, "A runtime conditional branch requires at least one element");
    static_assert(
        (runtime_conditional_element_supported<Elements>() && ...),
        "Runtime conditional elements must be DEST-only or use caller-managed input lifecycles");

    std::tuple<Elements...> elements;

    constexpr explicit RuntimeConditionalBranch(Elements... elements_) noexcept : elements(elements_...) {}
};

template <class Branch>
struct RuntimeConditionalBranchTraits;

template <class... Elements>
struct RuntimeConditionalBranchTraits<RuntimeConditionalBranch<Elements...>> {
    static constexpr bool any_reader = (is_cb_reader_op_v<Elements> || ...);
    static constexpr bool touches_srca = ((dfb_for_side<Side::SrcA, Elements>() != NO_PREV_DFB) || ...);
    static constexpr bool touches_srcb = ((dfb_for_side<Side::SrcB, Elements>() != NO_PREV_DFB) || ...);
    static constexpr uint32_t lane_width = []() {
        uint32_t result = 1;
        ((result = result < elem_lane_width_v<Elements> ? elem_lane_width_v<Elements> : result), ...);
        return result;
    }();
};

template <class... Branches>
struct RuntimeConditionalSequenceTraits {
    static constexpr bool any_reader = (RuntimeConditionalBranchTraits<Branches>::any_reader || ...);
    static constexpr bool touches_srca = (RuntimeConditionalBranchTraits<Branches>::touches_srca || ...);
    static constexpr bool touches_srcb = (RuntimeConditionalBranchTraits<Branches>::touches_srcb || ...);
    static constexpr uint32_t lane_width = []() {
        uint32_t result = 1;
        ((result = result < RuntimeConditionalBranchTraits<Branches>::lane_width
                       ? RuntimeConditionalBranchTraits<Branches>::lane_width
                       : result),
         ...);
        return result;
    }();
};

template <bool AnyReader>
struct RuntimeConditionalSequenceBase;

template <>
struct RuntimeConditionalSequenceBase<false> : DestOnlyTag {};

template <>
struct RuntimeConditionalSequenceBase<true> : CbReaderTag {
    static constexpr InputLifecycle Policy = InputLifecycle::CallerManaged;
    static constexpr uint32_t dfb = INVALID_DFB;
    static constexpr bool is_upfront = false;

    static constexpr uint32_t dfb_a_id() { return INVALID_DFB; }
    static constexpr InputLifecycle a_policy() { return Policy; }
};

}  // namespace detail

/// One runtime-selected chain position. `selected_branch == sizeof...(Branches)`
/// means no branch is selected, which is how an unmatched `runtime_if(...)` becomes inert.
template <class... Branches>
struct RuntimeConditionalSequence
    : detail::RuntimeConditionalSequenceBase<detail::RuntimeConditionalSequenceTraits<Branches...>::any_reader>,
      RuntimeConditionalTag,
      RuntimeConditionalSequenceTag {
    static_assert(sizeof...(Branches) > 0, "A runtime conditional requires at least one branch");

    using Traits = detail::RuntimeConditionalSequenceTraits<Branches...>;

    // A conditional reader may or may not change each input-side format. Poison the
    // outer previous-CB fold so a following reader reconfigures conservatively.
    static constexpr uint32_t reconfig_srca_dfb = Traits::touches_srca ? CONDITIONAL_DFB : NO_PREV_DFB;
    static constexpr uint32_t reconfig_srcb_dfb = Traits::touches_srcb ? CONDITIONAL_DFB : NO_PREV_DFB;
    static constexpr uint32_t lane_width = Traits::lane_width;

    uint32_t selected_branch;
    std::tuple<Branches...> branches;

    constexpr RuntimeConditionalSequence(uint32_t selected_branch_, std::tuple<Branches...> branches_) noexcept;

    template <uint32_t SlotBase, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero>
    ALWI void apply(
        uint32_t i_flat,
        uint32_t ht,
        uint32_t wt,
        uint32_t inner_count,
        uint32_t chain_lane_width,
        uint32_t Ht,
        uint32_t Wt) const;

private:
    template <
        std::size_t BranchIndex,
        uint32_t SlotBase,
        uint32_t PrevA,
        uint32_t PrevB,
        uint32_t PrevP,
        bool PackHetero>
    ALWI void apply_selected(
        uint32_t i_flat,
        uint32_t ht,
        uint32_t wt,
        uint32_t inner_count,
        uint32_t chain_lane_width,
        uint32_t Ht,
        uint32_t Wt) const;
};

/// Open ordered conditional. It is already a valid chain element with an implicit
/// empty else, and may be extended with `.else_if(...)` or `.otherwise(...)`.
template <class... Branches>
struct RuntimeIfBuilder : RuntimeConditionalSequence<Branches...> {
    using Base = RuntimeConditionalSequence<Branches...>;

    constexpr RuntimeIfBuilder(uint32_t selected_branch_, std::tuple<Branches...> branches_) noexcept;

    template <class... Elements>
    ALWI auto else_if(bool condition, Elements... elements) const;

    template <class... Elements>
    ALWI auto otherwise(Elements... elements) const;
};

/// Start an ordered, exclusive runtime if / else-if / else chain. The result may
/// be used directly, in which case an unmatched condition performs no element work.
template <class... Elements>
ALWI auto runtime_if(bool condition, Elements... elements);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.inl"
