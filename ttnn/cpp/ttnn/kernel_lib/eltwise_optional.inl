// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_optional.inl
 * @brief Implementation of compile-time optional and runtime-conditional elements.
 *
 * Included from `eltwise_optional.hpp`. Do NOT include directly.
 */

namespace compute_kernel_lib {
namespace detail {

template <class Inner>
constexpr bool runtime_conditional_element_supported() {
    if constexpr (is_runtime_conditional_op_v<Inner> || is_cb_writer_op_v<Inner>) {
        return false;
    } else if constexpr (is_dest_only_op_v<Inner>) {
        return true;
    } else if constexpr (is_binary_fpu_op_v<Inner>) {
        return !Inner::uses_dest_accumulation && Inner::APolicy == InputLifecycle::CallerManaged &&
               (Inner::same_dfb || Inner::BPolicy == InputLifecycle::CallerManaged);
    } else if constexpr (is_cb_reader_op_v<Inner>) {
        return Inner::Policy == InputLifecycle::CallerManaged;
    } else {
        return false;
    }
}

template <Side S, std::size_t Index, uint32_t Prev, class First, class... Rest>
constexpr uint32_t runtime_sequence_prev() {
    if constexpr (Index == 0) {
        return Prev;
    } else {
        constexpr uint32_t current = dfb_for_side<S, First>();
        constexpr uint32_t next = current == NO_PREV_DFB ? Prev : current;
        return runtime_sequence_prev<S, Index - 1, next, Rest...>();
    }
}

template <
    uint32_t SlotBase,
    uint32_t PrevA,
    uint32_t PrevB,
    uint32_t PrevP,
    bool PackHetero,
    class... Elements,
    std::size_t... Is>
ALWI void apply_runtime_branch(
    const RuntimeConditionalBranch<Elements...>& branch,
    std::index_sequence<Is...>,
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t Ht,
    uint32_t Wt) {
    // Runtime sequences cannot be boot-hoisted: the selected branch owns each
    // element's transition + init immediately before that element executes.
    (elem_apply_compute<
         true,
         true,
         SlotBase,
         runtime_sequence_prev<Side::SrcA, Is, PrevA, Elements...>(),
         runtime_sequence_prev<Side::SrcB, Is, PrevB, Elements...>(),
         runtime_sequence_prev<Side::Pack, Is, PrevP, Elements...>(),
         PackHetero,
         Elements>(std::get<Is>(branch.elements), i_flat, ht, wt, inner_count, chain_lane_width, Ht, Wt),
     ...);
}

}  // namespace detail

template <class... Branches>
constexpr RuntimeConditionalSequence<Branches...>::RuntimeConditionalSequence(
    uint32_t selected_branch_, std::tuple<Branches...> branches_) noexcept :
    selected_branch(selected_branch_), branches(branches_) {}

template <class... Branches>
template <uint32_t SlotBase, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero>
ALWI void RuntimeConditionalSequence<Branches...>::apply(
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t Ht,
    uint32_t Wt) const {
    apply_selected<0, SlotBase, PrevA, PrevB, PrevP, PackHetero>(i_flat, ht, wt, inner_count, chain_lane_width, Ht, Wt);
}

template <class... Branches>
template <std::size_t BranchIndex, uint32_t SlotBase, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero>
ALWI void RuntimeConditionalSequence<Branches...>::apply_selected(
    uint32_t i_flat,
    uint32_t ht,
    uint32_t wt,
    uint32_t inner_count,
    uint32_t chain_lane_width,
    uint32_t Ht,
    uint32_t Wt) const {
    if constexpr (BranchIndex < sizeof...(Branches)) {
        if (selected_branch == BranchIndex) {
            const auto& branch = std::get<BranchIndex>(branches);
            detail::apply_runtime_branch<SlotBase, PrevA, PrevB, PrevP, PackHetero>(
                branch,
                std::make_index_sequence<std::tuple_size_v<decltype(branch.elements)>>{},
                i_flat,
                ht,
                wt,
                inner_count,
                chain_lane_width,
                Ht,
                Wt);
        } else {
            apply_selected<BranchIndex + 1, SlotBase, PrevA, PrevB, PrevP, PackHetero>(
                i_flat, ht, wt, inner_count, chain_lane_width, Ht, Wt);
        }
    }
}

template <class... Branches>
constexpr RuntimeIfBuilder<Branches...>::RuntimeIfBuilder(
    uint32_t selected_branch_, std::tuple<Branches...> branches_) noexcept :
    selected_branch(selected_branch_), branches(branches_) {}

template <class... Branches>
template <class... Elements>
ALWI auto RuntimeIfBuilder<Branches...>::else_if(bool condition, Elements... elements) const {
    static_assert(sizeof...(Elements) > 0, "else_if requires at least one chain element");
    using Branch = detail::RuntimeConditionalBranch<Elements...>;
    constexpr uint32_t unmatched = sizeof...(Branches);
    const uint32_t next_selected =
        selected_branch == unmatched ? (condition ? unmatched : unmatched + 1) : selected_branch;
    return RuntimeIfBuilder<Branches..., Branch>{
        next_selected, std::tuple_cat(branches, std::tuple<Branch>{Branch{elements...}})};
}

template <class... Branches>
template <class... Elements>
ALWI auto RuntimeIfBuilder<Branches...>::otherwise(Elements... elements) const {
    static_assert(sizeof...(Elements) > 0, "otherwise requires at least one chain element");
    using Branch = detail::RuntimeConditionalBranch<Elements...>;
    return RuntimeConditionalSequence<Branches..., Branch>{
        selected_branch, std::tuple_cat(branches, std::tuple<Branch>{Branch{elements...}})};
}

template <class... Elements>
ALWI auto when(bool condition, Elements... elements) {
    static_assert(sizeof...(Elements) > 0, "when requires at least one chain element");
    using Branch = detail::RuntimeConditionalBranch<Elements...>;
    return RuntimeConditionalSequence<Branch>{condition ? 0u : 1u, std::tuple<Branch>{Branch{elements...}}};
}

template <class... Elements>
ALWI auto runtime_if(bool condition, Elements... elements) {
    static_assert(sizeof...(Elements) > 0, "runtime_if requires at least one chain element");
    using Branch = detail::RuntimeConditionalBranch<Elements...>;
    return RuntimeIfBuilder<Branch>{condition ? 0u : 1u, std::tuple<Branch>{Branch{elements...}}};
}

}  // namespace compute_kernel_lib
