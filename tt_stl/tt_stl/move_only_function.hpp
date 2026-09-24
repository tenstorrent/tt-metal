// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include <zoo/FunctionPolicy.h>

namespace ttsl {

namespace detail {

// Parity with std::function, which also stores 2 pointers inline on libstdc++, so replacing one
// with the other regresses nothing on size. Raise it here if profiling justifies it: a larger
// buffer keeps bigger captures off the heap at the cost of sizeof on every instance.
inline constexpr std::size_t kMoveOnlyFunctionInlinePointers = 2;

}  // namespace detail

// A move-only type-erased callable: what std::function would be without the copyability nobody
// uses. Holds move-only captures such as std::unique_ptr directly, instead of forcing them into a
// shared_ptr to satisfy std::function's copy requirement.
//
// Chosen over fu2 on measured results; see GitHub #57444.
//
// Two things about this spelling are load-bearing and not obvious:
//
//   - zoo::Function, not zoo::VTableFunction. The latter is the more natural-looking alias, but it
//     is an AnyContainer, which has neither operator bool nor has_value(). Both live on
//     zoo::Function.
//
//   - The RTTI affordance is not optional. Without it operator bool falls back to comparing the
//     vtable's destroy pointer against Destroy::noOp; for a trivially destructible target -- a
//     captureless lambda, a function pointer -- those two functions are byte-identical, so a linker
//     doing identical code folding merges them and an engaged function reports itself empty. With
//     the affordance operator bool goes through type(), which cannot collapse. The affordance costs
//     nothing per object: it adds one pointer to the per-type static vtable, not to the instance.
//
// Note zoo does not compile under -fno-rtti, with or without that affordance. Device kernels are
// compiled -fno-rtti but do not include tt_stl, so this only constrains host code.
//
// Calling an empty instance throws zoo::bad_function_call, matching std::function.
//
// TODO(#57444): switch to std::move_only_function when tt-metal moves to C++23. Deliberately a
// one-line change here rather than a feature-macro guard, because it is not behaviour-neutral: an
// empty call becomes undefined rather than throwing, and the inline capacity becomes
// implementation-defined.
template <typename Signature>
using move_only_function = zoo::Function<
    zoo::AnyContainer<
        zoo::Policy<void * [detail::kMoveOnlyFunctionInlinePointers], zoo::Destroy, zoo::Move, zoo::RTTI>>,
    Signature>;

}  // namespace ttsl
