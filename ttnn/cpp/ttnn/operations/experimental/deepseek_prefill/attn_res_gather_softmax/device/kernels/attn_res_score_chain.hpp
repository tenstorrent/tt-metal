// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What the AttnRes scoring needs on top of `eltwise_chain`, shared by every AttnRes compute
// kernel that scores.
//
// The scoring folds over tiles — a max or a sum across candidates, a sum across
// tensor-parallel partials — inside one dest session, in fp32 dest on the SFPU. A chain's
// elements are a flat list, so a fold is spelled out once per tile it reads: `unroll<N>(step)`
// concatenates the elements `step(i)` returns for each i in [0, N), and `eltwise_chain_of`
// runs a chain over such pieces. N is a compile-time count (candidates, partials), so the
// result is the same chain as one written out by hand.
//
// A fold reads several tiles of one buffer, one element apiece, so only one of those elements
// may own the buffer's window: `tile_at` is a tile read at its element's offset with the
// window left to the caller or to the owner, and `owning_tile_at` the owner, whose upfront wait
// at offset k covers tiles [0, k]. Giving ownership to the element that reads the last tile
// waits for the whole window.

#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace attn_res {

namespace detail {

template <typename Step, uint32_t... Is>
ALWI auto unroll(Step step, std::integer_sequence<uint32_t, Is...>) {
    return std::tuple_cat(step(std::integral_constant<uint32_t, Is>{})...);
}

}  // namespace detail

// `step(i)` for every i in [0, N), each a tuple of chain elements, concatenated in order.
// `i` is a std::integral_constant, so it can pick element types.
template <uint32_t N, typename Step>
ALWI auto unroll(Step step) {
    return detail::unroll(step, std::make_integer_sequence<uint32_t, N>{});
}

// One eltwise_chain over the elements of `parts`, in order.
template <typename... Parts>
ALWI void eltwise_chain_of(compute_kernel_lib::IterationShape shape, Parts... parts) {
    std::apply(
        [shape](auto... elts) __attribute__((always_inline)) { compute_kernel_lib::eltwise_chain(shape, elts...); },
        std::tuple_cat(parts...));
}

// A tile of `cb` at its element's offset: `CopyTile<tile_at(cb)>{buf, offset}`.
constexpr compute_kernel_lib::InputSpec tile_at(uint32_t cb) {
    using namespace compute_kernel_lib;
    return input(
        cb,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Scalar,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset);
}

// The same, owning `cb`'s window: waits for tiles [0, offset] before the chain, and with
// PopPolicy::AtEnd pops them after it.
constexpr compute_kernel_lib::InputSpec owning_tile_at(uint32_t cb, compute_kernel_lib::PopPolicy pop) {
    using namespace compute_kernel_lib;
    return input(
        cb, WaitPolicy::Upfront, pop, InputTileMapping::Scalar, DataFormatReconfig::Enabled, TileAddressing::Offset);
}

}  // namespace attn_res
