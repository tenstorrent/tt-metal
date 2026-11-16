// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace kda {

// Copy Count tiles of `in` (buffer id In) to `out` (Out) and every buffer in `more` (More), as many tiles per DST
// acquire as fit. An UnpackToDest input in the outputs' format makes the copy bit-exact; `in` is consumed.
template <uint32_t Count, uint32_t In, uint32_t Out, uint32_t... More, typename... DFBMore>
FORCE_INLINE void copy_exact(DataflowBuffer& in, DataflowBuffer& out, DFBMore&... more) {
    static_assert((std::is_same_v<DFBMore, DataflowBuffer> && ...), "copy_exact outputs are DataflowBuffers");
    static_assert(sizeof...(More) == sizeof...(DFBMore), "copy_exact needs one buffer id per extra output");
    namespace ckl = compute_kernel_lib;
    constexpr auto source =
        ckl::input(In, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
    constexpr auto whole = [](uint32_t id) {
        return ckl::output(id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);
    };
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(Count).block_size(ckl::DEST_AUTO_LIMIT),
        ckl::CopyTile<source>{in},
        ckl::PackTile<whole(Out)>{out},
        ckl::PackTile<whole(More)>{more}...);
}

}  // namespace kda
