// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// div compute (divide_round_block): NEO t of C computes this cluster's tiles t, t+C, ... (its slice of every
// STRIDED ring), `block_tiles` of them per eltwise_chain call, the last block ragged.
//
// One chain per block, everything in DEST:
//   dividend tile -> D0 (CopyTile)
//   divisor       -> D1 (CopyTile of the divisor ring, or FillBitcast of the number's float32 bits)
//   D0 = D0 / D1        (DivBinary: a * reciprocal(b) with Newton steps, 0/0 -> NaN, x/0 -> +-inf, x/x -> 1)
//   [Floor | Trunc](D0) (rounding_mode 1 | 2, on the quotient before it is stored)
//   D0 -> quotient ring (PackTile: one rounding into the output dtype)
// The number goes through the same div_binary_tile as a tensor, so it divides like a one-element tensor (R7).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/rounding.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t ROUND_NONE = 0;
constexpr uint32_t ROUND_FLOOR = 1;
constexpr uint32_t ROUND_TRUNC = 2;

constexpr auto DIVIDEND =
    ckl::input(dfb::in_a, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
#if !DIVISOR_IS_NUMBER
constexpr auto DIVISOR =
    ckl::input(dfb::in_b, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
#endif
constexpr auto QUOTIENT = ckl::output(dfb::out, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);

// One block: load both operands, divide (or the probe's op), round, pack.
template <uint32_t rounding_mode, class LoadDividend, class LoadDivisor, class Op, class Pack>
ALWI void divide_round_block(
    ckl::IterationShape shape, LoadDividend load_a, LoadDivisor load_b, Op op, Pack pack) {
    if constexpr (rounding_mode == ROUND_FLOOR) {
        ckl::eltwise_chain(shape, load_a, load_b, op, ckl::Floor<ckl::Dst::D0>{}, pack);
    } else if constexpr (rounding_mode == ROUND_TRUNC) {
        ckl::eltwise_chain(shape, load_a, load_b, op, ckl::Trunc<ckl::Dst::D0>{}, pack);
    } else {
        ckl::eltwise_chain(shape, load_a, load_b, op, pack);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);
    constexpr uint32_t rounding_mode = get_arg(args::rounding_mode);
    constexpr uint32_t tag_threads = get_arg(args::tag_threads);  // test probe: odd NEOs multiply
    static_assert(rounding_mode <= ROUND_TRUNC, "rounding_mode: 0 none, 1 floor, 2 trunc");
    static_assert(block_tiles >= 1, "block_tiles must be at least 1");

    const uint32_t t = get_my_thread_id();
    const uint32_t my_tiles = num_tiles / C + (t < num_tiles % C ? 1u : 0u);

    DataflowBuffer dfb_dividend(dfb::in_a);
    DataflowBuffer dfb_quotient(dfb::out);
#if DIVISOR_IS_NUMBER
    const uint32_t divisor_bits = get_arg(args::divisor_bits);
    compute_kernel_hw_startup(dfb::in_a, dfb::out);
#else
    DataflowBuffer dfb_divisor(dfb::in_b);
    compute_kernel_hw_startup(dfb::in_a, dfb::in_b, dfb::out);
#endif

    for (uint32_t done = 0; done < my_tiles; done += block_tiles) {
        const uint32_t n = my_tiles - done < block_tiles ? my_tiles - done : block_tiles;
        const auto shape = ckl::IterationShape::tiles(n).block_size(block_tiles);
        const auto load_a = ckl::CopyTile<DIVIDEND, ckl::Dst::D0>{dfb_dividend};
#if DIVISOR_IS_NUMBER
        const auto load_b = ckl::FillBitcast<ckl::Dst::D1>{divisor_bits};
#else
        const auto load_b = ckl::CopyTile<DIVISOR, ckl::Dst::D1>{dfb_divisor};
#endif
        const auto pack = ckl::PackTile<QUOTIENT, ckl::Dst::D0>{dfb_quotient};
        if constexpr (tag_threads) {
            if (t & 1u) {
                divide_round_block<rounding_mode>(
                    shape, load_a, load_b, ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{}, pack);
                continue;
            }
        }
        divide_round_block<rounding_mode>(
            shape, load_a, load_b, ckl::DivBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{}, pack);
    }
}
