// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_special.hpp"   // Where
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"      // FillBitcast / FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

// where(cond, true_value, false_value) — no-broadcast binary_ng path (WHERE_TTS / WHERE_TST).
//
// Original kernel (eltwise_where_no_bcast.cpp before migration) processed a block of
// `num_tiles_per_cycle` tiles per outer iter, with 3 DST slots per tile:
//   cond  -> i*3       (copy from cb_in0)
//   TTS: true tensor -> i*3+1 (copy from cb_in1), false scalar -> i*3+2 (fill)
//   TST: false tensor -> i*3+2 (copy from cb_in1), true scalar -> i*3+1 (fill)
//   where_tile<DF>(i*3, i*3+1, i*3+2, i*3); pack i*3 -> cb_out
// It waited/popped `num_tiles_per_cycle` per outer iter on each input CB (the intermediate
// CBs are only sized for num_tiles_per_cycle tiles), looping `num_tiles` times.
//
// Chain mapping: EltwiseShape::tiles(num_tiles, num_tiles_per_cycle) gives outer=num_tiles,
// block_size=num_tiles_per_cycle. The chain lane width is 3 (max DST slot + 1), so lane j
// writes to slot base + j*3 — reproducing the original i*3 stride exactly.
//   - cond/tensor: CopyTile<..., InputLifecycle::Chunked, OperandKind::Block>. Chunked
//     waits block_size per outer iter and pops block_size per outer iter — matching the
//     original cb_wait_front/cb_pop_front(num_tiles_per_cycle). CopyTileReconfig::Input
//     emits copy_tile_to_dst_init_short(cb) per CB, matching the original's two init_shorts.
//   - scalar fill: FillInt (int dtypes) / FillBitcast (float dtypes), to the slot the
//     tensor did NOT occupy. OptionalChainElement folds the inactive flavor to a no-op.
//   - Where<WHERE_DATA_FORMAT, D0, D1, D2, D0>: the host passes WHERE_DATA_FORMAT mirroring
//     get_sfpu_init_fn(WHERE, a_dtype) — the exact format the legacy BINARY_SFPU_OP baked in.
//   - PackTile<cb_out, OutputLifecycle::Chunked, OperandKind::Block>.

#ifdef FILL_WITH_VALUE_INT
constexpr bool kIsInt = true;
#else
constexpr bool kIsInt = false;
#endif
constexpr bool kIsFloat = !kIsInt;

constexpr DataFormat kWhereDF = DataFormat::WHERE_DATA_FORMAT;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_cond = tt::CBIndex::c_0;
    constexpr auto cb_tensor = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // TTS: tensor (cb_in1) is the TRUE value -> D1, scalar is the FALSE value -> D2.
    // TST: tensor (cb_in1) is the FALSE value -> D2, scalar is the TRUE value -> D1.
#if WHERE_TTS
    constexpr auto kTensorSlot = compute_kernel_lib::Dst::D1;
    constexpr auto kFillSlot = compute_kernel_lib::Dst::D2;
#else  // WHERE_TST
    constexpr auto kTensorSlot = compute_kernel_lib::Dst::D2;
    constexpr auto kFillSlot = compute_kernel_lib::Dst::D1;
#endif

    init_sfpu(cb_cond, cb_out);  // caller-owned BIG init

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(num_tiles, num_tiles_per_cycle),
        // cond -> D0 (block read, init_short for cb_cond).
        compute_kernel_lib::CopyTile<
            cb_cond,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::InputLifecycle::Chunked,
            compute_kernel_lib::CopyTileReconfig::Input,
            compute_kernel_lib::OperandKind::Block>{},
        // tensor -> D1 (TTS) / D2 (TST) (block read, init_short for cb_tensor).
        compute_kernel_lib::CopyTile<
            cb_tensor,
            kTensorSlot,
            compute_kernel_lib::InputLifecycle::Chunked,
            compute_kernel_lib::CopyTileReconfig::Input,
            compute_kernel_lib::OperandKind::Block>{},
        // scalar fill -> the other slot. Inactive flavor folds to a no-op.
        compute_kernel_lib::OptionalChainElement<kIsInt, compute_kernel_lib::FillInt<kWhereDF, kFillSlot>>{
            scalar_value},
        compute_kernel_lib::OptionalChainElement<kIsFloat, compute_kernel_lib::FillBitcast<kFillSlot>>{scalar_value},
        // where(D0, D1, D2) -> D0.
        compute_kernel_lib::Where<
            kWhereDF,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Dst::D1,
            compute_kernel_lib::Dst::D2,
            compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_out,
            compute_kernel_lib::OutputLifecycle::Chunked,
            compute_kernel_lib::PackTileReconfig::None>{});
}
