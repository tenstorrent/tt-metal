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

namespace ckl = compute_kernel_lib;

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

#if WHERE_TTS
    constexpr auto kTensorSlot = ckl::Dst::D1;
    constexpr auto kFillSlot = ckl::Dst::D2;
#else
    constexpr auto kTensorSlot = ckl::Dst::D2;
    constexpr auto kFillSlot = ckl::Dst::D1;
#endif

    init_sfpu(cb_cond, cb_out);

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles, num_tiles_per_cycle),
        // cond -> D0 (block read, init_short for cb_cond).
        ckl::CopyTile<cb_cond, ckl::Dst::D0, ckl::input(ckl::InputLifecycle::Chunked, ckl::OperandKind::Block)>{},
        // tensor -> D1 (TTS) / D2 (TST) (block read, init_short for cb_tensor).
        ckl::CopyTile<cb_tensor, kTensorSlot, ckl::input(ckl::InputLifecycle::Chunked, ckl::OperandKind::Block)>{},
        // scalar fill -> the other slot. Inactive flavor folds to a no-op.
        ckl::OptionalChainElement<kIsInt, ckl::FillInt<kWhereDF, kFillSlot>>{scalar_value},
        ckl::OptionalChainElement<kIsFloat, ckl::FillBitcast<kFillSlot>>{scalar_value},
        // where(D0, D1, D2) -> D0.
        ckl::Where<kWhereDF, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<cb_out, ckl::output(ckl::OutputLifecycle::Chunked, ckl::DataFormatReconfig::Disabled)>{});
}
