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

#if defined(INP_INT32) || defined(INP_UINT32)
constexpr bool kIsInt = true;
#else
constexpr bool kIsInt = false;
#endif
constexpr bool kIsFloat = !kIsInt;

#if defined(INP_INT32)
constexpr DataFormat kWhereDF = DataFormat::Int32;
#elif defined(INP_UINT32)
constexpr DataFormat kWhereDF = DataFormat::UInt32;
#elif defined(INP_FLOAT32)
constexpr DataFormat kWhereDF = DataFormat::Float32;
#else
constexpr DataFormat kWhereDF = DataFormat::Float16_b;
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        // cond -> D0. Single CB read: Streaming (wait 1 / pop 1 per iter), Scalar index.
        ckl::CopyTile<
            cb_input,
            ckl::Dst::D0,
            ckl::input(ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{},
        // true_value -> D1 (inactive flavor folds to a FillTileTag no-op).
        // kWhereDF carries main's #48602 fix: Int32 for int32 inputs, UInt32 for uint32 inputs.
        ckl::OptionalChainElement<kIsInt, ckl::FillInt<kWhereDF, ckl::Dst::D1>>{packed_scalar1},
        ckl::OptionalChainElement<kIsFloat, ckl::FillBitcast<ckl::Dst::D1>>{packed_scalar1},
        // false_value -> D2.
        ckl::OptionalChainElement<kIsInt, ckl::FillInt<kWhereDF, ckl::Dst::D2>>{packed_scalar2},
        ckl::OptionalChainElement<kIsFloat, ckl::FillBitcast<ckl::Dst::D2>>{packed_scalar2},
        // where(D0, D1, D2) -> D0.
        ckl::Where<kWhereDF, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<cb_output, ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
}
