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

// where(cond, true_value, false_value): copy cond -> D0, fill true -> D1,
// fill false -> D2, Where(D0,D1,D2 -> D0), pack D0 -> out. The factory used to
// inject this as the SFPU_OP_CHAIN_0 macro, but for WHERE_TSS the op-chain is
// always exactly [WHERE_TSS] (where_tile<DF>(0,1,2,0)), so the macro is removed
// and the explicit `Where` chain element expresses it directly.
//
// `Where`'s DataFormat template arg matches unary_op_utils.cpp:538-545 exactly:
//   INT32 -> Int32, UINT32 -> UInt32, FLOAT32 -> Float32, else -> Float16_b.
// The fills follow the original kernel: int dtypes fill via fill_tile_int<Int32>
// (Int32 for BOTH int32 and uint32), float dtypes fill the float whose bits are
// the packed scalar arg — FillBitcast(bits) == the original fill_tile(*float*),
// without the strict-aliasing type-pun.
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
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);  // true_value (bits)
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);  // false_value (bits)

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);  // caller-owned BIG init

    compute_kernel_lib::eltwise_chain(
        num_tiles,
        // cond -> D0. Single CB read: Streaming (wait 1 / pop 1 per iter), Scalar index.
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        // true_value -> D1 (inactive flavor folds to a FillTileTag no-op).
        compute_kernel_lib::
            OptionalChainElement<kIsInt, compute_kernel_lib::FillInt<DataFormat::Int32, compute_kernel_lib::Dst::D1>>{
                packed_scalar1},
        compute_kernel_lib::
            OptionalChainElement<kIsFloat, compute_kernel_lib::FillBitcast<compute_kernel_lib::Dst::D1>>{
                packed_scalar1},
        // false_value -> D2.
        compute_kernel_lib::
            OptionalChainElement<kIsInt, compute_kernel_lib::FillInt<DataFormat::Int32, compute_kernel_lib::Dst::D2>>{
                packed_scalar2},
        compute_kernel_lib::
            OptionalChainElement<kIsFloat, compute_kernel_lib::FillBitcast<compute_kernel_lib::Dst::D2>>{
                packed_scalar2},
        // where(D0, D1, D2) -> D0.
        compute_kernel_lib::Where<
            kWhereDF,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Dst::D1,
            compute_kernel_lib::Dst::D2,
            compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_output,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::None>{});
}
