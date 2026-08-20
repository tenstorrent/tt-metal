// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/special.hpp"    // Where
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"  // FillBitcast / FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"    // Optional

namespace ckl = compute_kernel_lib;

constexpr auto kWhereDF = static_cast<DataFormat>(get_compile_time_arg_val(0));
constexpr bool kIsInt = kWhereDF == DataFormat::Int32 || kWhereDF == DataFormat::UInt32;
constexpr bool kIsFloat = kWhereDF == DataFormat::Float32 || kWhereDF == DataFormat::Float16_b;
static_assert(kIsInt || kIsFloat, "where_tss supports only Int32, UInt32, Float32, and Float16_b");

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto dfb_input_id = tt::CBIndex::c_0;
    constexpr auto dfb_output_id = tt::CBIndex::c_2;

    compute_kernel_hw_startup(dfb_input_id, dfb_output_id);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        // cond -> D0. Single DFB read: Streaming (wait 1 / pop 1 per iter), Scalar index.
        ckl::CopyTile<
            ckl::input(
                dfb_input_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        // true_value -> D1 (inactive flavor folds to a FillTileTag no-op).
        // kWhereDF carries main's #48602 fix: Int32 for int32 inputs, UInt32 for uint32 inputs.
        ckl::Optional<kIsInt, ckl::FillInt<kWhereDF, ckl::Dst::D1>>{packed_scalar1},
        ckl::Optional<kIsFloat, ckl::FillBitcast<ckl::Dst::D1>>{packed_scalar1},
        // false_value -> D2.
        ckl::Optional<kIsInt, ckl::FillInt<kWhereDF, ckl::Dst::D2>>{packed_scalar2},
        ckl::Optional<kIsFloat, ckl::FillBitcast<ckl::Dst::D2>>{packed_scalar2},
        // where(D0, D1, D2) -> D0.
        ckl::Where<kWhereDF, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_output_id,
            ckl::ReservePolicy::PerTile,
            ckl::PushPolicy::PerTile,
            ckl::DataFormatReconfig::Disabled)>{});
}
