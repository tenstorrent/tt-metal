// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {

// Wraps the host-defined TERNARY_SFPU_OP_FUNC macro (where/lerp/addcmul/addcdiv).
template <
    compute_kernel_lib::Dst In0 = compute_kernel_lib::Dst::D0,
    compute_kernel_lib::Dst In1 = compute_kernel_lib::Dst::D1,
    compute_kernel_lib::Dst In2 = compute_kernel_lib::Dst::D2,
    compute_kernel_lib::Dst Out = compute_kernel_lib::Dst::D0>
struct TernarySfpuOp : compute_kernel_lib::TernaryOp<TernarySfpuOp<In0, In1, In2, Out>, In0, In1, In2, Out> {
    static ALWI void init() { TERNARY_SFPU_OP_INIT(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        TERNARY_SFPU_OP_FUNC(
            compute_kernel_lib::to_u32(In0) + slot_offset,
            compute_kernel_lib::to_u32(In1) + slot_offset,
            compute_kernel_lib::to_u32(In2) + slot_offset,
            compute_kernel_lib::to_u32(Out) + slot_offset);
    }
};

// Wraps the host-defined FILL_LLK macro (fill_tile / fill_tile_int<Int32> / fill_tile_uint<UInt32>).
// Carries the runtime scalar value as an instance member.
template <compute_kernel_lib::Dst DstSlot>
struct FillLlk : compute_kernel_lib::FillTileTag {
    uint32_t value;
    constexpr explicit FillLlk(uint32_t v) noexcept : value(v) {}
    static ALWI void init() { fill_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
#ifdef FILL_WITH_VALUE_FLOAT
        const auto scalar_val = reinterpret_cast<const float*>(&value);
        FILL_LLK(compute_kernel_lib::to_u32(DstSlot) + slot_offset, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
        FILL_LLK(compute_kernel_lib::to_u32(DstSlot) + slot_offset, value);
#endif
    }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    constexpr bool scalar_is_true = get_compile_time_arg_val(1);           // 1=TST, 0=TTS
    (void)num_tiles_per_cycle;

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_pre_in1, cb_pre_in2, cb_out);

    if constexpr (scalar_is_true) {
        // TST: tensor=false (D2), scalar=true (D1)
        eltwise_chain(
            num_tiles,
            CopyTile<cb_pre_in1, Dst::D0, CopyTilePolicy::WaitAndPop>{},
            CopyTile<cb_pre_in2, Dst::D2, CopyTilePolicy::WaitAndPop>{},
            FillLlk<Dst::D1>{scalar_value},
            TernarySfpuOp<Dst::D0, Dst::D1, Dst::D2, Dst::D0>{},
            PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
    } else {
        // TTS: tensor=true (D1), scalar=false (D2)
        eltwise_chain(
            num_tiles,
            CopyTile<cb_pre_in1, Dst::D0, CopyTilePolicy::WaitAndPop>{},
            CopyTile<cb_pre_in2, Dst::D1, CopyTilePolicy::WaitAndPop>{},
            FillLlk<Dst::D2>{scalar_value},
            TernarySfpuOp<Dst::D0, Dst::D1, Dst::D2, Dst::D0>{},
            PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
    }
}
