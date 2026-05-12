// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace bcast_kernel_detail {
template <ckernel::EltwiseBinaryType T>
struct FpuOpForBinaryType;
template <>
struct FpuOpForBinaryType<ckernel::EltwiseBinaryType::ELWADD> {
    static constexpr auto value = compute_kernel_lib::BinaryFpuOp::Add;
};
template <>
struct FpuOpForBinaryType<ckernel::EltwiseBinaryType::ELWSUB> {
    static constexpr auto value = compute_kernel_lib::BinaryFpuOp::Sub;
};
template <>
struct FpuOpForBinaryType<ckernel::EltwiseBinaryType::ELWMUL> {
    static constexpr auto value = compute_kernel_lib::BinaryFpuOp::Mul;
};
}  // namespace bcast_kernel_detail
constexpr auto FPU_OP = bcast_kernel_detail::FpuOpForBinaryType<ckernel::EltwiseBinaryType::BCAST_LLKOP>::value;

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = B * Ht * Wt;

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

#ifdef BCAST_SCALAR
    using BinElt = BinaryFpu<
        cb_a,
        cb_b,
        cb_out,
        FPU_OP,
        BroadcastDim::Scalar,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPop,
        CopyTilePolicy::NoWaitNoPop,
        CbIndexMode::FirstTile,
        Dst::D0>;
    cb_wait_front(cb_b, 1);
    eltwise_chain(num_tiles, BinElt{}, PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#else
    using BinElt = BinaryFpu<
        cb_a,
        cb_b,
        cb_out,
        FPU_OP,
        BroadcastDim::Scalar,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPop,
        CopyTilePolicy::WaitAndPop,
        CbIndexMode::FirstTile,
        Dst::D0>;
    eltwise_chain(num_tiles, BinElt{}, PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#endif
}
