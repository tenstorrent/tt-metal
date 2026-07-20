// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/bcast.h"

namespace compute_kernel_lib {

template <BroadcastDim Dim, uint32_t Cb, InputLifecycle Policy, UnaryBcastReconfig Reconfig, Dst DstSlot>
struct UnaryBcast
    : InputStream<Cb, detail::InputSpecConfig::encode(input(Policy, OperandKind::Block, TileOffset::Unset))>,
      UnaryBcastTag {
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT, "UnaryBcast: DEST slot exceeds DEST_AUTO_LIMIT");

    static constexpr uint32_t dfb_a_id() { return Cb; }
    static constexpr InputLifecycle a_policy() { return Policy; }
    static constexpr bool is_upfront =
        is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;

    static constexpr uint32_t reconfig_srca_dfb = (Reconfig == UnaryBcastReconfig::Input) ? Cb : NO_PREV_DFB;
    static constexpr uint32_t reconfig_srcb_dfb = (Reconfig == UnaryBcastReconfig::Input) ? Cb : NO_PREV_DFB;

    static ALWI void init() {
        constexpr ckernel::BroadcastType bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
        const std::uint32_t dst_format = get_operand_dst_format(Cb);
#ifndef ARCH_QUASAR
        const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                           (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                           (dst_format == (std::uint32_t)DataFormat::Int32);
        if (enable_unpack_to_dest) {
            UNPACK((llk_unpack_A_init<bt, false, ckernel::EltwiseBinaryReuseDestType::NONE, true>(false, false, Cb)));
            MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::A2D, DST_ACCUM_MODE, bt>(Cb)));
        } else {
            UNPACK((llk_unpack_A_init<bt, false, ckernel::EltwiseBinaryReuseDestType::NONE, false>(false, false, Cb)));
            MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, DST_ACCUM_MODE, bt>(Cb)));
        }
#else
        const bool enable_unpack_to_dest =
            (dst_format == (std::uint32_t)DataFormat::Float32) || (dst_format == (std::uint32_t)DataFormat::Int32);
        if (enable_unpack_to_dest) {
            ASSERT(false);
            UNPACK((llk_unpack_A_init<false, true>(Cb)));
            MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::A2D, true>(Cb)));
        } else {
            UNPACK((llk_unpack_A_init<false, false>(Cb)));
            MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, false>(Cb)));
        }
#endif
#endif
    }

    ALWI void exec(uint32_t, uint32_t, uint32_t, uint32_t slot_offset) const {
        constexpr ckernel::BroadcastType bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        ::unary_bcast<bt>(Cb, 0, to_u32(DstSlot) + slot_offset);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    ALWI void wait_per_row() const {}
    ALWI void pop_per_row() const {}
};

template <
    BroadcastDim Dim,
    uint32_t CbIn,
    uint32_t CbOut,
    InputLifecycle Lifecycle,
    OutputSpec Output,
    UnaryBcastReconfig Reconfig>
ALWI void unary_bcast(EltwiseShape shape) {
    eltwise_chain(shape, UnaryBcast<Dim, CbIn, Lifecycle, Reconfig>{}, PackTile<CbOut, Output>{});
}

}  // namespace compute_kernel_lib
