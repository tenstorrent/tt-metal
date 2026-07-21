// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/bcast.h"

namespace compute_kernel_lib {

namespace detail {

struct UnaryBcastConfig {
    using DimField = ConfigField<BroadcastDim, first_config_bit, BroadcastDim::Scalar>;
    using InputField = ConfigField<uint16_t, DimField::end, static_cast<uint16_t>(InputSpecConfig::storage_mask)>;
    using DstField = ConfigField<Dst, InputField::end, Dst::D15>;

    uint32_t bits;

    constexpr UnaryBcastConfig(BroadcastDim dim, InputSpec input_spec, Dst dst) noexcept :
        bits(
            DimField::encode(dim) | InputField::encode(InputSpecConfig::encode(input_spec)) |
            DstField::encode(dst)) {}
    constexpr explicit UnaryBcastConfig(uint32_t encoded) noexcept : bits(encoded) {}

    constexpr BroadcastDim dim() const noexcept { return DimField::decode(bits); }
    constexpr InputSpec input_spec() const noexcept { return InputSpecConfig::decode(InputField::decode(bits)); }
    constexpr Dst dst() const noexcept { return DstField::decode(bits); }
};

constexpr uint32_t unary_bcast_config_bits(BroadcastDim dim, InputSpec input_spec, Dst dst) noexcept {
    return UnaryBcastConfig{dim, input_spec, dst}.bits;
}

}  // namespace detail

template <uint32_t Cb, uint32_t ConfigBits>
struct detail::UnaryBcastImpl : InputStream, UnaryBcastTag {
    static constexpr UnaryBcastConfig Config{ConfigBits};
    static constexpr BroadcastDim Dim = Config.dim();
    static constexpr InputSpec Input = Config.input_spec();
    static constexpr Dst DstSlot = Config.dst();
    static constexpr InputLifecycle Policy = Input.lifecycle;
    static constexpr OperandKind IndexMode = Input.index;
    static constexpr TileOffset Offset = Input.offset;
    using Base = InputStream;
    using Base::tile_base;

    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT, "UnaryBcast: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(
        is_legal_kind_lifecycle(IndexMode, Policy), "UnaryBcast: input lifecycle and operand kind are incompatible");
    static_assert(
        detail::valid_policy_mode_v<Policy, IndexMode>,
        "UnaryBcast: Row and Col operand kinds require a non-streaming lifecycle");
    static_assert(
        Offset == TileOffset::Unset || is_legal_input_lifecycle_with_base(Policy),
        "UnaryBcast: TileOffset::Set requires a Bulk-family or CallerManaged lifecycle");

    static constexpr uint32_t dfb = Cb;
    static constexpr uint32_t dfb_a_id() { return Cb; }
    static constexpr InputLifecycle a_policy() { return Policy; }
    static constexpr bool is_upfront =
        is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;

    static constexpr uint32_t reconfig_srca_dfb = Input.reconfig == DataFormatReconfig::Enabled ? Cb : NO_PREV_DFB;
    static constexpr uint32_t reconfig_srcb_dfb = Input.reconfig == DataFormatReconfig::Enabled ? Cb : NO_PREV_DFB;

    constexpr UnaryBcastImpl() noexcept = default;
    constexpr explicit UnaryBcastImpl(uint32_t base) noexcept : Base(base) {}

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

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        constexpr ckernel::BroadcastType bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        const uint32_t in_idx = tile_base_value<Offset>(tile_base) + detail::idx<IndexMode>(i_flat, ht, wt);
        ::unary_bcast<bt>(Cb, in_idx, to_u32(DstSlot) + slot_offset);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;
};

template <BroadcastDim Dim, uint32_t CbIn, uint32_t CbOut, InputSpec Input, OutputSpec Output>
ALWI void unary_bcast(EltwiseShape shape) {
    eltwise_chain(shape, UnaryBcast<Dim, CbIn, Input>{}, PackTile<CbOut, Output>{});
}

}  // namespace compute_kernel_lib
