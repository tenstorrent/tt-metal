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
        bits(DimField::encode(dim) | InputField::encode(InputSpecConfig::encode(input_spec)) | DstField::encode(dst)) {}
    constexpr explicit UnaryBcastConfig(uint32_t encoded) noexcept : bits(encoded) {}

    constexpr BroadcastDim dim() const noexcept { return DimField::decode(bits); }
    constexpr InputSpec input_spec(uint32_t cb_id) const noexcept {
        return InputSpecConfig::decode(InputField::decode(bits), cb_id);
    }
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
    static constexpr InputSpec Input = Config.input_spec(Cb);
    static constexpr Dst DstSlot = Config.dst();
    static constexpr WaitPolicy Wait = Input.wait;
    static constexpr PopPolicy Pop = Input.pop;
    static constexpr InputTileMapping Mapping = Input.mapping;
    static constexpr TileAddressing Addressing = Input.addressing;
    using Base = InputStream;
    using Base::tile_base;

    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT, "UnaryBcast: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(
        is_legal_input_policy_for_mapping(Mapping, Wait, Pop),
        "UnaryBcast: input wait/pop pair is incompatible with input tile mapping");
    static_assert(
        Addressing == TileAddressing::Direct || is_legal_input_policy_with_base(Wait, Pop),
        "UnaryBcast: TileAddressing::Offset requires an upfront, deferred-pop, or caller-managed input pair");
    static_assert(
        Addressing != TileAddressing::Strided || ((Wait == WaitPolicy::None) && (Pop == PopPolicy::None)),
        "UnaryBcast: TileAddressing::Strided requires caller-managed (None, None) input policies");

    static constexpr uint32_t dfb = Cb;
    static constexpr uint32_t dfb_a_id() { return Cb; }
    static constexpr InputSpec a_input() { return Input; }

    static constexpr uint32_t reconfig_srca_dfb = Input.reconfig == DataFormatReconfig::Enabled ? Cb : NO_PREV_DFB;
    static constexpr uint32_t reconfig_srcb_dfb = Input.reconfig == DataFormatReconfig::Enabled ? Cb : NO_PREV_DFB;

    template <class... Args, std::enable_if_t<rejects_constructor_args<UnaryBcastImpl, 1, Args...>(), int> = 0>
    UnaryBcastImpl(Args&&...) noexcept : Base(MissingDataflowBuffer{}) {
        require_bound_dataflow_buffers<rejects_constructor_args<UnaryBcastImpl, 1, Args...>()>();
    }
    ALWI explicit UnaryBcastImpl(DataflowBuffer& in) noexcept : Base(in) { check_bound_dataflow_buffer<Cb>(in); }
    ALWI UnaryBcastImpl(DataflowBuffer& in, uint32_t base) noexcept : Base(in, base) {
        check_bound_dataflow_buffer<Cb>(in);
    }
    ALWI UnaryBcastImpl(DataflowBuffer& in, StridedTileRange range) noexcept : Base(in, range) {
        check_bound_dataflow_buffer<Cb>(in);
    }

    static ALWI void init() {
        constexpr ckernel::BroadcastType bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        ::unary_bcast_init<bt>(Cb);
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        constexpr ckernel::BroadcastType bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Dim));
        const uint32_t in_idx = tile_base_value<Addressing>(tile_base) +
                                detail::input_idx<Mapping, Wait, Pop, Addressing>(i_flat, ht, wt, row_stride);
        ::unary_bcast<bt>(Cb, in_idx, to_u32(DstSlot) + slot_offset);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;
};

template <BroadcastDim Dim, InputSpec Input, OutputSpec Output>
ALWI void unary_bcast(IterationShape shape, DataflowBuffer& in, DataflowBuffer& out) {
    eltwise_chain(shape, UnaryBcast<Dim, Input>{in}, PackTile<Output>{out});
}

}  // namespace compute_kernel_lib
