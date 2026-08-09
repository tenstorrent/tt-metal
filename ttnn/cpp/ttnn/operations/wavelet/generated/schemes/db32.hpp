// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db32_inverse;

struct db32 {
    static constexpr const char* name = "db32";
    static constexpr uint32_t tap_size = 64U;
    static constexpr int32_t delay_even = 16;
    static constexpr int32_t delay_odd = 16;
    static constexpr uint32_t num_steps = 35U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db32.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db32";
    using inverse = db32_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db32::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeff1cc2U>;
    static_assert(type::k == 1U);
};

template <>
struct db32::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf254db4U, 0x3e271d47U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf29ca58U, 0x3eb6b6a2U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf447325U, 0x3ee9eff5U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf54ebdeU, 0x3ee9530cU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf71948aU, 0x3f031dffU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf66b785U, 0x3f0d326dU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbecdf824U, 0x3f281b74U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe471472U, 0x3f898a54U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef3a3faU, 0x3f515fe7U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf6facd8U, 0x3f504d0cU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf85c179U, 0x3f55d4efU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf8c1467U, 0x3f588036U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf926397U, 0x3f5988afU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf9828cfU, 0x3f5ceaf3U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf9aa1baU, 0x40ff3f20U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe011fd3U, 0xb9a4f5f0U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc2a294a2U, 0xc2909930U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xb884ff11U, 0x3b9f0db1U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc37945d5U, 0x43571e88U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbba87a6eU, 0x3b2c4f4fU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc3d4b500U, 0x434225deU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbbbefe80U, 0x3b1a0abeU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc3f3a06cU, 0x432b904fU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbbddb3fdU, 0x3b06802fU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc40fb319U, 0x4313cd20U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc057e3dU, 0x3ae40800U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc431f338U, 0x42f57741U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc2c209fU, 0x3ab8243bU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc4749edfU, 0x42be5edfU>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc8504faU, 0x3a85f452U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x237494deU, 0x42765706U>;
    static_assert(type::k == 2U);
};

template <>
struct db32::step<32> {
    using type = StaticStep<StepType::kPredict, 0, 0xba021544U>;
    static_assert(type::k == 1U);
};

template <>
struct db32::step<33> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x451a9664U>;
    static_assert(type::k == 1U);
};

template <>
struct db32::step<34> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x39d3f87bU>;
    static_assert(type::k == 1U);
};

struct db32_inverse {
    static constexpr const char* name = "db32-inverse";
    static constexpr uint32_t tap_size = 64U;
    static constexpr uint32_t num_steps = 35U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db32.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db32_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db32_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x451a9664U>;
    static_assert(type::k == 1U);
};

template <>
struct db32_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x39d3f87aU>;
    static_assert(type::k == 1U);
};

template <>
struct db32_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3a021544U>;
    static_assert(type::k == 1U);
};

template <>
struct db32_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa37494deU, 0xc2765706U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c8504faU, 0xba85f452U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x44749edfU, 0xc2be5edfU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c2c209fU, 0xbab8243bU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4431f338U, 0xc2f57741U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c057e3dU, 0xbae40800U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x440fb319U, 0xc313cd20U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3bddb3fdU, 0xbb06802fU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x43f3a06cU, 0xc32b904fU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3bbefe80U, 0xbb1a0abeU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x43d4b500U, 0xc34225deU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ba87a6eU, 0xbb2c4f4fU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x437945d5U, 0xc3571e88U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3884ff11U, 0xbb9f0db1U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x42a294a2U, 0x42909930U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e011fd3U, 0x39a4f5f0U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f9aa1baU, 0xc0ff3f20U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f9828cfU, 0xbf5ceaf3U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f926397U, 0xbf5988afU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f8c1467U, 0xbf588036U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f85c179U, 0xbf55d4efU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f6facd8U, 0xbf504d0cU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef3a3faU, 0xbf515fe7U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e471472U, 0xbf898a54U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ecdf824U, 0xbf281b74U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f66b785U, 0xbf0d326dU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f71948aU, 0xbf031dffU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f54ebdeU, 0xbee9530cU>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f447325U, 0xbee9eff5U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f29ca58U, 0xbeb6b6a2U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f254db4U, 0xbe271d47U>;
    static_assert(type::k == 2U);
};

template <>
struct db32_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eff1cc2U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
