// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym20_inverse;

struct sym20 {
    static constexpr const char* name = "sym20";
    static constexpr uint32_t tap_size = 40U;
    static constexpr int32_t delay_even = 10;
    static constexpr int32_t delay_odd = 10;
    static constexpr uint32_t num_steps = 23U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym20.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym20";
    using inverse = sym20_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym20::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbff8c1deU>;
    static_assert(type::k == 1U);
};

template <>
struct sym20::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dbef96cU, 0x3ed04d37U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x419dd438U, 0xbfe343ffU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba737205U, 0xbd477979U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x441cf84bU, 0x41181fb6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb05029eU, 0xbacb83d0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc26602fcU, 0x43f2b360U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbb72c7cU, 0x3c0969d6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2153b75U, 0x4251713fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba30f866U, 0x3b987d1aU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x41c4567dU, 0x40d0a78dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b967978U, 0xbb1d66b2U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x42bcca24U, 0xc2224af0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bef3902U, 0xbbc01fe8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x4481f4c4U, 0xc2b42f21U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb7a4093aU, 0xba7b2a46U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x45bac9f6U, 0x4599ee7eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x37d0d2feU, 0xb7be62c6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc682b176U, 0xc5c762d8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x379128f6U, 0x380d9efcU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20::step<20> {
    using type = StaticStep<StepType::kPredict, 0, 0xc6659311U>;
    static_assert(type::k == 1U);
};

template <>
struct sym20::step<21> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc31ea13fU>;
    static_assert(type::k == 1U);
};

template <>
struct sym20::step<22> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbbce91a4U>;
    static_assert(type::k == 1U);
};

struct sym20_inverse {
    static constexpr const char* name = "sym20-inverse";
    static constexpr uint32_t tap_size = 40U;
    static constexpr uint32_t num_steps = 23U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym20.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym20_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym20_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc31ea13fU>;
    static_assert(type::k == 1U);
};

template <>
struct sym20_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbbce91a5U>;
    static_assert(type::k == 1U);
};

template <>
struct sym20_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x46659311U>;
    static_assert(type::k == 1U);
};

template <>
struct sym20_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb79128f6U, 0xb80d9efcU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4682b176U, 0x45c762d8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb7d0d2feU, 0x37be62c6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc5bac9f6U, 0xc599ee7eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x37a4093aU, 0x3a7b2a46U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc481f4c4U, 0x42b42f21U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbef3902U, 0x3bc01fe8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2bcca24U, 0x42224af0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb967978U, 0x3b1d66b2U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1c4567dU, 0xc0d0a78dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a30f866U, 0xbb987d1aU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x42153b75U, 0xc251713fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bb72c7cU, 0xbc0969d6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x426602fcU, 0xc3f2b360U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b05029eU, 0x3acb83d0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc41cf84bU, 0xc1181fb6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a737205U, 0x3d477979U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc19dd438U, 0x3fe343ffU>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdbef96cU, 0xbed04d37U>;
    static_assert(type::k == 2U);
};

template <>
struct sym20_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ff8c1deU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
