// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db30_inverse;

struct db30 {
    static constexpr const char* name = "db30";
    static constexpr uint32_t tap_size = 60U;
    static constexpr int32_t delay_even = 15;
    static constexpr int32_t delay_odd = 15;
    static constexpr uint32_t num_steps = 33U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db30.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db30";
    using inverse = db30_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db30::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf08bdb7U>;
    static_assert(type::k == 1U);
};

template <>
struct db30::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf31bf73U, 0x3e11f70dU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf38f5f1U, 0x3ea8d73fU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf5dd3cbU, 0x3eea0947U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbffa09deU, 0x3ef8d706U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cd24a78U, 0x3eb5d95fU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fc30259U, 0x40806b28U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe24a191U, 0x3dcd08a8U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0d63d05U, 0x40ad4a33U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe0ea3c7U, 0x3de700a5U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0ecb7ecU, 0x40bacee7U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe1e9a33U, 0x3dde5a06U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc10d731bU, 0x3ff63e7aU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf0ab3deU, 0x3c933259U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x40c200caU, 0x3fb2bc7cU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3abd0c8cU, 0xbe3d42d2U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x407f9ed4U, 0x41fd07aaU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd0a7e71U, 0x3c9f17f6U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc261c013U, 0x41f781fcU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd138171U, 0x3c91053eU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc27f50a7U, 0x41de1720U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd28f906U, 0x3c80564fU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2944fbaU, 0x41c1ec64U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd4799a4U, 0x3c5cf0b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2b2e505U, 0x41a42b0fU>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd779cd8U, 0x3c372b57U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2e71782U, 0x418455f1U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdaa814dU, 0x3c0dcbe4U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xc332e95dU, 0x41402e94U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x1eb19bf8U, 0x3bb726e5U>;
    static_assert(type::k == 2U);
};

template <>
struct db30::step<30> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0ba4c2dU>;
    static_assert(type::k == 1U);
};

template <>
struct db30::step<31> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc8018997U>;
    static_assert(type::k == 1U);
};

template <>
struct db30::step<32> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb6fcf62aU>;
    static_assert(type::k == 1U);
};

struct db30_inverse {
    static constexpr const char* name = "db30-inverse";
    static constexpr uint32_t tap_size = 60U;
    static constexpr uint32_t num_steps = 33U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db30.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db30_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db30_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc8018997U>;
    static_assert(type::k == 1U);
};

template <>
struct db30_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb6fcf62aU>;
    static_assert(type::k == 1U);
};

template <>
struct db30_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x40ba4c2dU>;
    static_assert(type::k == 1U);
};

template <>
struct db30_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x9eb19bf8U, 0xbbb726e5U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4332e95dU, 0xc1402e94U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3daa814dU, 0xbc0dcbe4U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x42e71782U, 0xc18455f1U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d779cd8U, 0xbc372b57U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x42b2e505U, 0xc1a42b0fU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d4799a4U, 0xbc5cf0b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x42944fbaU, 0xc1c1ec64U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d28f906U, 0xbc80564fU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x427f50a7U, 0xc1de1720U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d138171U, 0xbc91053eU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x4261c013U, 0xc1f781fcU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d0a7e71U, 0xbc9f17f6U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc07f9ed4U, 0xc1fd07aaU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbabd0c8cU, 0x3e3d42d2U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0c200caU, 0xbfb2bc7cU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f0ab3deU, 0xbc933259U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x410d731bU, 0xbff63e7aU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e1e9a33U, 0xbdde5a06U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x40ecb7ecU, 0xc0bacee7U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e0ea3c7U, 0xbde700a5U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x40d63d05U, 0xc0ad4a33U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e24a191U, 0xbdcd08a8U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfc30259U, 0xc0806b28U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcd24a78U, 0xbeb5d95fU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ffa09deU, 0xbef8d706U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f5dd3cbU, 0xbeea0947U>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f38f5f1U, 0xbea8d73fU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f31bf73U, 0xbe11f70dU>;
    static_assert(type::k == 2U);
};

template <>
struct db30_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f08bdb7U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
