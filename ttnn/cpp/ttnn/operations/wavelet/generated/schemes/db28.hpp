// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db28_inverse;

struct db28 {
    static constexpr const char* name = "db28";
    static constexpr uint32_t tap_size = 56U;
    static constexpr int32_t delay_even = 14;
    static constexpr int32_t delay_odd = 14;
    static constexpr uint32_t num_steps = 31U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db28.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db28";
    using inverse = db28_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db28::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf0909a9U>;
    static_assert(type::k == 1U);
};

template <>
struct db28::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf164d5eU, 0x3e1bd4ccU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbec5c36bU, 0x3eb9b0feU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbebdded0U, 0x3f0c0e54U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf02ebfeU, 0x3f0fcb14U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3ecab2U, 0x3f28bb9fU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf4d538aU, 0x3f2bde78U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf704deeU, 0x3f4563c6U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf714a6cU, 0x3f428906U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf895679U, 0x3f566475U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf886631U, 0x3f2e9b86U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb126d3U, 0x3e5addeaU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf1d60f4U, 0x3e2a3f74U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc59e096U, 0x3f27c31fU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf90f3d0U, 0x3fba5e73U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf36e1daU, 0x3eebafcaU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0163130U, 0x3fb0b0dbU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf4d039eU, 0x3ed96d02U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc029a7a3U, 0x3f9faddfU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf6ab8f6U, 0x3ec11a9bU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0452f1bU, 0x3f8b991eU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8acaf5U, 0x3ea62dd6U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc06e3ae9U, 0x3f6c17c5U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfac809eU, 0x3e898c35U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc09a2dc4U, 0x3f3df4eeU>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfee0e1eU, 0x3e548852U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0ef376cU, 0x3f09a616U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x23019701U, 0x3e08fb03U>;
    static_assert(type::k == 2U);
};

template <>
struct db28::step<28> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe852b40U>;
    static_assert(type::k == 1U);
};

template <>
struct db28::step<29> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4658bd9eU>;
    static_assert(type::k == 1U);
};

template <>
struct db28::step<30> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x38972f6eU>;
    static_assert(type::k == 1U);
};

struct db28_inverse {
    static constexpr const char* name = "db28-inverse";
    static constexpr uint32_t tap_size = 56U;
    static constexpr uint32_t num_steps = 31U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db28.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db28_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db28_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4658bd9eU>;
    static_assert(type::k == 1U);
};

template <>
struct db28_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x38972f6eU>;
    static_assert(type::k == 1U);
};

template <>
struct db28_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e852b40U>;
    static_assert(type::k == 1U);
};

template <>
struct db28_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa3019701U, 0xbe08fb03U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40ef376cU, 0xbf09a616U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fee0e1eU, 0xbe548852U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x409a2dc4U, 0xbf3df4eeU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fac809eU, 0xbe898c35U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x406e3ae9U, 0xbf6c17c5U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8acaf5U, 0xbea62dd6U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x40452f1bU, 0xbf8b991eU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f6ab8f6U, 0xbec11a9bU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x4029a7a3U, 0xbf9faddfU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f4d039eU, 0xbed96d02U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x40163130U, 0xbfb0b0dbU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f36e1daU, 0xbeebafcaU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f90f3d0U, 0xbfba5e73U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c59e096U, 0xbf27c31fU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f1d60f4U, 0xbe2a3f74U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb126d3U, 0xbe5addeaU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f886631U, 0xbf2e9b86U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f895679U, 0xbf566475U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f714a6cU, 0xbf428906U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f704deeU, 0xbf4563c6U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f4d538aU, 0xbf2bde78U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3ecab2U, 0xbf28bb9fU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f02ebfeU, 0xbf0fcb14U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ebdded0U, 0xbf0c0e54U>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ec5c36bU, 0xbeb9b0feU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f164d5eU, 0xbe1bd4ccU>;
    static_assert(type::k == 2U);
};

template <>
struct db28_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f0909a9U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
