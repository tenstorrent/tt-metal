// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db36_inverse;

struct db36 {
    static constexpr const char* name = "db36";
    static constexpr uint32_t tap_size = 72U;
    static constexpr int32_t delay_even = 18;
    static constexpr int32_t delay_odd = 18;
    static constexpr uint32_t num_steps = 39U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db36.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db36";
    using inverse = db36_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db36::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbedfe496U>;
    static_assert(type::k == 1U);
};

template <>
struct db36::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf117873U, 0x3e50104cU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf139a5dU, 0x3ee93569U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf280730U, 0x3f1eae22U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf307d13U, 0x3f23f38cU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf46fe8cU, 0x3f347200U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf4de6b2U, 0x3f365d14U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf64e1b0U, 0x3f4222b7U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf6cd7f7U, 0x3f267581U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf909f05U, 0x3ecc4df4U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfcc2798U, 0x3e96b535U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf340896U, 0x3ec825d9U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdcd6c8dU, 0x3f7f6108U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf016589U, 0x3fedcff4U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbefc179cU, 0x3ee2786eU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc00cc6f7U, 0x3fd57aefU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf1ae8d0U, 0x3edf60a3U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0183299U, 0x3fcf8dacU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf274618U, 0x3ed76832U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc023a8c3U, 0x40145625U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee8507aU, 0xbc784537U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0127f4dU, 0xbf8d13f4U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbaec52eaU, 0x3e02b006U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc11fe9eaU, 0x40d092b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe2efa4bU, 0x3d9bb1b3U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc16bec77U, 0x40bb269aU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe461defU, 0x3d8ae44dU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc186e6e4U, 0x40a565b0U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe6570b2U, 0x3d72e703U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc19eb28cU, 0x408ed131U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe89bb6eU, 0x3d4e7b20U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1c3e6feU, 0x406de93bU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeb1070eU, 0x3d27445dU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc2063968U, 0x403919e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf08620cU, 0x3cf420f7U>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x1cf947b5U, 0x3ff043baU>;
    static_assert(type::k == 2U);
};

template <>
struct db36::step<36> {
    using type = StaticStep<StepType::kPredict, 0, 0xbc6dc9f0U>;
    static_assert(type::k == 1U);
};

template <>
struct db36::step<37> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4754be9aU>;
    static_assert(type::k == 1U);
};

template <>
struct db36::step<38> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x379a066dU>;
    static_assert(type::k == 1U);
};

struct db36_inverse {
    static constexpr const char* name = "db36-inverse";
    static constexpr uint32_t tap_size = 72U;
    static constexpr uint32_t num_steps = 39U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db36.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db36_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db36_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4754be9bU>;
    static_assert(type::k == 1U);
};

template <>
struct db36_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x379a066dU>;
    static_assert(type::k == 1U);
};

template <>
struct db36_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3c6dc9f0U>;
    static_assert(type::k == 1U);
};

template <>
struct db36_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x9cf947b5U, 0xbff043baU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f08620cU, 0xbcf420f7U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x42063968U, 0xc03919e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eb1070eU, 0xbd27445dU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41c3e6feU, 0xc06de93bU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e89bb6eU, 0xbd4e7b20U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x419eb28cU, 0xc08ed131U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e6570b2U, 0xbd72e703U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4186e6e4U, 0xc0a565b0U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e461defU, 0xbd8ae44dU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x416bec77U, 0xc0bb269aU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e2efa4bU, 0xbd9bb1b3U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x411fe9eaU, 0xc0d092b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3aec52eaU, 0xbe02b006U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40127f4dU, 0x3f8d13f4U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee8507aU, 0x3c784537U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4023a8c3U, 0xc0145625U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f274618U, 0xbed76832U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40183299U, 0xbfcf8dacU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f1ae8d0U, 0xbedf60a3U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x400cc6f7U, 0xbfd57aefU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3efc179cU, 0xbee2786eU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f016589U, 0xbfedcff4U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dcd6c8dU, 0xbf7f6108U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f340896U, 0xbec825d9U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fcc2798U, 0xbe96b535U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f909f05U, 0xbecc4df4U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f6cd7f7U, 0xbf267581U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f64e1b0U, 0xbf4222b7U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f4de6b2U, 0xbf365d14U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f46fe8cU, 0xbf347200U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f307d13U, 0xbf23f38cU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f280730U, 0xbf1eae22U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f139a5dU, 0xbee93569U>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f117873U, 0xbe50104cU>;
    static_assert(type::k == 2U);
};

template <>
struct db36_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3edfe496U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
