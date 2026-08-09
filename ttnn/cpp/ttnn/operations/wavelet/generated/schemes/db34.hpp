// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db34_inverse;

struct db34 {
    static constexpr const char* name = "db34";
    static constexpr uint32_t tap_size = 68U;
    static constexpr int32_t delay_even = 17;
    static constexpr int32_t delay_odd = 17;
    static constexpr uint32_t num_steps = 37U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db34.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db34";
    using inverse = db34_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db34::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeef353bU>;
    static_assert(type::k == 1U);
};

template <>
struct db34::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf1aa85fU, 0x3e58e0ebU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf1b360eU, 0x3eef515cU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf30c1c2U, 0x3f210190U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf3af8f2U, 0x3f1a786cU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf598b55U, 0x3f07230cU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf7878c9U, 0x3ec3a270U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf978e7dU, 0x3ed0a5fbU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfbb2ad3U, 0x3efa0a97U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3de4b7b0U, 0x3f065712U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eacfc50U, 0x4123bb06U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdb660bbU, 0x3ce6f452U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1f62f7dU, 0x41c0032bU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd1d6b76U, 0x3cf49bc5U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc20205dcU, 0x41c2b4faU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd29eb59U, 0x3cf3835fU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc20bfb6dU, 0x41de4150U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd1b5a9aU, 0xbb9b39c0U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x431bb5e0U, 0xc05c3c83U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a987b8bU, 0xbc0bf8a6U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x408966bfU, 0x43c73fe5U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb2fe69fU, 0x3a822c66U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc48b923dU, 0x4404c3cdU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb0a41d0U, 0x3a6aebb4U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc49de380U, 0x43ed0135U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb1e40fbU, 0x3a4f89c9U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc4b71271U, 0x43cf0f4aU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb3a6d11U, 0x3a32fd5aU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xc4dc1eb0U, 0x43afc4faU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb667b15U, 0x3a14dd46U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xc50dacf5U, 0x438e2c20U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb9e2741U, 0x39e749f6U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xc55a9dbfU, 0x434f30fbU>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x18983426U, 0x3995e365U>;
    static_assert(type::k == 2U);
};

template <>
struct db34::step<34> {
    using type = StaticStep<StepType::kPredict, 0, 0xc2c985d7U>;
    static_assert(type::k == 1U);
};

template <>
struct db34::step<35> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xca094df3U>;
    static_assert(type::k == 1U);
};

template <>
struct db34::step<36> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb4eea6eeU>;
    static_assert(type::k == 1U);
};

struct db34_inverse {
    static constexpr const char* name = "db34-inverse";
    static constexpr uint32_t tap_size = 68U;
    static constexpr uint32_t num_steps = 37U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db34.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db34_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db34_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xca094df3U>;
    static_assert(type::k == 1U);
};

template <>
struct db34_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb4eea6eeU>;
    static_assert(type::k == 1U);
};

template <>
struct db34_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x42c985d7U>;
    static_assert(type::k == 1U);
};

template <>
struct db34_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x98983426U, 0xb995e365U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x455a9dbfU, 0xc34f30fbU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b9e2741U, 0xb9e749f6U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x450dacf5U, 0xc38e2c20U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b667b15U, 0xba14dd46U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x44dc1eb0U, 0xc3afc4faU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b3a6d11U, 0xba32fd5aU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x44b71271U, 0xc3cf0f4aU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b1e40fbU, 0xba4f89c9U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x449de380U, 0xc3ed0135U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b0a41d0U, 0xba6aebb4U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x448b923dU, 0xc404c3cdU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b2fe69fU, 0xba822c66U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc08966bfU, 0xc3c73fe5U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba987b8bU, 0x3c0bf8a6U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc31bb5e0U, 0x405c3c83U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d1b5a9aU, 0x3b9b39c0U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x420bfb6dU, 0xc1de4150U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d29eb59U, 0xbcf3835fU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x420205dcU, 0xc1c2b4faU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d1d6b76U, 0xbcf49bc5U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x41f62f7dU, 0xc1c0032bU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3db660bbU, 0xbce6f452U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeacfc50U, 0xc123bb06U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbde4b7b0U, 0xbf065712U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fbb2ad3U, 0xbefa0a97U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f978e7dU, 0xbed0a5fbU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f7878c9U, 0xbec3a270U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f598b55U, 0xbf07230cU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f3af8f2U, 0xbf1a786cU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f30c1c2U, 0xbf210190U>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f1b360eU, 0xbeef515cU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f1aa85fU, 0xbe58e0ebU>;
    static_assert(type::k == 2U);
};

template <>
struct db34_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eef353bU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
