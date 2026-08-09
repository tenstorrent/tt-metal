// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif17_inverse;

struct coif17 {
    static constexpr const char* name = "coif17";
    static constexpr uint32_t tap_size = 102U;
    static constexpr int32_t delay_even = 25;
    static constexpr int32_t delay_odd = 26;
    static constexpr uint32_t num_steps = 55U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif17.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif17";
    using inverse = coif17_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif17::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfa15e67U>;
    static_assert(type::k == 1U);
};

template <>
struct coif17::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef947eaU, 0x3fb37babU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf29b23aU, 0x3f1435d7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfc917f9U, 0x3fc31db5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf17e569U, 0x3efbf870U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe17930U, 0x3f9a06a9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf20124fU, 0x3ed0eb0eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb0f85dU, 0x3f91fc09U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbed231c5U, 0x3edfa014U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf925c18U, 0x3f7c9999U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbedcd1a4U, 0x3e92282aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf5c9429U, 0x3f4b3971U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8b10dfU, 0x3e7152c5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3b9e36U, 0x3ee957b5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe270626U, 0x3e11a386U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbecec79eU, 0x3e4ca7beU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd97d922U, 0x3cea2a39U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd9e9deeU, 0xbd83455eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3cc1ea2fU, 0xbda86af5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e61cbefU, 0xbe9fd12aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3de858a0U, 0xbe4b7271U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f02a91eU, 0xbf0a46ebU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e558c9dU, 0xbe965536U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f2e30d8U, 0xbf64d4f8U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ead775dU, 0xbea5d6c6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f5fc6fdU, 0xbf91aa30U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eb4b3d3U, 0xbf00ee0dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f92af37U, 0xbf9603d7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f007b8eU, 0xbeee6d71U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f979534U, 0xbfb652afU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee6194eU, 0xbf2d307cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f97987dU, 0xbfe58c9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f00cffaU, 0xbf202ceeU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fbf3269U, 0xc26bc117U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c8afc67U, 0x3aded4d1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc41265ccU, 0xc3d856cbU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x3b163bf3U, 0xbb609aa0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x43915beeU, 0xc4003502U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3aff491dU, 0xbb789958U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4383c50cU, 0xc40ddb2bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ae6faa8U, 0xbb8abfceU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0x436c2a11U, 0xc4200cf2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<42> {
    using type = StaticStep<StepType::kPredict, -1, 0x3accbc26U, 0xbb9e6739U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<43> {
    using type = StaticStep<StepType::kUpdate, 0, 0x434edd4dU, 0xc4393378U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<44> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ab0ee9aU, 0xbbba4d5cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<45> {
    using type = StaticStep<StepType::kUpdate, 0, 0x432fe2e4U, 0xc45e648aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<46> {
    using type = StaticStep<StepType::kPredict, -1, 0x3a9357cbU, 0xbbe615ecU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<47> {
    using type = StaticStep<StepType::kUpdate, 0, 0x430e6aa2U, 0xc48f0392U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<48> {
    using type = StaticStep<StepType::kPredict, -1, 0x3a651fdfU, 0xbc1dc44fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<49> {
    using type = StaticStep<StepType::kUpdate, 0, 0x42cfb2edU, 0xc4dc8b56U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<50> {
    using type = StaticStep<StepType::kPredict, -1, 0x3a1493f0U, 0xbca21e34U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17::step<51> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif17::step<52> {
    using type = StaticStep<StepType::kPredict, 0, 0x424a1fd9U>;
    static_assert(type::k == 1U);
};

template <>
struct coif17::step<53> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x37f0463cU>;
    static_assert(type::k == 1U);
};

template <>
struct coif17::step<54> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc70860a0U>;
    static_assert(type::k == 1U);
};

struct coif17_inverse {
    static constexpr const char* name = "coif17-inverse";
    static constexpr uint32_t tap_size = 102U;
    static constexpr uint32_t num_steps = 55U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif17.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif17_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif17_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb7f0463bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif17_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x470860a0U>;
    static_assert(type::k == 1U);
};

template <>
struct coif17_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc24a1fd9U>;
    static_assert(type::k == 1U);
};

template <>
struct coif17_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif17_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xba1493f0U, 0x3ca21e34U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc2cfb2edU, 0x44dc8b56U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xba651fdfU, 0x3c1dc44fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc30e6aa2U, 0x448f0392U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xba9357cbU, 0x3be615ecU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc32fe2e4U, 0x445e648aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbab0ee9aU, 0x3bba4d5cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc34edd4dU, 0x44393378U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbaccbc26U, 0x3b9e6739U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc36c2a11U, 0x44200cf2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbae6faa8U, 0x3b8abfceU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc383c50cU, 0x440ddb2bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbaff491dU, 0x3b789958U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc3915beeU, 0x44003502U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbb163bf3U, 0x3b609aa0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x441265ccU, 0x43d856cbU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc8afc67U, 0xbaded4d1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfbf3269U, 0x426bc117U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf00cffaU, 0x3f202ceeU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf97987dU, 0x3fe58c9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee6194eU, 0x3f2d307cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf979534U, 0x3fb652afU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf007b8eU, 0x3eee6d71U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf92af37U, 0x3f9603d7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeb4b3d3U, 0x3f00ee0dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf5fc6fdU, 0x3f91aa30U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbead775dU, 0x3ea5d6c6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf2e30d8U, 0x3f64d4f8U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe558c9dU, 0x3e965536U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf02a91eU, 0x3f0a46ebU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xbde858a0U, 0x3e4b7271U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe61cbefU, 0x3e9fd12aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcc1ea2fU, 0x3da86af5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d9e9deeU, 0x3d83455eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d97d922U, 0xbcea2a39U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ecec79eU, 0xbe4ca7beU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e270626U, 0xbe11a386U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3b9e36U, 0xbee957b5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<42> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8b10dfU, 0xbe7152c5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<43> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f5c9429U, 0xbf4b3971U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<44> {
    using type = StaticStep<StepType::kPredict, -1, 0x3edcd1a4U, 0xbe92282aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<45> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f925c18U, 0xbf7c9999U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<46> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ed231c5U, 0xbedfa014U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<47> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb0f85dU, 0xbf91fc09U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<48> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f20124fU, 0xbed0eb0eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<49> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe17930U, 0xbf9a06a9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<50> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f17e569U, 0xbefbf870U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<51> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fc917f9U, 0xbfc31db5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<52> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f29b23aU, 0xbf1435d7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<53> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef947eaU, 0xbfb37babU>;
    static_assert(type::k == 2U);
};

template <>
struct coif17_inverse::step<54> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fa15e67U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
