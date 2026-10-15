// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tuple>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/extended.hpp"  // BitwiseAndBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/int.hpp"       // MulIntBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"       // FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"            // Typecast

namespace ckl = compute_kernel_lib;

constexpr uint32_t kExponent = get_arg(args::exponent);
constexpr auto kDataFormat = static_cast<DataFormat>(get_arg(args::data_format));
constexpr bool kIsUint16 = kDataFormat == DataFormat::UInt16;
// UInt16 is widened to UInt32 in the 32-bit Dest, multiplied there, and truncated back on the way out.
constexpr DataFormat kMathFormat = kIsUint16 ? DataFormat::UInt32 : kDataFormat;
constexpr uint32_t kUint16Format = static_cast<uint32_t>(DataFormat::UInt16);
constexpr uint32_t kUint32Format = static_cast<uint32_t>(DataFormat::UInt32);
// Bit below the leading one. The leading one is the copy of x, so x^1 never multiplies.
// x^0 is a fill, and __builtin_clz(0) is undefined.
constexpr int kFirstBit = kExponent == 0 ? -1 : 30 - __builtin_clz(kExponent);
// x^0 and x^1 leave the value in D0. Every larger exponent writes the square into D1.
constexpr bool kResultInBase = kExponent <= 1;
constexpr bool kPackProduct = !kResultInBase;
constexpr bool kFillOnes = kExponent == 0;
constexpr bool kWidenBase = kIsUint16 && kExponent != 0;
constexpr bool kMaskBase = kIsUint16 && kResultInBase;
constexpr bool kMaskProduct = kIsUint16 && !kResultInBase;

constexpr auto kInput =
    ckl::input(dfb::in, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
constexpr auto kOutput =
    ckl::output(dfb::out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled);

// Square the base on the first bit and the running product after that.
template <int Bit>
using SquareAt = ckl::MulIntBinary<
    kMathFormat,
    (Bit == kFirstBit ? ckl::Dst::D0 : ckl::Dst::D1),
    (Bit == kFirstBit ? ckl::Dst::D0 : ckl::Dst::D1),
    ckl::Dst::D1>;

template <int Bit>
using MulByBaseAt = ckl::MulIntBinary<kMathFormat, ckl::Dst::D1, ckl::Dst::D0, ckl::Dst::D1>;

// High bit to low bit. Bits above the leading one are not instantiated.
template <int Bit, class Tuple>
ALWI auto append_pow_bit(Tuple bits) {
    if constexpr (Bit < 0) {
        return bits;
    } else if constexpr (kFirstBit >= 0 && Bit <= kFirstBit && ((kExponent >> Bit) & 1u)) {
        return append_pow_bit<Bit - 1>(std::tuple_cat(bits, std::tuple<SquareAt<Bit>, MulByBaseAt<Bit>>{}));
    } else if constexpr (kFirstBit >= 0 && Bit <= kFirstBit) {
        return append_pow_bit<Bit - 1>(std::tuple_cat(bits, std::tuple<SquareAt<Bit>>{}));
    } else {
        return append_pow_bit<Bit - 1>(bits);
    }
}

template <class... BitOps, std::size_t... BitIndex>
ALWI void run_pow(uint32_t num_tiles, std::tuple<BitOps...> bit_ops, std::index_sequence<BitIndex...>) {
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        ckl::CopyTile<kInput, ckl::Dst::D0>{},
        // x^0 ignores the copied tile. The copy still pops the input buffer.
        ckl::Optional<kFillOnes, ckl::FillInt<kMathFormat, ckl::Dst::D0>>{1u},
        ckl::Optional<kWidenBase, ckl::Typecast<kUint16Format, kUint32Format, ckl::Dst::D0>>{},
        std::get<BitIndex>(bit_ops)...,
        // The UInt32 -> UInt16 typecast saturates, so mask to 16 bits first.
        ckl::Optional<kIsUint16, ckl::FillInt<DataFormat::UInt32, ckl::Dst::D2>>{0xFFFFu},
        ckl::Optional<kMaskBase, ckl::BitwiseAndBinary<DataFormat::UInt32, ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D0>>{},
        ckl::Optional<
            kMaskProduct,
            ckl::BitwiseAndBinary<DataFormat::UInt32, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>>{},
        ckl::Optional<kMaskBase, ckl::Typecast<kUint32Format, kUint16Format, ckl::Dst::D0>>{},
        ckl::Optional<kMaskProduct, ckl::Typecast<kUint32Format, kUint16Format, ckl::Dst::D1>>{},
        ckl::Optional<kResultInBase, ckl::PackTile<kOutput, ckl::Dst::D0>>{},
        ckl::Optional<kPackProduct, ckl::PackTile<kOutput, ckl::Dst::D1>>{});
}

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    const auto bit_ops = append_pow_bit<30>(std::tuple<>{});
    run_pow(num_tiles, bit_ops, std::make_index_sequence<std::tuple_size_v<decltype(bit_ops)>>{});
}
